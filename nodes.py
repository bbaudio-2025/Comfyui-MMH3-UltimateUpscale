"""MMH3 Ultimate Upscale - one node for the full latent re-enhancement loop.

Pipeline (auto, no graph wiring):
    input AV latent
      -> temporal split (outer loop)
      ->   latent upscale  (H3 3D upscaler, per chunk, video only)
      ->   spatial split   (inner loop)
      ->   per-tile sampling with preview
      ->   spatial stitch
      -> temporal stitch
      -> output AV latent

Helpers (frame/token mapping, re-anchoring, spatial tiling, seam blending,
stitching) are self-contained copies of the logic used by the
Comfyui-MiniMax-H3-LatentSplit project so this plugin has no dependency on it.

Frame/token mapping mirrors comfy.ldm.minimax.model:
  * video latent token k covers FRAME_PER_TOKEN[k % 5] = (1, 4, 4, 4, 4) pixel
    frames (periodic grid, 17 frames per 5 tokens)
  * audio latent frames run at FRAME_RESCALE = 5/3 per pixel frame (40 vs 24 Hz)

The H3 3D upscaler inference code (model classes, loading, normalization stats)
is copied from the Comfyui_Minimax_h3_latent_Upscaler plugin so it works with
the minimax_h3_latent_upscaler_3d checkpoints directly.
"""

import glob
import math
import os
import re
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
from comfy_api.latest import io

try:
    import comfy_extras.nodes_lt as _ltx_nodes
except Exception:
    _ltx_nodes = None

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

H3_UPSCALE_PARAM = io.Custom("H3_UPSCALE_PARAM")
H3_TEMPORAL_PARAM = io.Custom("H3_TEMPORAL_PARAM")
H3_SPATIAL_PARAM = io.Custom("H3_SPATIAL_PARAM")

# Spatial compression factor of the Minimax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16

# ---------------------------------------------------------------------------
# frame <-> token helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def frames_for_tokens(n):
    """Pixel frames covered by the first `n` video latent tokens."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f):
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def audio_range(f0, f1):
    """Audio latent token range [a0, a1) for the pixel-frame span [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def compute_segments(tv, chunk_length, overlap):
    """Per-chunk (video_token_start, frame_start, video_token_end, frame_end).

    Same rules as the Split node: every boundary is snapped to a keyframe token
    (index % 5 == 0), the realized overlap is a whole number of 17-frame grid
    steps, and the last chunk always ends on the exact total frame count.
    """
    frame_count = frames_for_tokens(tv)
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_length <= overlap:
        raise ValueError("overlap must be smaller than chunk_length")

    hop = chunk_length - overlap
    bounds = []
    prev_end_k = 0
    i = 0
    while True:
        s = i * hop
        e = min(s + chunk_length, frame_count)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = snap_frame_boundary(s, tv, phase=5)
            if k0 > prev_end_k:
                k0, f0 = prev_end_k, frames_for_tokens(prev_end_k)
        if e >= frame_count:
            k1, f1 = tv, frame_count
        else:
            k1, f1 = snap_frame_boundary(e, tv, phase=5)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_for_tokens(k1)
            if k1 >= tv:
                k1, f1 = tv, frame_count
        bounds.append((k0, f0, k1, f1))
        if k1 >= tv:
            break
        prev_end_k = k1
        i += 1
    return bounds, frame_count


def snap_frame_boundary(f, max_tokens, phase=None):
    """Nearest video-token boundary to pixel frame f (optionally on a phase grid)."""
    step = phase if phase is not None else 1
    best_k, best_f, best_d = 0, 0, f
    for k in range(0, max_tokens + 1, step):
        acc = frames_for_tokens(k)
        d = abs(acc - f)
        if d < best_d:
            best_k, best_f, best_d = k, acc, d
    return best_k, best_f


def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)


# ---------------------------------------------------------------------------
# spatial tiling helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def _grid_1d(size, tile, ol, min_tile):
    """Tile origins/dims for one axis plus per-seam overlaps.

    If the leftover edge tile would be smaller than min_tile, the last origin is
    pulled left until the edge reaches min_tile; the extra overlap that creates
    is reported per-seam so stitching blends over its full width."""
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl


def compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    """Tile a latent of size (h, w) with tiles (th, tw) and overlap (ol_h, ol_w).

    Returns (row_offsets, col_offsets, true_row_dims, true_col_dims,
    row_overlaps, col_overlaps) in latent units. Horizontal and vertical
    overlaps are independent. min_th/min_tw (0 = disabled) force the leftover
    edge tile to at least that size when possible, growing the seam overlap."""
    if th <= 0 or tw <= 0:
        raise ValueError("tile dimensions must be positive")
    if ol_h >= th or ol_w >= tw:
        raise ValueError("overlap must be smaller than the tile size")
    if min_th < 0 or min_tw < 0:
        raise ValueError("minimum tile size must be non-negative")
    if min_th > th or min_tw > tw:
        raise ValueError("minimum tile size must not exceed the tile size")
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def spatial_fade_mask(tile_h, tile_w, ol_h, ol_w, done_top, done_left, fade_h=0, fade_w=0):
    """Per-tile video noise mask [tile_h, tile_w]: 1 = re-sample freely, 0 = frozen.

    Every tile is sampled at its true extent (no padding), so the mask only
    freezes the overlap strips shared with an already-processed neighbor
    (done_top / done_left). Each overlap strip splits into a FROZEN segment on
    the seam side (mask = 0, keeps the neighbour's content) and a FADE segment
    on the interior side (mask rises 0 -> 1 toward the tile interior).
    fade_width/fade_height is the FADE segment length; the frozen segment takes
    the rest of the overlap strip (ol - fade). 0 (default) = whole strip
    frozen. The two axes use independent fade widths."""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)
    if done_left and ol_w > 0:
        if fade_w == 0:
            mask[:, :ol_w] = 0.0
        else:
            f = min(fade_w, ol_w)
            frozen_w = ol_w - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:, :frozen_w] = 0.0
            mask[:, frozen_w:ol_w] = torch.minimum(mask[:, frozen_w:ol_w], w[None, :])
    if done_top and ol_h > 0:
        if fade_h == 0:
            mask[:ol_h, :] = 0.0
        else:
            f = min(fade_h, ol_h)
            frozen_h = ol_h - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:frozen_h, :] = 0.0
            mask[frozen_h:ol_h, :] = torch.minimum(mask[frozen_h:ol_h, :], w[:, None])
    return mask


def blend_weights(t, overlap_blend, overlap_mode):
    """Weight given to the NEW tile's content across an overlap band.

    t runs 0..1 from the done-seam toward the tile interior. overlap_mode 'later'
    hands the band to the new tile; 'earlier' to the accumulated content.
    overlap_blend selects the transition shape."""
    if overlap_blend == "overwrite":
        return torch.ones_like(t) if overlap_mode == "later" else torch.zeros_like(t)
    if overlap_blend == "midpoint":
        step = (t >= 0.5).to(t.dtype)
    elif overlap_blend == "smoothstep":
        step = t * t * (3.0 - 2.0 * t)
    else:
        step = t
    if overlap_mode == "earlier":
        return step
    return 1.0 - step


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    """Spatially crop every keyframe's video latent to a tile of the source frame.

    Keyframes whose latent already matches the source spatial size are cropped to
    the tile's latent region. If a keyframe is at a DIFFERENT spatial scale (e.g.
    the source was latent-upscaled before tiling, or a different VAE/resolution
    produced the conditioning), it is resized to the source spatial size first so
    the cropped keyframe exactly matches the tile's row count - otherwise the
    model's cond/video row broadcast (`all_video_rows[~img_update] = cond_video_rows`)
    fails with a shape mismatch. Audio keyframes are untouched (audio is not spatial)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    kh, kw = lt.shape[3], lt.shape[4]
                    if kh == src_h and kw == src_w:
                        nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    else:
                        # Keyframe latent is at a different spatial scale than the
                        # tile source. Resize it to (src_h, src_w) so the crop
                        # produces a keyframe that matches the tile dimensions.
                        B, C, T, H, W = lt.shape
                        lt_r = torch.nn.functional.interpolate(
                            lt.to(torch.float32).reshape(B * T, C, H, W),
                            size=(src_h, src_w), mode="bilinear", align_corners=False,
                        ).reshape(B, C, T, src_h, src_w)
                        nkf["latent"] = lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    cropped.append(nkf)
                else:
                    cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def trim_keyframe(kf, f0, f1):
    """Copy a keyframe cut to the portion fully inside pixel frames [f0, f1)."""
    idx = kf["resolved_frame_index"]
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    has_v = latent is not None
    has_a = audio_latent is not None

    if not has_v and not has_a:
        if idx < f0 or idx >= f1:
            return None
        return {"resolved_frame_index": idx - f0}

    out = {}
    if has_v:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if has_a:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    if "latent" not in out and "audio_latent" not in out:
        return None
    return out


def reanchor_conditioning(cond, f0, f1, spatial=None):
    """Cut/re-anchor minimax_keyframes to the pixel-frame segment [f0, f1).

    When `spatial` (latent_h, latent_w) is given, keyframe video latents whose
    spatial size differs are resized to it (bilinear)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [trim_keyframe(kf, f0, f1) for kf in kfs]
            trimmed = [kf for kf in trimmed if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear", align_corners=False
                            ).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def anchor_conditioning(cond, prev_video, f0, strength):
    """Replace the frame-0 keyframe with the previous chunk's re-sampled frame.

    Mirrors the 'Anchor MiniMax H3 Latent' node: keyframes are frozen rows in
    the H3 packed sequence, so pinning frame 0 to the content the previous chunk
    ended with removes the detail mismatch at the seam. `strength` becomes
    minimax_visual_cond_noise_aug (0.999 = model default)."""
    t = tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("previous result does not extend to the current segment's start frame")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def normalize_minimax_refs(cond):
    """Make minimax_refs blocks SELF-CONSISTENT for the H3 packed layout.

    The model counts frozen rows from two paths that must agree exactly:
      * PackedLayout reserves ref rows from each block's METADATA
        (latent_h/latent_w/latent_t) and does NOT check whether the block
        actually carries a "latent";
      * cond_video_latents delivers rows from blocks where "latent" EXISTS,
        sized by the latent's real shape.
    If an upstream node/version writes metadata that disagrees with the latent
    (or emits a visual block without a latent), layout reserves one or more
    phantom frames and sampling crashes with
    'all_video_rows[~img_update] = cond_video_rows: shape mismatch'.
    Here we drop visual blocks without a latent and rewrite the metadata from
    the real latent shape, so both paths can never diverge."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        refs = nd.get("minimax_refs")
        if refs:
            fixed = []
            for blk in refs:
                nblk = dict(blk)
                lt = nblk.get("latent")
                if lt is None:
                    # visual block without a latent would reserve phantom rows
                    if nblk.get("kind") in ("image", "video", "video_audio"):
                        continue
                    fixed.append(nblk)
                    continue
                nblk["latent_h"] = int(lt.shape[3])
                nblk["latent_w"] = int(lt.shape[4])
                if nblk.get("kind") in ("video", "video_audio"):
                    nblk["latent_t"] = int(lt.shape[2])
                fixed.append(nblk)
            nd["minimax_refs"] = fixed
        out.append([tensor, nd])
    return out


def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    w = w.view(shape)
    return a + (b - a) * w


# ---------------------------------------------------------------------------
# H3 3D latent upscaler (copied from Comfyui_Minimax_h3_latent_Upscaler)
# ---------------------------------------------------------------------------

_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180265264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, t, hh, w = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2).transpose(1, 2)
        v = self.v(h).flatten(2).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(b, c, t, hh, w)
        return x + self.proj_out(h)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class _LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(_AttnBlock3D(channels))
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(_AttnBlock3D(channels))
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


_MODEL_CACHE = {}


def _get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]


def _scan_models():
    files = []
    model_dir = _get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
    if not names:
        return [f"(no upscale models found in: {model_dir})"]
    return names


def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False
    return cfg


def load_upscale_model(name, device, precision):
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key].to(device)

    path = os.path.join(_get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)
    if cfg["in_channels"] != 24:
        raise ValueError(
            f"Checkpoint '{name}' is not an H3 latent upscaler "
            f"(expected 24 input channels, got {cfg['in_channels']})."
        )

    model = _LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    _MODEL_CACHE[cache_key] = model
    print(f"[MMH3-UltimateUpscale] Loaded upscale model: {name}")
    return model


def unload_upscale_model(name, device, precision):
    """Free VRAM after upscaling: move the cached upscale model back to CPU. It stays
    in _MODEL_CACHE so the next chunk only re-copies weights to GPU, not re-reads disk."""
    cache_key = f"{name}::{device}::{precision}"
    model = _MODEL_CACHE.get(cache_key)
    if model is not None and str(next(model.parameters()).device) != "cpu":
        model.to("cpu")
        print(f"[MMH3-UltimateUpscale] Offloaded upscale model: {name}")
    if str(device) == "cuda":
        torch.cuda.empty_cache()


def _compute_upscale_target(width, height, h_in, w_in):
    """Pixel target W/H + effective scale from EXPLICIT target dimensions.

    The upscale target is always an exact pixel size (it must match the
    conditioning's generation size)."""
    ds = VAE_DOWNSAMPLE
    w_px = float(width)
    h_px = float(height)
    eff = (w_px / (w_in * ds) + h_px / (h_in * ds)) / 2.0

    w_px_f = round(w_px / ds) * ds
    h_px_f = round(h_px / ds) * ds
    w_out = max(1, int(w_px_f // ds))
    h_out = max(1, int(h_px_f // ds))
    return h_out, w_out, eff


def upscale_video(video, param):
    """Upscale one chunk's video latent with the H3 3D upscaler. Audio untouched.

    Returns (upscaled_video, new_h, new_w). The target is computed in pixel
    space (explicit width/height, snapped to the VAE 16x grid), then the
    H3 network resizes to it. scale 1.0 (or an equivalent target) is a no-op."""
    model_name = param["model_name"]
    width = int(param["width"])
    height = int(param["height"])
    device = param["device"]
    precision = param["precision"]

    orig_dtype = video.dtype
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    _, c, t, h_in, w_in = video.shape
    h_out, w_out, eff = _compute_upscale_target(width, height, h_in, w_in)

    if eff < 1.0 and (w_out < w_in or h_out < h_in):
        raise ValueError("This model only supports upscaling (effective scale >= 1.0).")
    if w_out == w_in and h_out == h_in:
        return video, h_in, w_in

    if str(model_name).startswith('('):
        raise ValueError("Please place H3 upscale model files into the latent_upscale_models directory")

    s = video.to(device=dev, dtype=compute_dtype, copy=True)
    model = load_upscale_model(model_name, dev, precision)
    norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

    with torch.inference_mode():
        s = s.sub(norm_mean).div(norm_std)
        out = model(s, scale=eff, target_size=(t, h_out, w_out))
        del s
        out = out.mul(norm_std).add(norm_mean)

    out = out.to(device="cpu", dtype=orig_dtype)
    unload_upscale_model(model_name, dev, precision)
    return out, h_out, w_out


def upscale_video_interp(video, param):
    """Model-free upscale of one chunk's video latent via interpolation (audio
    untouched) - mirrors ComfyUI's 'Upscale Latent' node. Returns (upscaled_video,
    new_h, new_w); the video latent [B,24,T,H,W] is resized in HxW only."""
    method = param["method"]
    width = int(param["width"])
    height = int(param["height"])

    _, c, t, h_in, w_in = video.shape
    h_out, w_out, _ = _compute_upscale_target(width, height, h_in, w_in)
    if h_out == h_in and w_out == w_in:
        return video, h_in, w_in

    video_bt = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h_in, w_in)
    up = torch.nn.functional.interpolate(video_bt, size=(h_out, w_out), mode=method)
    up = up.reshape(video.shape[0], t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    return up, h_out, w_out


def upscale_latent(video, param):
    """Dispatch a chunk's video upscale: H3 3D model (param has 'model_name') or
    model-free interpolation (param has 'method'). Audio is never touched."""
    if "model_name" in param:
        return upscale_video(video, param)
    return upscale_video_interp(video, param)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------

def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg):
    """Sample one piece (full chunk or tile). Mirrors SamplerCustomAdvanced,
    including the x0 preview callback. Returns nested samples (video+audio)."""
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_image
    noise_mask = latent.get("noise_mask")

    guider = build_guider(model, cond, negative, cfg)
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(latent), latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples


# ---------------------------------------------------------------------------
# stitching helpers
# ---------------------------------------------------------------------------

def temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0):
    """Stitch one re-sampled chunk into the accumulated latent (cross-fade).
    Mirrors 'Append MiniMax H3 Latents'. Returns (result_v, result_a)."""
    if acc_v is None:
        return chunk_v, chunk_a

    gi = k0
    agi = round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    result_v = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                           device=acc_v.device, dtype=acc_v.dtype)
    result_a = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    result_v[:, :, :acc_v.shape[2]] = acc_v
    result_a[:, :, :, :acc_a.shape[-1]] = acc_a

    v = chunk_v
    a = chunk_a
    ov = (acc_v.shape[2] - gi) if index > 0 else 0
    if ov > 0:
        ov = min(ov, v.shape[2])
        tail = result_v[:, :, gi:gi + ov].clone()
        result_v[:, :, gi:gi + ov] = _crossfade(tail, v[:, :, :ov], dim=2)
        v = v[:, :, ov:]
    write_v = gi + max(ov, 0)
    if v.shape[2] > 0:
        result_v[:, :, write_v:write_v + v.shape[2]] = v

    ova = (acc_a.shape[-1] - agi) if index > 0 else 0
    if ova > 0:
        ova = min(ova, a.shape[-1])
        tail = result_a[:, :, :, agi:agi + ova].clone()
        result_a[:, :, :, agi:agi + ova] = _crossfade(tail, a[:, :, :, :ova], dim=3)
        a = a[:, :, :, ova:]
    write_a = agi + max(ova, 0)
    if a.shape[-1] > 0:
        result_a[:, :, :, write_a:write_a + a.shape[-1]] = a

    return result_v, result_a


def spatial_process(chunk_v, chunk_a, cond, sp, model, noise, sampler, sigmas, negative, cfg):
    """Inner loop: spatial split -> per-tile sampling -> spatial stitch.
    Mirrors the spatial split/extract/append trio. Audio is carried unchanged
    (frozen in every tile, never re-sampled). Returns (reassembled_video, info)."""
    tw = int(sp["tile_width"]) // 16
    th = int(sp["tile_height"]) // 16
    ol_w = int(sp["spatial_w_overlap"]) // 16
    ol_h = int(sp["spatial_h_overlap"]) // 16
    fw = int(sp["fade_width"]) // 16
    fh = int(sp["fade_height"]) // 16
    min_tile = int(sp["min_tile_size"]) // 16
    overlap_mode = sp["overlap_mode"]
    overlap_blend = sp["overlap_blend"]

    if tw <= 0 or th <= 0:
        raise ValueError("tile_width/tile_height must be multiples of 32 pixels")
    if ol_w >= tw or ol_h >= th:
        raise ValueError("spatial_w_overlap/spatial_h_overlap must be smaller than the tile size")
    if min_tile > th or min_tile > tw:
        raise ValueError("min_tile_size must not exceed the tile size")

    _, c, t, h, w = chunk_v.shape
    rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_tile, min_tile)
    nrows, ncols = len(rows), len(cols)
    ta = chunk_a.shape[-1]

    acc_v = chunk_v.clone()
    tile_info = {
        "rows": rows, "cols": cols, "tile_h": th, "tile_w": tw,
        "overlap_h": ol_h, "overlap_w": ol_w,
        "row_overlaps": row_ovl, "col_overlaps": col_ovl, "min_tile": min_tile,
        "tile_rows": trows, "tile_cols": tcols, "n_cols": ncols,
        "orig_h": h, "orig_w": w, "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
    }

    for i in range(nrows):
        for j in range(ncols):
            r0, c0 = rows[i], cols[j]
            tr, tc = trows[i], tcols[j]
            ovh = row_ovl[i]
            ovw = col_ovl[j]

            tile = torch.zeros((1, c, t, tr, tc), device=chunk_v.device, dtype=chunk_v.dtype)
            tile[:, :, :, :, :] = chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc]
            # pre-fill done-overlap strips from the accumulated re-sampled result
            if j > 0 and ovw > 0:
                tile[:, :, :, :, :ovw] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + ovw]
            if i > 0 and ovh > 0:
                tile[:, :, :, :ovh, :] = acc_v[:, :, :, r0:r0 + ovh, c0:c0 + tc]

            m = spatial_fade_mask(tr, tc, ovh, ovw,
                                  done_top=(i > 0), done_left=(j > 0),
                                  fade_h=fh, fade_w=fw)
            mv = m[None, None, None]
            ma = torch.zeros((1, 32, 2, ta), device=chunk_a.device, dtype=chunk_a.dtype)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
            }

            cond_tile = crop_keyframes_to_tile(cond, h, w, r0, c0, tr, tc)
            out = sample_piece(piece, cond_tile, model, noise, sampler, sigmas, negative, cfg)
            tile_v = out.tensors[0]

            region = acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            if j > 0 and ovw > 0:
                tt = torch.linspace(0.0, 1.0, ovw, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts[None, None, None, None, :])
                                            + tile_v[:, :, :, :, :ovw] * wts[None, None, None, None, :])
            if i > 0 and ovh > 0:
                tt = torch.linspace(0.0, 1.0, ovh, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts[None, None, None, :, None])
                                            + tile_v[:, :, :, :ovh, :] * wts[None, None, None, :, None])
            band = torch.zeros((1, 1, 1, tr, tc), device=region.device, dtype=torch.bool)
            if j > 0 and ovw > 0:
                band[:, :, :, :, :ovw] = True
            if i > 0 and ovh > 0:
                band[:, :, :, :ovh, :] = True
            region = torch.where(band, region, tile_v)
            acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

    return acc_v, tile_info


# ---------------------------------------------------------------------------
# parameter nodes
# ---------------------------------------------------------------------------

class MMH3LatentUpscaleWithModelParams(io.ComfyNode):
    """Bundle the H3 3D model-based latent upscale settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentUpscaleWithModelParams",
            display_name="MMH3 Latent Upscale with Model Params",
            category="model/latent/minimax",
            description=(
                "Bundle the H3 3D latent upscale settings for the 'MMH3 Ultimate "
                "Upscale' node. Uses the minimax_h3_latent_upscaler_3d checkpoints "
                "from the latent_upscale_models folder (not the standard LatentUpscale "
                "loader - the H3 weights do not match its supported architectures)."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 upscale"],
            inputs=[
                io.Combo.Input("model_name", options=_scan_models(),
                               tooltip="The H3 latent upscale model file in the latent_upscale_models folder (e.g. minimax_h3_latent_upscaler_3d_*.safetensors). Loading a non-H3 upscale model may error."),
                io.Int.Input("width", default=1280, min=64, max=4096, step=32,
                             tooltip="Target overall pixel width of the upscaled frame (snapped to a multiple of 32, the H3 upscaler's required grid). Must match the conditioning's generation size."),
                io.Int.Input("height", default=704, min=64, max=4096, step=32,
                             tooltip="Target overall pixel height of the upscaled frame (snapped to a multiple of 32, the H3 upscaler's required grid). Must match the conditioning's generation size."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp16", "fp32", "bf16"], default="fp16"),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Upscale settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, model_name, width, height, device, precision) -> io.NodeOutput:
        width = int(round(width / 32.0)) * 32
        height = int(round(height / 32.0)) * 32
        param = {
            "model_name": model_name,
            "width": width,
            "height": height,
            "device": device,
            "precision": precision,
        }
        return io.NodeOutput(param)


class MMH3LatentUpscaleParams(io.ComfyNode):
    """Bundle model-free latent upscale settings (interpolation) consumed by the
    Ultimate Upscale node. The video latent is resized spatially, audio passes
    through. Mirrors ComfyUI's 'Upscale Latent' node but keeps the H3 nested
    (video+audio) structure intact."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentUpscaleParams",
            display_name="MMH3 Latent Upscale Params",
            category="model/latent/minimax",
            description=(
                "Bundle model-free latent upscale settings for the 'MMH3 Ultimate "
                "Upscale' node. The chunk's video latent is resized spatially by "
                "interpolation (audio untouched) - no H3 upscale model is loaded. "
                "Target size must match the conditioning's generation size. Reference: "
                "ComfyUI 'Upscale Latent'."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 latent upscale", "model-free upscale"],
            inputs=[
                io.Combo.Input("method", options=["nearest-exact", "bilinear", "area", "bicubic"],
                               default="bilinear",
                               tooltip="Interpolation used to resize the video latent's spatial HxW (same as Upscale Latent)."),
                io.Int.Input("width", default=1280, min=64, max=4096, step=32,
                                tooltip="Target overall pixel width of the upscaled frame (snapped to a multiple of 32). Must match the conditioning's generation size."),
                io.Int.Input("height", default=704, min=64, max=4096, step=32,
                                tooltip="Target overall pixel height of the upscaled frame (snapped to a multiple of 32). Must match the conditioning's generation size."),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Model-free upscale settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, method, width, height) -> io.NodeOutput:
        width = int(round(width / 32.0)) * 32
        height = int(round(height / 32.0)) * 32
        param = {
            "method": method,
            "width": width,
            "height": height,
        }
        return io.NodeOutput(param)


class MMH3TemporalSplitParams(io.ComfyNode):
    """Bundle the temporal split settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalSplitParams",
            display_name="MMH3 Temporal Split Params",
            category="model/latent/minimax",
            description=(
                "Bundle the temporal split settings for the 'MMH3 Ultimate Upscale' "
                "node: how the input latent is cut into overlapping time chunks "
                "(outer loop) and how seams are anchored."
            ),
            search_aliases=["h3 temporal params", "temporal split param", "time split"],
            inputs=[
                io.Int.Input("chunk_length", default=136, min=17, max=100000, step=17,
                             tooltip="Target pixel frames per chunk (at 24 fps). MUST be a multiple of 17 (one keyframe grid step). 136 = ~5.7s, 153 = ~6.4s."),
                io.Int.Input("temporal_overlap", default=17, min=0, max=100000, step=17,
                             tooltip="Pixel frames of overlap between consecutive chunks. MUST be a multiple of 17; recommended 17. Must be smaller than chunk_length."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="How much of the previous chunk's re-sampled boundary the frozen frame-0 anchor keeps: 1.0 = exact content, 0.999 = model default, 0.0 = no anchoring."),
            ],
            outputs=[
                H3_TEMPORAL_PARAM.Output("temporal_split_param",
                                         tooltip="Temporal split settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, chunk_length, temporal_overlap, anchor_strength) -> io.NodeOutput:
        if chunk_length % 17 != 0:
            raise ValueError(f"chunk_length must be a multiple of 17 (the model's keyframe grid step); got {chunk_length}")
        if temporal_overlap % 17 != 0:
            raise ValueError(f"temporal_overlap must be a multiple of 17 (the model's keyframe grid step); got {temporal_overlap}")
        if temporal_overlap >= chunk_length:
            raise ValueError("temporal_overlap must be smaller than chunk_length")
        param = {
            "chunk_length": chunk_length,
            "temporal_overlap": temporal_overlap,
            "anchor_strength": anchor_strength,
        }
        return io.NodeOutput(param)


class MMH3SpatialSplitParams(io.ComfyNode):
    """Bundle the spatial split settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialSplitParams",
            display_name="MMH3 Spatial Split Params",
            category="model/latent/minimax",
            description=(
                "Bundle the spatial tile settings for the 'MMH3 Ultimate Upscale' "
                "node: tile size, per-axis overlap and fade, and seam stitching "
                "rules (inner loop)."
            ),
            search_aliases=["h3 spatial params", "spatial split param", "tile param"],
            inputs=[
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="Tile width in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="Tile height in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("spatial_w_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Horizontal overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile width."),
                io.Int.Input("spatial_h_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Vertical overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile height."),
                io.Int.Input("fade_width", default=32, min=0, max=100000, step=32,
                             tooltip="Width in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0, keeps the neighbour's content) + this FADE segment (interior side). fade_width sets the fade length; the frozen segment takes the rest (overlap - fade). Default 32. Set to 0 to freeze the entire overlap strip."),
                io.Int.Input("fade_height", default=32, min=0, max=100000, step=32,
                             tooltip="Height in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0, keeps the neighbour's content) + this FADE segment (interior side). fade_height sets the fade length; the frozen segment takes the rest (overlap - fade). Default 32. Set to 0 to freeze the entire overlap strip."),
                io.Int.Input("min_tile_size", default=256, min=0, max=100000, step=32,
                             tooltip="Minimum PIXEL size of edge tiles. If a leftover edge tile would be smaller, the last tile is pulled back until it reaches at least this size; the seam overlap then grows and is blended over its full width. 256 (default) keeps small leftover tiles as-is. Must not exceed the tile size."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier",
                               tooltip="Who wins each shared overlap band when stitching. 'earlier' (default): the already-stitched content wins. 'later': the re-sampled tile wins. Does NOT affect the noise mask."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="How the overlap band transitions when stitching: linear cross-fade (default), smoothstep (eased), overwrite (whole band from the overlap_mode side), midpoint (hard switch at the band's middle)."),
            ],
            outputs=[
                H3_SPATIAL_PARAM.Output("spatial_split_param",
                                        tooltip="Spatial split settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, tile_width, tile_height, spatial_w_overlap, spatial_h_overlap,
                fade_width, fade_height, min_tile_size, overlap_mode, overlap_blend) -> io.NodeOutput:
        for name, v in (("tile_width", tile_width), ("tile_height", tile_height),
                        ("spatial_w_overlap", spatial_w_overlap), ("spatial_h_overlap", spatial_h_overlap),
                        ("fade_width", fade_width), ("fade_height", fade_height),
                        ("min_tile_size", min_tile_size)):
            if v % 32 != 0:
                raise ValueError(f"'{name}' must be a multiple of 32 pixels (the model's 2x2 latent patch grid); got {v}.")
        if spatial_w_overlap >= tile_width:
            raise ValueError("spatial_w_overlap must be smaller than tile_width")
        if spatial_h_overlap >= tile_height:
            raise ValueError("spatial_h_overlap must be smaller than tile_height")
        if fade_width > spatial_w_overlap:
            raise ValueError("fade_width must not exceed spatial_w_overlap")
        if fade_height > spatial_h_overlap:
            raise ValueError("fade_height must not exceed spatial_h_overlap")
        if min_tile_size > tile_width or min_tile_size > tile_height:
            raise ValueError("min_tile_size must not exceed the tile size")
        param = {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "spatial_w_overlap": spatial_w_overlap,
            "spatial_h_overlap": spatial_h_overlap,
            "fade_width": fade_width,
            "fade_height": fade_height,
            "min_tile_size": min_tile_size,
            "overlap_mode": overlap_mode,
            "overlap_blend": overlap_blend,
        }
        return io.NodeOutput(param)


# ---------------------------------------------------------------------------
# main node
# ---------------------------------------------------------------------------

class MMH3UltimateUpscale(io.ComfyNode):
    """One node for the full latent re-enhancement pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3UltimateUpscale",
            display_name="MMH3 Ultimate Upscale",
            category="model/latent/minimax",
            description=(
                "Re-sample an already-denoised MiniMax H3 AV latent through the full "
                "auto pipeline in one node: temporal split (outer loop) -> latent "
                "upscale (per chunk) -> spatial split (inner loop) -> per-tile "
                "sampling with preview -> spatial stitch -> temporal stitch. Each "
                "chunk/tile is sampled with a fresh guider built from the per-piece "
                "conditioning (re-anchored and cropped keyframes), keeping peak VRAM "
                "to one tile. 'latent_upscale_param', 'temporal_split_param' and "
                "'spatial_split_param' are optional - leave any unconnected to skip "
                "that stage (no upscale / single chunk / full-chunk sampling)."
            ),
            search_aliases=["h3 ultimate upscale", "ultimate upscale", "h3 auto upscale", "h3 enhance"],
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model used to re-sample every chunk/tile (guider is built internally)."),
                io.Conditioning.Input("conditioning",
                                      tooltip="Conditioning used to generate this latent. Per chunk it is re-anchored in time; per tile its keyframes are spatially cropped; the frame-0 keyframe is pinned to the previous chunk's re-sampled frame."),
                io.Latent.Input("latent", tooltip="Denoised MiniMax H3 AV latent to enhance."),
                io.Noise.Input("noise", tooltip="Noise source; one noise tensor is generated per piece."),
                io.Sampler.Input("sampler", tooltip="Sampler used for every chunk/tile."),
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule used for every chunk/tile."),
                io.Conditioning.Input("negative", optional=True,
                                      tooltip="Negative conditioning. When connected, a CFGGuider is used with the 'cfg' value; otherwise a basic guider (positive only)."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="CFG scale used when 'negative' is connected."),
                H3_UPSCALE_PARAM.Input("latent_upscale_param", optional=True,
                                       tooltip="Output of 'MMH3 Latent Upscale with Model Params' (H3 3D upscaler) OR 'MMH3 Latent Upscale Params' (model-free interpolation). Leave unconnected to skip upscaling."),
                H3_TEMPORAL_PARAM.Input("temporal_split_param", optional=True,
                                        tooltip="Output of 'MMH3 Temporal Split Params'. Leave unconnected to process the latent as a single chunk."),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True,
                                       tooltip="Output of 'MMH3 Spatial Split Params'. Leave unconnected to sample each chunk whole (no tiling)."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Upscaled, re-sampled, stitched MiniMax H3 AV latent."),
                io.Dict.Output("segments_info",
                               tooltip="DEBUG ONLY. Per-chunk metadata: frame start/count, video/audio token ranges, upscale applied."),
                io.Dict.Output("tiles_info",
                               tooltip="DEBUG ONLY. Per-chunk spatial grid metadata: offsets, tile extents, overlaps, stitching mode."),
            ],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0,
                temporal_split_param=None, spatial_split_param=None,
                latent_upscale_param=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MMH3UltimateUpscale expects a MiniMax H3 AV latent (nested video [B,24,T,H,W] + audio [B,32,2,T])")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("MMH3UltimateUpscale expects a single-video latent (batch 1)")

        # keep the H3 packed layout and cond row counts in lockstep (refs metadata
        # rewritten from the real latents; phantom latent-less visual refs dropped)
        conditioning = normalize_minimax_refs(conditioning)

        # fail early if the upscale target is smaller than the spatial tile size;
        # tiles can never cover a chunk smaller than one tile, which would only
        # surface as a confusing error during the sampling/stitching phase.
        if latent_upscale_param is not None and spatial_split_param is not None:
            up_w = int(latent_upscale_param["width"])
            up_h = int(latent_upscale_param["height"])
            tile_w = int(spatial_split_param["tile_width"])
            tile_h = int(spatial_split_param["tile_height"])
            if up_w < tile_w:
                raise ValueError(
                    f"Upscale width ({up_w}) must be >= tile_width ({tile_w})"
                )
            if up_h < tile_h:
                raise ValueError(
                    f"Upscale height ({up_h}) must be >= tile_height ({tile_h})"
                )

        tv = video.shape[2]
        ta = audio.shape[-1]

        if temporal_split_param is not None:
            chunk_length = int(temporal_split_param["chunk_length"])
            overlap = int(temporal_split_param["temporal_overlap"])
            bounds, frame_count = compute_segments(tv, chunk_length, overlap)
            anchor_strength = temporal_split_param["anchor_strength"]
        else:
            frame_count = frames_for_tokens(tv)
            bounds = [(0, 0, tv, frame_count)]
            anchor_strength = 0.999

        acc_v = None
        acc_a = None
        segments_debug = []
        tiles_debug = []

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, ta)
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            # 1. upscale this chunk's video (audio untouched). While the 3D upscaler
            #    is on the GPU the diffusion model isn't needed, so offload it first
            #    to avoid H3 + upscaler resident simultaneously; the next sample
            #    reloads H3 automatically.
            upscaled = False
            if latent_upscale_param is not None:
                use_model = "model_name" in latent_upscale_param
                if use_model and str(latent_upscale_param["device"]) == "cuda" and hasattr(model, "clone_base_uuid"):
                    # the 3D upscaler is on the GPU during upscale; offload the
                    # diffusion model so they don't reside simultaneously
                    comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                    comfy.model_management.soft_empty_cache()
                chunk_v, _, _ = upscale_latent(chunk_v, latent_upscale_param)
                upscaled = True

            # 2. time re-anchor; keyframe video latents are always resized to the
            #    (possibly upscaled) chunk size - the H3 packed layout requires
            #    keyframes on the sampled target's spatial grid, and in the intended
            #    workflow the conditioning is generated at the upscaled size already
            cond_i = reanchor_conditioning(conditioning, f0, f1, (chunk_v.shape[3], chunk_v.shape[4]))

            # 3. pin frame-0 keyframe to the previous chunk's re-sampled frame
            if i > 0 and acc_v is not None:
                cond_i = anchor_conditioning(cond_i, acc_v, f0, anchor_strength)

            # 4. inner loop: spatial split -> sample -> stitch
            if spatial_split_param is not None:
                chunk_out_v, tile_info = spatial_process(
                    chunk_v, chunk_a, cond_i, spatial_split_param,
                    model, noise, sampler, sigmas, negative, cfg,
                )
                tile_info = dict(tile_info)
                tile_info["chunk"] = i
                tiles_debug.append(tile_info)
            else:
                piece = {"samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))}
                out = sample_piece(piece, cond_i, model, noise, sampler, sigmas, negative, cfg)
                chunk_out_v = out.tensors[0]

            # 5. temporal stitch
            acc_v, acc_a = temporal_append(acc_v, acc_a, chunk_out_v, chunk_a, i, k0, f0)

            segments_debug.append({
                "chunk": i,
                "frame_start": f0,
                "frame_count": f1 - f0,
                "video_tokens": [k0, k1],
                "audio_tokens": list(audio_range(f0, f1)),
                "upscaled": upscaled,
                "spatial_h": chunk_v.shape[3],
                "spatial_w": chunk_v.shape[4],
            })

        # all chunks sampled & stitched: the diffusion model is no longer needed,
        # unload it so the caller (e.g. VAE decode of the large latent) gets the VRAM
        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        out = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        return io.NodeOutput(out, segments_debug, tiles_debug)


# ===========================================================================
# LTX2.5 Ultimate Upscale
# Mirrors MMH3 Ultimate Upscale, adapted for LTX2.5 (spatial 32x, temporal 8x,
# video 128ch, 8k+1 frame grid). Audio is carried unchanged (frozen, mask=0)
# so its cross-fade reassembly is lossless.
# ===========================================================================

LTX_VAE_DOWNSAMPLE = 32
LTX_TEMPORAL_FACTOR = 8
LTX_VIDEO_CHANNELS = 128

LTX_UPSCALE_PARAM = io.Custom("LTX_UPSCALE_PARAM")
LTX_TEMPORAL_PARAM = io.Custom("LTX_TEMPORAL_PARAM")
LTX_SPATIAL_PARAM = io.Custom("LTX_SPATIAL_PARAM")
LTX_MSR_PARAM = io.Custom("LTX_MSR_REFERENCE_PARAMETERS")
LTX25_REF_GUIDES = io.Custom("LTX25_REFERENCE_GUIDES")


def is_ltx_av_latent(samples):
    """True if samples is a nested (video, audio) LTX2.5 latent."""
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == LTX_VIDEO_CHANNELS)


def ltx_frames_for_tokens(n):
    """Pixel frames covered by the first `n` LTX video latent tokens (8k+1 grid)."""
    if n <= 0:
        return 0
    return (n - 1) * LTX_TEMPORAL_FACTOR + 1


def ltx_tokens_for_frames(f):
    """Smallest LTX token count whose cumulative frames reach at least `f`."""
    if f <= 1:
        return 1
    return (f - 1) // LTX_TEMPORAL_FACTOR + 1


def ltx_compute_segments(tv, chunk_length, overlap):
    """Per-chunk (video_token_start, frame_start, video_token_end, frame_end)
    on the LTX 8k+1 grid. chunk_length/overlap are pixel frames."""
    frame_count = ltx_frames_for_tokens(tv)
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_length <= overlap:
        raise ValueError("overlap must be smaller than chunk_length")
    hop = chunk_length - overlap
    bounds = []
    i = 0
    while True:
        s = i * hop
        e = min(s + chunk_length, frame_count)
        k0 = ltx_tokens_for_frames(s) if i > 0 else 0
        f0 = ltx_frames_for_tokens(k0)
        if e >= frame_count:
            k1, f1 = tv, frame_count
        else:
            k1 = ltx_tokens_for_frames(e)
            f1 = ltx_frames_for_tokens(k1)
            if k1 <= k0:
                k1 = k0 + 1
                f1 = ltx_frames_for_tokens(k1)
            if k1 >= tv:
                k1, f1 = tv, frame_count
        bounds.append((k0, f0, k1, f1))
        if k1 >= tv:
            break
        i += 1
    return bounds, frame_count


def ltx_upscale_latent(video, upscale_model, vae):
    """2x upscale of one chunk's video latent via the LTX latent upscaler
    (per_channel_statistics normalize/un_normalize). Mirrors LTXVLatentUpscaler.
    Audio is untouched (handled by the caller)."""
    orig_dtype = video.dtype
    device = upscale_model.load_device
    model = upscale_model.model
    model_dtype = upscale_model.model_dtype()
    comfy.model_management.load_models_gpu(
        [upscale_model], memory_required=math.prod(video.shape) * 3000.0)
    latents = video.to(dtype=model_dtype, device=device)
    stats = vae.first_stage_model.per_channel_statistics
    latents = stats.un_normalize(latents)
    upsampled = model(latents)
    upsampled = stats.normalize(upsampled)
    upsampled = upsampled.to(
        dtype=orig_dtype, device=comfy.model_management.intermediate_device())
    return upsampled


def ltx_resize_latent(video, width, height):
    """Resize one chunk's LTX video latent HxW to the target (height, width) in
    PIXELS via interpolation. Applied after the 2x latent upscaler so the final
    chunk matches the requested overall upscaled size - the LTX upscaler is fixed
    2x, so this is a no-op when the target equals exactly 2x the input."""
    width = int(round(width / LTX_VAE_DOWNSAMPLE)) * LTX_VAE_DOWNSAMPLE
    height = int(round(height / LTX_VAE_DOWNSAMPLE)) * LTX_VAE_DOWNSAMPLE
    _, c, t, h_in, w_in = video.shape
    h_out, w_out = height // LTX_VAE_DOWNSAMPLE, width // LTX_VAE_DOWNSAMPLE
    if h_out == h_in and w_out == w_in:
        return video
    video_bt = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h_in, w_in)
    up = torch.nn.functional.interpolate(video_bt, size=(h_out, w_out), mode="bilinear", align_corners=False)
    up = up.reshape(video.shape[0], t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    return up


# ---------------------------------------------------------------------------
# LTX2.5 reference guides (native LTXVAddGuide mechanism; MSR-compatible)
# ---------------------------------------------------------------------------

def ltx_encode_reference(vae, latent_h, latent_w, image, ref_frames):
    """Encode one reference still into an LTX video guide latent at (latent_h, latent_w).

    Mirrors LTXVAddGuide.encode for a repeated still: resize (center-crop) to the
    chunk's pixel grid, repeat to ref_frames pixel frames (snapped to 8k+1), encode.
    After encoding, the guide is spatially resized to EXACTLY (latent_h, latent_w) to
    avoid any VAE rounding mismatch. Returns (guide_latent [B,128,F,H,W], scale_factors)."""
    time_scale, width_scale, height_scale = vae.downscale_index_formula
    repeated = image.repeat(ref_frames, 1, 1, 1)
    keep = ((repeated.shape[0] - 1) // time_scale) * time_scale + 1
    repeated = repeated[:keep]
    target_w = int(latent_w * width_scale)
    target_h = int(latent_h * height_scale)
    pixels = comfy.utils.common_upscale(
        repeated.movedim(-1, 1), target_w, target_h, "bilinear", crop="center").movedim(1, -1)
    pixels = pixels[..., :3]
    guide = vae.encode(pixels)
    if guide.shape[3] != latent_h or guide.shape[4] != latent_w:
        B, C, T, H, W = guide.shape
        guide = F.interpolate(
            guide.reshape(B * C * T, 1, H, W),
            size=(latent_h, latent_w), mode="bilinear", align_corners=False
        ).reshape(B, C, T, latent_h, latent_w)
    return guide, vae.downscale_index_formula


def ltx_msr_slot_embedding(slot_state, slot_id, device, dtype):
    """Fourier-MLP reference-slot embedding from an MSR LoRA checkpoint
    (same convention as ComfyUI-LTX2.5-MSR: slot_id / 16 -> sin/cos features -> MLP)."""
    frequencies = slot_state["frequencies"].to(device=device, dtype=torch.float32)
    scaled = torch.tensor([float(slot_id) / 16.0], device=device, dtype=torch.float32)
    phases = scaled * frequencies
    features = torch.cat((scaled, torch.sin(phases), torch.cos(phases)))
    w0 = slot_state["net.0.weight"].to(device=device, dtype=torch.float32)
    b0 = slot_state["net.0.bias"].to(device=device, dtype=torch.float32)
    hidden = F.silu(F.linear(features, w0, b0))
    w2 = slot_state["net.2.weight"].to(device=device, dtype=torch.float32)
    b2 = slot_state["net.2.bias"].to(device=device, dtype=torch.float32)
    return F.linear(hidden, w2, b2).to(dtype=dtype)


def ltx_append_guides(chunk_v, video_mask, positive, negative, ref_guides):
    """Append reference guide frames to one chunk via the native LTXVAddGuide mechanism.

    `ref_guides` is the LTX25ReferenceParams output bundle (guides / offsets /
    scale_factors / strength). Each guide latent is first spatially resized to the
    chunk's exact grid, then appended: guides sit at the END of the latent while
    their recorded keyframe positions are restored by RoPE inside the model; the
    noise_mask marks guide frames with 1 - strength so they act as near-clean
    conditioning tokens. Returns (work_v, video_mask, positive, negative, appended_frames).
    `positive`/`negative` inputs stay untouched (conditioning_set_values copies)."""
    if _ltx_nodes is None:
        raise RuntimeError("This ComfyUI build does not expose comfy_extras.nodes_lt (LTX guide support).")
    if negative is None:
        negative = positive  # cfg-less run: this branch's conds are discarded anyway
    strength = float(ref_guides["strength"])
    scale_factors = ref_guides["scale_factors"]
    _, _, Tv, H, W = chunk_v.shape
    appended = 0
    for gl, offset in zip(ref_guides["guides"], ref_guides["offsets"]):
        gl = gl.to(dtype=chunk_v.dtype, device=chunk_v.device)
        if gl.shape[3] != H or gl.shape[4] != W:
            B, C, T, Gh, Gw = gl.shape
            gl = F.interpolate(gl.reshape(B * C * T, 1, Gh, Gw), size=(H, W),
                               mode="bilinear", align_corners=False).reshape(B, C, T, H, W)
        positive, negative, chunk_v, video_mask = _ltx_nodes.LTXVAddGuide.append_keyframe(
            positive, negative, offset, chunk_v, video_mask, gl,
            strength, scale_factors, causal_fix=True)
        appended += gl.shape[2]
    return chunk_v, video_mask, positive, negative, appended


def ltx_temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0):
    """Stitch one re-sampled LTX chunk (cross-fade over overlap). Audio is
    frozen (never re-sampled) so its cross-fade mixes identical content and is
    a no-op - the audio is reassembled losslessly. Audio layout is (B, C, time,
    freq): TIME is axis 2."""
    if acc_v is None:
        return chunk_v, chunk_a
    gi = k0
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    # LTX audio layout is (B, C, time, freq): TIME is axis 2.
    total_a = max(acc_a.shape[2], gi + chunk_a.shape[2])
    result_v = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                           device=acc_v.device, dtype=acc_v.dtype)
    a_shape = list(acc_a.shape)
    a_shape[2] = total_a
    result_a = torch.zeros(a_shape, device=acc_a.device, dtype=acc_a.dtype)
    result_v[:, :, :acc_v.shape[2]] = acc_v
    result_a[:, :, :acc_a.shape[2]] = acc_a

    v, a = chunk_v, chunk_a
    ov = (acc_v.shape[2] - gi) if index > 0 else 0
    if ov > 0:
        ov = min(ov, v.shape[2])
        result_v[:, :, gi:gi + ov] = _crossfade(
            result_v[:, :, gi:gi + ov].clone(), v[:, :, :ov], dim=2)
        v = v[:, :, ov:]
    wv = gi + max(ov, 0)
    if v.shape[2] > 0:
        result_v[:, :, wv:wv + v.shape[2]] = v

    ova = (acc_a.shape[2] - gi) if index > 0 else 0
    if ova > 0:
        ova = min(ova, a.shape[2])
        result_a[:, :, gi:gi + ova] = _crossfade(
            result_a[:, :, gi:gi + ova].clone(), a[:, :, :ova], dim=2)
        a = a[:, :, ova:]
    wa = gi + max(ova, 0)
    if a.shape[2] > 0:
        result_a[:, :, wa:wa + a.shape[2]] = a
    return result_v, result_a


def ltx_spatial_process(chunk_v, chunk_a, cond, sp, model, noise, sampler, sigmas,
                        negative, cfg, vmask=None, bypass_audio=True):
    """Inner loop: spatial split -> per-tile sampling -> spatial stitch.
    Mirrors MMH3 spatial_process, adapted for LTX (32x VAE). Audio is carried
    unchanged (frozen in every tile, never re-sampled). T2V conditioning has no
    spatial keyframes, so it is passed through uncropped. Returns
    (reassembled_video, tile_info)."""
    tw = int(sp["tile_width"]) // LTX_VAE_DOWNSAMPLE
    th = int(sp["tile_height"]) // LTX_VAE_DOWNSAMPLE
    ol_w = int(sp["spatial_w_overlap"]) // LTX_VAE_DOWNSAMPLE
    ol_h = int(sp["spatial_h_overlap"]) // LTX_VAE_DOWNSAMPLE
    fw = int(sp["fade_width"]) // LTX_VAE_DOWNSAMPLE
    fh = int(sp["fade_height"]) // LTX_VAE_DOWNSAMPLE
    min_tile = int(sp["min_tile_size"]) // LTX_VAE_DOWNSAMPLE
    overlap_mode = sp["overlap_mode"]
    overlap_blend = sp["overlap_blend"]

    if tw <= 0 or th <= 0:
        raise ValueError("tile_width/tile_height must be multiples of 32 pixels")
    if ol_w >= tw or ol_h >= th:
        raise ValueError("spatial_w_overlap/spatial_h_overlap must be smaller than the tile size")
    if min_tile > th or min_tile > tw:
        raise ValueError("min_tile_size must not exceed the tile size")

    _, c, t, h, w = chunk_v.shape
    rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(
        h, w, th, tw, ol_h, ol_w, min_tile, min_tile)
    nrows, ncols = len(rows), len(cols)

    acc_v = chunk_v.clone()
    tile_info = {
        "rows": rows, "cols": cols, "tile_h": th, "tile_w": tw,
        "overlap_h": ol_h, "overlap_w": ol_w,
        "row_overlaps": row_ovl, "col_overlaps": col_ovl, "min_tile": min_tile,
        "tile_rows": trows, "tile_cols": tcols, "n_cols": ncols,
        "orig_h": h, "orig_w": w, "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
    }

    first_audio = None
    for i in range(nrows):
        for j in range(ncols):
            r0, c0 = rows[i], cols[j]
            tr, tc = trows[i], tcols[j]
            ovh = row_ovl[i]
            ovw = col_ovl[j]

            tile = torch.zeros((1, c, t, tr, tc), device=chunk_v.device, dtype=chunk_v.dtype)
            tile[:, :, :, :, :] = chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc]
            if j > 0 and ovw > 0:
                tile[:, :, :, :, :ovw] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + ovw]
            if i > 0 and ovh > 0:
                tile[:, :, :, :ovh, :] = acc_v[:, :, :, r0:r0 + ovh, c0:c0 + tc]

            m = spatial_fade_mask(tr, tc, ovh, ovw,
                                  done_top=(i > 0), done_left=(j > 0),
                                  fade_h=fh, fade_w=fw)
            mv = m[None, None, None]
            # Fold the temporal keyframe anchor (per-frame, spatially uniform) into
            # the spatial fade mask: a tile location is frozen if EITHER axis pins it.
            if vmask is not None:
                tile_vmask = vmask[:, :, :, r0:r0 + tr, c0:c0 + tc]
                mv = torch.min(tile_vmask, mv)
            # Audio mask: 0 = frozen (bypass), 1 = re-sampled. When re-sampled, the
            # FIRST tile's audio is kept for the whole time block (see return below).
            ma = torch.zeros_like(chunk_a) if bypass_audio else torch.ones_like(chunk_a)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
            }

            cond_tile = cond  # T2V: no spatial keyframe cropping
            out = sample_piece(piece, cond_tile, model, noise, sampler, sigmas, negative, cfg)
            tile_v = out.tensors[0]
            if i == 0 and j == 0:
                first_audio = out.tensors[1]

            region = acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            if j > 0 and ovw > 0:
                tt = torch.linspace(0.0, 1.0, ovw, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts[None, None, None, None, :])
                                            + tile_v[:, :, :, :, :ovw] * wts[None, None, None, None, :])
            if i > 0 and ovh > 0:
                tt = torch.linspace(0.0, 1.0, ovh, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts[None, None, None, :, None])
                                            + tile_v[:, :, :, :ovh, :] * wts[None, None, None, :, None])
            band = torch.zeros((1, 1, 1, tr, tc), device=region.device, dtype=torch.bool)
            if j > 0 and ovw > 0:
                band[:, :, :, :, :ovw] = True
            if i > 0 and ovh > 0:
                band[:, :, :, :ovh, :] = True
            region = torch.where(band, region, tile_v)
            acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

    # With spatial tiling, the whole time block takes a single audio track:
    #   * bypass_audio=True  -> carry the ORIGINAL input audio (chunk_a). The LTX
    #     AV model ignores audio_denoise_mask, so out.tensors[1] from any tile is
    #     freshly regenerated audio, NOT the frozen input. We must use chunk_a.
    #   * bypass_audio=False -> take the first tile's re-sampled (model-generated)
    #     audio as the block's audio.
    chunk_a_out = chunk_a if bypass_audio else (first_audio if first_audio is not None else chunk_a)
    return acc_v, chunk_a_out, tile_info


# ---------------------------------------------------------------------------
# LTX2.5 param nodes
# ---------------------------------------------------------------------------

class LTX25LatentUpscaleParams(io.ComfyNode):
    """Bundle the LTX2.5 latent upscale settings for the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25LatentUpscaleParams",
            display_name="LTX25 Latent Upscale Params",
            category="model/latent/ltxv",
            description="Bundle the LTX2.5 latent upscale settings for the 'LTX25 Ultimate Upscale' node. The LTX2.5 model is a fixed 2x latent spatial upscaler; the requested width/height are the overall upscaled frame size, reached by 2x upscaling then interpolating to the target.",
            search_aliases=["ltx25 upscale param", "ltx latent upscale"],
            inputs=[
                io.LatentUpscaleModel.Input("upscale_model",
                    tooltip="The LTX2.5 latent spatial upscaler (2x). Place ltx-2.5-latent-spatial-upscaler files in the latent_upscale_models folder."),
                io.Vae.Input("vae",
                    tooltip="The LTX2.5 VIDEO VAE (used for per_channel_statistics normalize/un_normalize during upscale). Must be the video VAE, not the audio VAE."),
                io.Int.Input("width", default=1280, min=64, max=4096, step=32,
                             tooltip="Target overall pixel width of the upscaled frame (snapped to a multiple of 32, the LTX VAE 32x grid). The 2x model upscale is followed by interpolation to this size. Must match the conditioning's generation size."),
                io.Int.Input("height", default=704, min=64, max=4096, step=32,
                             tooltip="Target overall pixel height of the upscaled frame (snapped to a multiple of 32, the LTX VAE 32x grid). The 2x model upscale is followed by interpolation to this size. Must match the conditioning's generation size."),
            ],
            outputs=[
                LTX_UPSCALE_PARAM.Output("upscale_param",
                    tooltip="LTX2.5 upscale settings consumed by 'LTX25 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, upscale_model, vae, width, height) -> io.NodeOutput:
        width = int(round(width / 32.0)) * 32
        height = int(round(height / 32.0)) * 32
        param = {"upscale_model": upscale_model, "vae": vae, "width": width, "height": height}
        return io.NodeOutput(param)


class LTX25TemporalSplitParams(io.ComfyNode):
    """Bundle the temporal split settings for the LTX25 Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25TemporalSplitParams",
            display_name="LTX25 Temporal Split Params",
            category="model/latent/ltxv",
            description="Bundle the temporal split settings for the 'LTX25 Ultimate Upscale' node: how the input latent is cut into overlapping time chunks (outer loop) and how chunk seams are anchored (anchor_mode: full band / first frame / ramp).",
            search_aliases=["ltx25 temporal param", "ltx chunk param"],
            inputs=[
                io.Int.Input("chunk_length", default=97, min=9, max=100000, step=8,
                             tooltip="Target pixel frames per chunk. MUST satisfy (n-1) % 8 == 0 (the LTX 8k+1 grid). 97 = ~4s @24fps."),
                io.Int.Input("temporal_overlap", default=9, min=1, max=100000, step=8,
                             tooltip="Pixel frames of overlap between consecutive chunks. MUST satisfy (n-1) % 8 == 0. Recommended 9 (one latent token). With 'full' anchor_mode this mostly shifts the seam position; with 'first_frame'/'ramp' it controls the visible seam transition width."),
                io.Combo.Input("anchor_mode", options=["full", "first_frame", "ramp"], default="full",
                               tooltip="How the next chunk's overlap band relates to the previous chunk's re-sampled result (pin strength set by 'anchor_strength'). "
                                       "'full' (default, original behaviour): the ENTIRE overlap is copied from the previous chunk and pinned via the noise mask (mask = 1 - anchor_strength); the stitch cross-fade mixes identical content, so temporal_overlap mostly just shifts the seam position. "
                                       "'first_frame' (Mode A, H3-style): only the FIRST latent token (~8 frames) is copied and pinned; the rest of the overlap re-samples freely and the stitch cross-fade blends the two versions across the whole band - temporal_overlap visibly controls the seam transition width. "
                                       "'ramp' (Mode B, temporal fade): the overlap is initialised from the previous chunk and its noise-mask ramps linearly from (1 - anchor_strength) at the seam to 1.0 at the band end - a true temporal fade whose width IS temporal_overlap."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001, round=0.001,
                               tooltip="Pin strength at the seam side of the overlap band, used by every anchor mode (LTX image-to-video noise_mask). 1.0 = keep previous content exactly, 0.999 = model default, 0.0 = disable anchoring (cross-fade only)."),
            ],
            outputs=[
                LTX_TEMPORAL_PARAM.Output("temporal_split_param",
                    tooltip="Temporal split settings consumed by 'LTX25 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, chunk_length, temporal_overlap, anchor_strength, anchor_mode="full") -> io.NodeOutput:
        if (chunk_length - 1) % LTX_TEMPORAL_FACTOR != 0:
            raise ValueError(f"chunk_length must satisfy (n-1) % 8 == 0 (LTX 8k+1 grid); got {chunk_length}")
        if (temporal_overlap - 1) % LTX_TEMPORAL_FACTOR != 0:
            raise ValueError(f"temporal_overlap must satisfy (n-1) % 8 == 0 (LTX 8k+1 grid); got {temporal_overlap}")
        if temporal_overlap >= chunk_length:
            raise ValueError("temporal_overlap must be smaller than chunk_length")
        param = {"chunk_length": chunk_length, "temporal_overlap": temporal_overlap,
                 "anchor_strength": anchor_strength, "anchor_mode": anchor_mode}
        return io.NodeOutput(param)


class LTX25SpatialSplitParams(io.ComfyNode):
    """Bundle the spatial tile settings for the LTX25 Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25SpatialSplitParams",
            display_name="LTX25 Spatial Split Params",
            category="model/latent/ltxv",
            description="Bundle the spatial tile settings for the 'LTX25 Ultimate Upscale' node: tile size, per-axis overlap and fade, and seam stitching rules (inner loop).",
            search_aliases=["ltx25 spatial param", "ltx tile param"],
            inputs=[
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="Tile width in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32 (LTX VAE 32x)."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="Tile height in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("spatial_w_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Horizontal overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile width."),
                io.Int.Input("spatial_h_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Vertical overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile height."),
                io.Int.Input("fade_width", default=32, min=0, max=100000, step=32,
                             tooltip="Width in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0) + this FADE segment (interior side). fade_width sets the fade length; the frozen segment takes the rest. Default 32. Set to 0 to freeze the entire overlap strip."),
                io.Int.Input("fade_height", default=32, min=0, max=100000, step=32,
                             tooltip="Height in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. See fade_width."),
                io.Int.Input("min_tile_size", default=256, min=0, max=100000, step=32,
                             tooltip="Minimum PIXEL size of edge tiles. If a leftover edge tile would be smaller, the last tile is pulled back until it reaches at least this size. Must not exceed the tile size."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier",
                               tooltip="Who wins each shared overlap band when stitching. 'earlier' (default): the already-stitched content wins. 'later': the re-sampled tile wins."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="How the overlap band transitions when stitching: linear cross-fade (default), smoothstep (eased), overwrite (whole band from the overlap_mode side), midpoint (hard switch at the band's middle)."),
            ],
            outputs=[
                LTX_SPATIAL_PARAM.Output("spatial_split_param",
                    tooltip="Spatial split settings consumed by 'LTX25 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, tile_width, tile_height, spatial_w_overlap, spatial_h_overlap,
                fade_width, fade_height, min_tile_size, overlap_mode, overlap_blend) -> io.NodeOutput:
        for name, v in (("tile_width", tile_width), ("tile_height", tile_height),
                        ("spatial_w_overlap", spatial_w_overlap), ("spatial_h_overlap", spatial_h_overlap),
                        ("fade_width", fade_width), ("fade_height", fade_height),
                        ("min_tile_size", min_tile_size)):
            if v % 32 != 0:
                raise ValueError(f"'{name}' must be a multiple of 32 pixels (LTX VAE 32x grid); got {v}.")
        if spatial_w_overlap >= tile_width:
            raise ValueError("spatial_w_overlap must be smaller than tile_width")
        if spatial_h_overlap >= tile_height:
            raise ValueError("spatial_h_overlap must be smaller than tile_height")
        if fade_width > spatial_w_overlap:
            raise ValueError("fade_width must not exceed spatial_w_overlap")
        if fade_height > spatial_h_overlap:
            raise ValueError("fade_height must not exceed spatial_h_overlap")
        if min_tile_size > tile_width or min_tile_size > tile_height:
            raise ValueError("min_tile_size must not exceed the tile size")
        param = {
            "tile_width": tile_width, "tile_height": tile_height,
            "spatial_w_overlap": spatial_w_overlap, "spatial_h_overlap": spatial_h_overlap,
            "fade_width": fade_width, "fade_height": fade_height,
            "min_tile_size": min_tile_size, "overlap_mode": overlap_mode,
            "overlap_blend": overlap_blend,
        }
        return io.NodeOutput(param)


# ---------------------------------------------------------------------------
# LTX2.5 MSR IC-LoRA Loader (copied from ComfyUI-LTX2.5-MSR so users don't
# need that package installed; output type matches its original loader)
# ---------------------------------------------------------------------------

_LTX_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)


def _ltx_metadata_bool(metadata, key, default=False):
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _ltx_extract_slot_state(lora):
    state = {}
    normal_lora = {}
    for key, value in lora.items():
        matched = False
        for prefix in _LTX_SLOT_PREFIXES:
            if key.startswith(prefix):
                state[key[len(prefix):]] = value.detach().cpu()
                matched = True
                break
        if not matched:
            normal_lora[key] = value
    return normal_lora, state


def _ltx_validate_slot_state(state, metadata):
    required = {
        "frequencies",
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    }
    missing = sorted(required.difference(state))
    enabled = _ltx_metadata_bool(metadata, "reference_slot_embedding_enabled", bool(state))
    if enabled and missing:
        raise ValueError(
            "MSR LoRA declares reference slot embeddings, but these tensors are missing: "
            + ", ".join(missing)
        )
    if not state:
        raise ValueError(
            "This LoRA does not contain reference_slot_embedding weights and is not an "
            "MSR multi-reference checkpoint."
        )


class LTX25ICLoRALoader(io.ComfyNode):
    """Load an MSR multi-reference IC-LoRA for LTX2.5.

    Applies the regular LoRA weights to the model and extracts the learned
    Fourier-MLP reference-slot embedding tensors, which 'LTX25 Reference Params'
    uses to embed each reference still into its own slot. The output type is
    compatible with the ComfyUI-LTX2.5-MSR package's IC-LoRA Loader."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25ICLoRALoader",
            display_name="LTX25 IC-LoRA Loader (MSR)",
            category="model/latent/ltxv",
            description=(
                "Load an LTX2.5 MSR multi-reference IC-LoRA: the regular weights are "
                "applied to the diffusion model and the learned reference-slot "
                "embedding is extracted for 'LTX25 Reference Params'. Connect this "
                "node instead of the ComfyUI-LTX2.5-MSR loader - no extra package "
                "required."
            ),
            search_aliases=["ltx25 ic-lora", "msr lora loader", "multi-reference lora"],
            inputs=[
                io.Model.Input("model",
                               tooltip="The LTX2.5 diffusion model the LoRA is applied to."),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                               tooltip="The MSR multi-reference LoRA checkpoint from your loras folder."),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                               tooltip="How strongly the regular LoRA weights affect the model."),
            ],
            outputs=[
                io.Model.Output("model",
                                tooltip="The model with the LoRA's regular weights applied."),
                LTX_MSR_PARAM.Output("msr_parameters",
                                     tooltip="Reference-slot parameters for 'LTX25 Reference Params' (same wire type as the ComfyUI-LTX2.5-MSR IC-LoRA Loader)."),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}
        normal_lora, slot_state = _ltx_extract_slot_state(lora)
        _ltx_validate_slot_state(slot_state, metadata)

        if strength_model != 0:
            loaded_model, _ = comfy.sd.load_lora_for_models(
                model, None, normal_lora, strength_model, 0, lora_metadata=metadata)
        else:
            loaded_model = model

        params = {
            "slot_state": slot_state,
            "metadata": dict(metadata),
            "lora_name": lora_name,
            "reference_downscale_factor": max(
                1, round(float(metadata.get("reference_downscale_factor", 1)))
            ),
            # ComfyUI compatibility mode intentionally uses its established
            # guide coordinates for every checkpoint, including LoRAs whose
            # training metadata records another temporal scale.
            "reference_temporal_scale_factor": 1,
        }
        print(f"[LTX25ICLoRALoader] Loaded {lora_name} with learned reference slot "
              f"embedding ({len(slot_state)} tensors), reference_downscale_factor="
              f"{params['reference_downscale_factor']}")
        return io.NodeOutput(loaded_model, params)


class LTX25ReferenceParams(io.ComfyNode):
    """Encode reference stills into LTX2.5 guide latents for 'LTX25 Ultimate Upscale'.

    Works with or without ComfyUI-LTX2.5-MSR: with the MSR IC-LoRA Loader output
    connected, learned slot embeddings are applied and consecutive negative temporal
    offsets are assigned (MSR training layout); without it, plain guides at offset 0.
    The guides are encoded ONCE here; the main upscale node resizes them to each
    chunk's grid and appends them as near-clean conditioning tokens."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25ReferenceParams",
            display_name="LTX25 Reference Params",
            category="model/latent/ltxv",
            description=(
                "Encode reference stills (each BATCH item = one reference) into LTX2.5 "
                "guide latents for 'LTX25 Ultimate Upscale'. Connect the output to the "
                "main node's 'reference_guides' input to pin identity/scene consistency "
                "across independently sampled chunks. Optionally connect an MSR IC-LoRA "
                "Loader output to use MSR slot embeddings and offsets."
            ),
            search_aliases=["ltx25 reference", "ltx25 msr params", "reference image guide", "ic-lora params"],
            inputs=[
                io.Image.Input("ref_images",
                               tooltip="Reference stills; each BATCH item is one reference (order = MSR slot order). Encoded once at this node."),
                io.Vae.Input("ref_vae",
                             tooltip="The LTX2.5 VIDEO VAE used to encode the references."),
                LTX_MSR_PARAM.Input("msr_parameters", optional=True,
                                    tooltip="Optional output of an MSR IC-LoRA loader: either this pack's 'LTX25 IC-LoRA Loader (MSR)' or the ComfyUI-LTX2.5-MSR package's 'IC-LoRA Loader' (same wire type). Adds learned slot embeddings and consecutive negative temporal offsets (MSR training layout). Leave unconnected for plain guides at offset 0."),
                io.Float.Input("ref_strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Reference guide conditioning strength: noise_mask value = 1 - strength (1.0 = guides stay fully clean/frozen; lower values let them drift slightly)."),
                io.Combo.Input("ref_frames", options=["25", "33"], default="33",
                               tooltip="Pixel frames each still is repeated to before encoding (25 -> 4 latent frames per reference, 33 -> 5)."),
            ],
            outputs=[
                LTX25_REF_GUIDES.Output("reference_guides",
                                        tooltip="Encoded reference guides consumed by 'LTX25 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, ref_images, ref_vae, msr_parameters=None,
                ref_strength=1.0, ref_frames="33"):
        if _ltx_nodes is None:
            raise RuntimeError("This ComfyUI build does not expose comfy_extras.nodes_lt (LTX guide support).")
        n = int(ref_images.shape[0])
        if n < 1:
            raise ValueError("ref_images must contain at least one image")
        ref_frames_n = int(ref_frames)
        if ref_frames_n not in (25, 33):
            raise ValueError(f"ref_frames must be 25 or 33, got {ref_frames}")
        msr_enabled = msr_parameters is not None
        dsf = 1
        if msr_enabled:
            try:
                dsf = max(1, round(float(msr_parameters.get("reference_downscale_factor", 1))))
            except (TypeError, ValueError):
                dsf = 1
            if dsf != 1:
                raise ValueError(
                    f"reference_downscale_factor={dsf} MSR LoRAs are not supported in "
                    "the chunked pipeline; use a factor-1 MSR checkpoint.")
        # Encode each still at its own pixel size snapped to the VAE grid; the main
        # node spatially resizes the resulting latents to each chunk's grid anyway.
        _, width_scale, height_scale = ref_vae.downscale_index_formula
        h_px, w_px = int(ref_images.shape[1]), int(ref_images.shape[2])
        latent_h = max(1, round(h_px / height_scale))
        latent_w = max(1, round(w_px / width_scale))
        guides = []
        for idx in range(n):
            guide, scale_factors = ltx_encode_reference(
                ref_vae, latent_h, latent_w, ref_images[idx:idx + 1], ref_frames_n)
            if msr_enabled:
                emb = ltx_msr_slot_embedding(
                    msr_parameters["slot_state"], idx + 1, guide.device, guide.dtype)
                if emb.numel() != guide.shape[1]:
                    raise ValueError(
                        f"MSR slot embedding dim {emb.numel()} != latent channels {guide.shape[1]}")
                guide = guide + emb.view(1, -1, 1, 1, 1)
            guides.append(guide.cpu().contiguous())
        return io.NodeOutput({
            "guides": guides,
            "offsets": [-(n - idx) for idx in range(n)] if msr_enabled else [0] * n,
            "scale_factors": scale_factors,
            "strength": float(ref_strength),
        })


# ---------------------------------------------------------------------------
# LTX2.5 main node
# ---------------------------------------------------------------------------

class LTX25UltimateUpscale(io.ComfyNode):
    """One node for the full LTX2.5 latent re-enhancement pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25UltimateUpscale",
            display_name="LTX25 Ultimate Upscale",
            category="model/latent/ltxv",
            description=(
                "Re-sample an already-denoised LTX2.5 AV latent through the full "
                "auto pipeline in one node: temporal split (outer loop) -> latent "
                "upscale (per chunk: 2x model upscale then resized to the target "
                "width/height) -> spatial split (inner loop) -> per-tile "
                "sampling with preview -> spatial stitch -> temporal stitch. "
                "When a temporal split is used, the next chunk's overlap is anchored "
                "to the previous chunk's re-sampled result (LTX image-to-video "
                "noise_mask) and chunks are joined by cross-fade; 'temporal_split_param.anchor_mode' "
                "selects the strategy: 'full' pins the whole overlap band (original behaviour), "
                "'first_frame' pins only the first token and blends across the band (H3-style), "
                "'ramp' applies a linear temporal fade across the band. "
                "OPTIONAL reference guides: connect 'reference_guides' from 'LTX25 Reference Params' "
                "to anchor identity/scene consistency across chunks - the encoded reference "
                "stills are appended to every chunk as near-clean guide tokens (native "
                "LTXVAddGuide mechanism), optionally with MSR slot embeddings when an "
                "IC-LoRA Loader output is wired into the param node. "
                "Audio: by default the INPUT audio is carried unchanged (bypass_audio); "
                "disable it to let the model RE-SAMPLE audio (with spatial tiling the first "
                "tile's audio is taken per time block, chunks cross-faded). "
                "'latent_upscale_param', 'temporal_split_param' and "
                "'spatial_split_param' are optional - leave any unconnected to "
                "skip that stage (no upscale / single chunk / full-chunk sampling)."
            ),
            search_aliases=["ltx25 ultimate upscale", "ltx ultimate upscale", "ltx enhance", "ltx reupscale"],
            inputs=[
                io.Model.Input("model", tooltip="The LTX2.5 diffusion model used to re-sample every chunk/tile (guider is built internally)."),
                io.Conditioning.Input("conditioning",
                                      tooltip="Conditioning used to generate this latent (LTXVConditioning with frame_rate). Passed through unchanged to every chunk/tile (T2V mode, no spatial keyframe cropping)."),
                io.Latent.Input("latent", tooltip="Denoised LTX2.5 AV latent to enhance (nested video+audio)."),
                io.Noise.Input("noise", tooltip="Noise source; one noise tensor is generated per piece."),
                io.Sampler.Input("sampler", tooltip="Sampler used for every chunk/tile."),
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule used for every chunk/tile."),
                io.Conditioning.Input("negative", optional=True,
                                      tooltip="Negative conditioning. When connected, a CFGGuider is used with the 'cfg' value; otherwise a basic guider (positive only)."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="CFG scale used when 'negative' is connected."),
                LTX_UPSCALE_PARAM.Input("latent_upscale_param", optional=True,
                                        tooltip="Output of 'LTX25 Latent Upscale Params'. Leave unconnected to skip upscaling."),
                LTX_TEMPORAL_PARAM.Input("temporal_split_param", optional=True,
                                          tooltip="Output of 'LTX25 Temporal Split Params'. Leave unconnected to process the latent as a single chunk. When connected, the next chunk's overlap is anchored to the previous chunk (strategy selected by 'anchor_mode': full band / first frame only / temporal ramp) and joined by cross-fade."),
                LTX_SPATIAL_PARAM.Input("spatial_split_param", optional=True,
                                         tooltip="Output of 'LTX25 Spatial Split Params'. Leave unconnected to sample each chunk whole (no tiling)."),
                LTX25_REF_GUIDES.Input("reference_guides", optional=True,
                                       tooltip="Optional output of 'LTX25 Reference Params'. When connected, the encoded reference stills are appended to EVERY chunk as near-clean guide tokens (native LTXVAddGuide mechanism), pinning identity/scene consistency across independently sampled chunks. Leave unconnected for the previous behaviour."),
                io.Boolean.Input("bypass_audio", default=True,
                                  tooltip="Audio handling. True = the output audio is the INPUT audio carried unchanged (frozen, never re-sampled). False = the audio is RE-SAMPLED by the model; with spatial tiling the FIRST tile's audio is taken for each time block, and consecutive chunks are cross-faded. Re-sampling costs extra compute but lets the model regenerate audio for the enhanced video."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Upscaled, re-sampled, stitched LTX2.5 AV latent."),
                io.Dict.Output("segments_info",
                               tooltip="DEBUG ONLY. Per-chunk metadata: frame start/count, video token ranges, upscale applied."),
                io.Dict.Output("tiles_info",
                               tooltip="DEBUG ONLY. Per-chunk spatial grid metadata: offsets, tile extents, overlaps, stitching mode."),
            ],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0,
                temporal_split_param=None, spatial_split_param=None,
                latent_upscale_param=None, bypass_audio=True,
                reference_guides=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_ltx_av_latent(samples):
            raise ValueError("LTX25UltimateUpscale expects an LTX2.5 AV latent (nested video [B,128,T,H,W] + audio)")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("LTX25UltimateUpscale expects a single-video latent (batch 1)")

        if latent_upscale_param is not None and spatial_split_param is not None:
            tile_w = int(spatial_split_param["tile_width"])
            tile_h = int(spatial_split_param["tile_height"])
            up_w = int(latent_upscale_param.get("width") or video.shape[4] * 2)
            up_h = int(latent_upscale_param.get("height") or video.shape[3] * 2)
            if up_w < tile_w:
                raise ValueError(f"Upscale width ({up_w}) must be >= tile_width ({tile_w})")
            if up_h < tile_h:
                raise ValueError(f"Upscale height ({up_h}) must be >= tile_height ({tile_h})")

        tv = video.shape[2]

        if temporal_split_param is not None:
            chunk_length = int(temporal_split_param["chunk_length"])
            overlap = int(temporal_split_param["temporal_overlap"])
            bounds, frame_count = ltx_compute_segments(tv, chunk_length, overlap)
            anchor_strength = float(temporal_split_param.get("anchor_strength", 0.0) or 0.0)
            anchor_mode = str(temporal_split_param.get("anchor_mode") or "full")
        else:
            frame_count = ltx_frames_for_tokens(tv)
            bounds = [(0, 0, tv, frame_count)]
            anchor_strength = 0.0
            anchor_mode = "full"

        # --- Optional reference guides (LTX25ReferenceParams output) ---
        guides = reference_guides

        acc_v = None
        acc_a = None
        segments_debug = []
        tiles_debug = []

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            # LTX audio layout is (B, C, time, freq): the TIME axis is index 2,
            # aligned 1:1 with the video token axis. Slice that, never the freq axis.
            a1 = min(k1, audio.shape[2])
            chunk_a = audio[:, :, k0:a1].contiguous()

            upscaled = False
            if latent_upscale_param is not None:
                upscale_model = latent_upscale_param["upscale_model"]
                vae = latent_upscale_param["vae"]
                # offload diffusion model while upscaler is on GPU
                if hasattr(model, "clone_base_uuid"):
                    comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                    comfy.model_management.soft_empty_cache()
                chunk_v = ltx_upscale_latent(chunk_v, upscale_model, vae)
                tw_ = int(latent_upscale_param.get("width"))
                th_ = int(latent_upscale_param.get("height"))
                chunk_v = ltx_resize_latent(chunk_v, tw_, th_)
                upscaled = True

            # --- Temporal keyframe anchoring (LTX image-to-video analogue of MMH3
            #     anchor_conditioning): pin part of the next chunk's overlap to the
            #     previous chunk's re-sampled frames via the noise_mask. Three modes
            #     (temporal_split_param.anchor_mode):
            #       'full'        - the whole overlap band is copied and pinned at
            #                       (1 - anchor_strength); the stitch cross-fade then
            #                       mixes identical content (original behaviour).
            #       'first_frame' - only the first latent token (~8 frames) is copied
            #                       and pinned; the rest re-samples freely and the
            #                       cross-fade blends across the full band width.
            #       'ramp'        - the band is initialised from the previous chunk and
            #                       the mask ramps linearly from (1 - anchor_strength)
            #                       at the seam to 1.0 at the band end.
            #     Cross-fade stitching is kept in every mode. ---
            vmask = None
            anchored = None
            if anchor_strength > 0.0 and i > 0 and acc_v is not None:
                n = acc_v.shape[2] - k0
                n = min(max(n, 0), chunk_v.shape[2])
                if n > 0:
                    Tv, H, W = chunk_v.shape[2], chunk_v.shape[3], chunk_v.shape[4]
                    vmask = torch.ones((1, 1, Tv, H, W), device=chunk_v.device, dtype=torch.float32)
                    prev = acc_v[:, :, k0:k0 + n].to(dtype=chunk_v.dtype, device=chunk_v.device)
                    if anchor_mode == "first_frame":
                        anchored = min(1, n)
                        chunk_v[:, :, :anchored] = prev[:, :, :anchored]
                        vmask[:, :, :anchored] = 1.0 - anchor_strength
                    elif anchor_mode == "ramp":
                        anchored = n
                        chunk_v[:, :, :n] = prev
                        w = torch.linspace(1.0 - anchor_strength, 1.0, n,
                                           device=chunk_v.device, dtype=torch.float32)
                        vmask[:, :, :n] = w.view(1, 1, n, 1, 1)
                    else:  # "full"
                        anchored = n
                        chunk_v[:, :, :n] = prev
                        vmask[:, :, :n] = 1.0 - anchor_strength

            cond_i = conditioning
            neg_i = negative

            # --- Optional reference guides (appended after anchoring so anchor
            #     indices reference pure video tokens, not guide frames) ---
            n_guide_frames = 0
            work_v = chunk_v
            video_mask = vmask  # may be None; branches handle the fallback
            if guides is not None:
                base_mask = vmask if vmask is not None else torch.ones(
                    (1, 1, chunk_v.shape[2], chunk_v.shape[3], chunk_v.shape[4]),
                    device=chunk_v.device, dtype=torch.float32)
                work_v, video_mask, cond_i, neg_i, n_guide_frames = ltx_append_guides(
                    chunk_v, base_mask, conditioning, negative, guides)

            if spatial_split_param is not None:
                chunk_out_v, chunk_out_a, tile_info = ltx_spatial_process(
                    work_v, chunk_a, cond_i, spatial_split_param,
                    model, noise, sampler, sigmas, neg_i, cfg, video_mask, bypass_audio,
                )
                tile_info = dict(tile_info)
                tile_info["chunk"] = i
                tiles_debug.append(tile_info)
            else:
                piece = {"samples": comfy.nested_tensor.NestedTensor((work_v, chunk_a))}
                # Audio mask: 0 = frozen (bypass, carried unchanged), 1 = re-sampled.
                # Always attach a nested noise_mask (video + audio) so the mask
                # structure matches the nested latent. Video is anchored (vmask) when
                # a temporal anchor exists, else fully re-sampled (ones).
                amask = torch.zeros_like(chunk_a) if bypass_audio else torch.ones_like(chunk_a)
                vmask_out = video_mask if video_mask is not None else torch.ones(
                    (1, 1, work_v.shape[2], work_v.shape[3], work_v.shape[4]),
                    device=work_v.device, dtype=torch.float32)
                piece["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask_out, amask))
                out = sample_piece(piece, cond_i, model, noise, sampler, sigmas, neg_i, cfg)
                chunk_out_v = out.tensors[0]
                chunk_out_a = chunk_a if bypass_audio else out.tensors[1]

            # Strip appended guide frames from the sampled result (guides sit at
            # the END of the latent sequence after ltx_append_guides).
            if n_guide_frames > 0:
                chunk_out_v = chunk_out_v[:, :, :chunk_v.shape[2]].contiguous()

            acc_v, acc_a = ltx_temporal_append(acc_v, acc_a, chunk_out_v, chunk_out_a, i, k0)

            segments_debug.append({
                "chunk": i,
                "frame_start": f0,
                "frame_count": f1 - f0,
                "video_tokens": [k0, k1],
                "upscaled": upscaled,
                "anchor_mode": anchor_mode if i > 0 else None,
                "anchored_tokens": anchored,
                "guide_frames": n_guide_frames,
                "spatial_h": work_v.shape[3],
                "spatial_w": work_v.shape[4],
            })

        if bypass_audio:
            # The LTX AV model ignores audio_denoise_mask (av_model._process_input
            # applies the mask to video only, then patchifies audio with no mask), so
            # re-sample can never freeze audio. For bypass we carry the ORIGINAL
            # input audio verbatim, merged in one shot here - no per-chunk slicing,
            # audio mask, or cross-fade is needed. The temporal upscale is spatial
            # only, so the input audio token count matches the (re-sampled) video.
            if audio.shape[2] == acc_v.shape[2]:
                acc_a = audio
            else:
                print(
                    f"[LTX25UltimateUpscale] bypass_audio: input audio has "
                    f"{audio.shape[2]} time tokens but the stitched video has "
                    f"{acc_v.shape[2]}; falling back to per-chunk audio."
                )
                # Fallback: the audio was not accumulated in bypass, so reconstruct it
                # by reusing the input audio truncated to the output video length.
                acc_a = audio[:, :, :acc_v.shape[2]].contiguous()

        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        out = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        return io.NodeOutput(out, segments_debug, tiles_debug)
