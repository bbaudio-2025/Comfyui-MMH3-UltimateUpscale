"""LTX2.5 Ultimate Upscale for the MMH3 Ultimate Upscale plugin.

The LTX2.5 block (spatial 32x, temporal 8x, video 128ch, 8k+1 frame grid),
including its MSR / IC-LoRA reference support which drives comfy_extras.nodes_lt
and mirrors the LTX2.5-MSR project (https://github.com/liconstudio/ComfyUI-LTX2.5-MSR).
Audio is carried unchanged (frozen, mask=0). Kept in its own module so that
upstream LTX algorithm updates only need to touch this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .nodes import (
    compute_spatial_grid,
    sample_piece,
    spatial_fade_mask,
    blend_weights,
    _crossfade,
    _solve_equal_tiles,
)

try:
    import comfy_extras.nodes_lt as _ltx_nodes
except Exception:
    _ltx_nodes = None

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
                        negative, cfg, vmask=None, bypass_audio=True, ref_guides=None):
    """Inner loop: spatial split -> per-tile sampling -> spatial stitch.
    Mirrors MMH3 spatial_process, adapted for LTX (32x VAE). Audio is carried
    unchanged (frozen in every tile, never re-sampled). T2V conditioning has no
    spatial keyframes, so it is passed through uncropped. Reference guides
    (`ref_guides`, optional) are appended PER TILE - after the spatial crop -
    so their keyframe coordinates match each tile's own grid; the appended
    suffix is stripped from the sampled result before stitching. Returns
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

            cond_tile = cond  # T2V: no spatial keyframe cropping
            pos_t, neg_t = cond_tile, negative
            sample_v, sample_mv = tile, mv
            n_guide = 0
            if ref_guides is not None:
                # Append guides AFTER the spatial crop so their recorded keyframe
                # coordinates are against THIS tile's grid (per-tile fresh conds;
                # conditioning_set_values never mutates the pristine inputs).
                sample_v, sample_mv, pos_t, neg_t, n_guide = ltx_append_guides(
                    tile, mv, cond_tile, negative, ref_guides)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((sample_v, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((sample_mv, ma)),
            }

            out = sample_piece(piece, pos_t, model, noise, sampler, sigmas, neg_t, cfg)
            tile_v = out.tensors[0]
            if n_guide > 0:
                # Strip the appended guide suffix (guides sit at the END).
                tile_v = tile_v[:, :, :t].contiguous()
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
            description="Bundle the spatial tile settings for the 'LTX25 Ultimate Upscale' node: tile size, per-axis overlap and fade, and seam stitching rules (inner loop). Two tile sizing modes: explicit pixel sizes, or a row/column count with an auto-solved equal-size tile grid (all tiles identical, edge tiles included); the resolved tile size is exposed as tile_width/tile_height outputs.",
            search_aliases=["ltx25 spatial param", "ltx tile param"],
            inputs=[
                io.Int.Input("upscale_width", default=1024, min=32, max=100000, step=32,
                             tooltip="[rows_cols mode] Overall upscaled frame WIDTH in PIXELS that gets split into grid_cols equal-size tile columns. Must be a multiple of 32 and must match the width set in 'LTX25 Latent Upscale Params'. Ignored in specific_size mode."),
                io.Int.Input("upscale_height", default=1024, min=32, max=100000, step=32,
                             tooltip="[rows_cols mode] Overall upscaled frame HEIGHT in PIXELS that gets split into grid_rows equal-size tile rows. Must be a multiple of 32 and must match the height set in 'LTX25 Latent Upscale Params'. Ignored in specific_size mode."),
                io.Combo.Input("tile_size_mode", options=["specific_size", "rows_cols"], default="specific_size",
                               tooltip="How the tile size is determined. 'specific_size' (default): use tile_width/tile_height below. 'rows_cols': split the frame given by upscale_width/upscale_height into grid_rows x grid_cols EQUAL-SIZE tiles (edge tiles included) - the per-axis overlap is auto-solved (multiple of 32px, one LTX latent token) so every tile ends up exactly the same size; errors out if the solved tiles would be smaller than min_tile_size."),
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="[specific_size mode] Tile width in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32 (LTX VAE 32x)."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="[specific_size mode] Tile height in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("grid_rows", default=2, min=1, max=9, step=1,
                             tooltip="[rows_cols mode] Number of tile ROWS along the height axis (1-9)."),
                io.Int.Input("grid_cols", default=2, min=1, max=9, step=1,
                             tooltip="[rows_cols mode] Number of tile COLUMNS along the width axis (1-9)."),
                io.Int.Input("spatial_w_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Horizontal overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile width. In rows_cols mode this is the DESIRED overlap; the node auto-solves the actual value. If the solved tile size would not be a multiple of 32 (the model's 2x2 latent patch grid, which eliminates seams), the overlap is automatically increased by 32px and the tiles re-solved at the next valid 32px grid alignment."),
                io.Int.Input("spatial_h_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Vertical overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile height. In rows_cols mode this is the DESIRED overlap; the node auto-solves the actual value. If the solved tile size would not be a multiple of 32 (the model's 2x2 latent patch grid, which eliminates seams), the overlap is automatically increased by 32px and the tiles re-solved at the next valid 32px grid alignment."),
                io.Int.Input("fade_width", default=32, min=0, max=100000, step=32,
                             tooltip="Width in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0) + this FADE segment (interior side). fade_width sets the fade length; the frozen segment takes the rest. Default 32. Set to 0 to freeze the entire overlap strip. Clamped to the solved overlap in rows_cols mode."),
                io.Int.Input("fade_height", default=32, min=0, max=100000, step=32,
                             tooltip="Height in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. See fade_width. Clamped to the solved overlap in rows_cols mode."),
                io.Int.Input("min_tile_size", default=256, min=0, max=100000, step=32,
                             tooltip="Minimum PIXEL size of edge tiles. If a leftover edge tile would be smaller, the last tile is pulled back until it reaches at least this size. Must not exceed the tile size. In rows_cols mode an error is raised if the solved tile size falls below this."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier",
                               tooltip="Who wins each shared overlap band when stitching. 'earlier' (default): the already-stitched content wins. 'later': the re-sampled tile wins."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="How the overlap band transitions when stitching: linear cross-fade (default), smoothstep (eased), overwrite (whole band from the overlap_mode side), midpoint (hard switch at the band's middle)."),
            ],
            outputs=[
                LTX_SPATIAL_PARAM.Output("spatial_split_param",
                    tooltip="Spatial split settings consumed by 'LTX25 Ultimate Upscale'."),
                io.Int.Output("tile_width",
                    tooltip="Resolved tile width in PIXELS: the validated input in specific_size mode, or the equal-tile solution computed from upscale_width/grid_cols in rows_cols mode."),
                io.Int.Output("tile_height",
                    tooltip="Resolved tile height in PIXELS: the validated input in specific_size mode, or the equal-tile solution computed from upscale_height/grid_rows in rows_cols mode."),
            ],
        )

    @classmethod
    def execute(cls, upscale_width, upscale_height, tile_size_mode, tile_width,
                tile_height, grid_rows, grid_cols,
                spatial_w_overlap, spatial_h_overlap,
                fade_width, fade_height, min_tile_size, overlap_mode,
                overlap_blend) -> io.NodeOutput:
        if tile_size_mode == "rows_cols":
            # Equal-size grid solved HERE so the tile outputs are always real.
            # Overlap granularity is one latent token: 32px for LTX.
            for name, v in (("upscale_width", upscale_width), ("upscale_height", upscale_height)):
                if v <= 0 or v % 32 != 0:
                    raise ValueError(f"'{name}' must be a positive multiple of 32 pixels; got {v}.")
            tw, ow = _solve_equal_tiles(upscale_width, grid_cols, spatial_w_overlap, LTX_VAE_DOWNSAMPLE)
            th, oh = _solve_equal_tiles(upscale_height, grid_rows, spatial_h_overlap, LTX_VAE_DOWNSAMPLE)
            if tw < min_tile_size or th < min_tile_size:
                raise ValueError(
                    f"rows_cols mode: solved tile size is {th}x{tw}px "
                    f"(grid {grid_rows}x{grid_cols} over {upscale_height}x{upscale_width}px), "
                    f"which is smaller than min_tile_size ({min_tile_size}px). "
                    f"Reduce grid_rows/grid_cols, or lower min_tile_size to at most "
                    f"{min(tw, th)}px.")
            param = {
                "tile_width": tw, "tile_height": th,
                "spatial_w_overlap": ow, "spatial_h_overlap": oh,
                "fade_width": min(fade_width, ow),
                "fade_height": min(fade_height, oh),
                "min_tile_size": min_tile_size,
                "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
                "tile_size_mode": tile_size_mode,
                "grid_rows": grid_rows, "grid_cols": grid_cols,
            }
            print(f"[LTX25 Spatial Split Params] rows_cols mode: {grid_rows}x{grid_cols} "
                  f"tiles of {th}x{tw}px over {upscale_height}x{upscale_width}px "
                  f"(overlap h={oh} w={ow}, fade h={param['fade_height']} w={param['fade_width']})")
            return io.NodeOutput(param, tw, th)

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
            "tile_size_mode": tile_size_mode,
        }
        return io.NodeOutput(param, tile_width, tile_height)


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

        # fail early if the upscale target is smaller than the spatial tile size
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

            # --- Optional reference guides. Appended AFTER anchoring so anchor
            #     indices reference pure video tokens. With spatial tiling the
            #     append must happen PER TILE (inside ltx_spatial_process) so the
            #     keyframe coordinates match each tile's own grid; only the
            #     whole-chunk branch pre-appends here. ---
            n_guide_frames = 0
            work_v = chunk_v
            video_mask = vmask  # may be None; branches handle the fallback
            if guides is not None and spatial_split_param is None:
                base_mask = vmask if vmask is not None else torch.ones(
                    (1, 1, chunk_v.shape[2], chunk_v.shape[3], chunk_v.shape[4]),
                    device=chunk_v.device, dtype=torch.float32)
                work_v, video_mask, cond_i, neg_i, n_guide_frames = ltx_append_guides(
                    chunk_v, base_mask, conditioning, negative, guides)

            if spatial_split_param is not None:
                chunk_out_v, chunk_out_a, tile_info = ltx_spatial_process(
                    chunk_v, chunk_a, conditioning, spatial_split_param,
                    model, noise, sampler, sigmas, negative, cfg, vmask, bypass_audio,
                    ref_guides=guides,
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
