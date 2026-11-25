"""H3 latent upscalers (3D and 2D) for MMH3 Ultimate Upscale.

Self-contained copy of the MiniMax H3 latent upscaler inference code taken
from the Comfyui_Minimax_h3_latent_Upscaler plugin (https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
so it works directly with the minimax_h3_latent_upscaler_3d_* and
minimax_h3_latent_upscaler_2d_* checkpoints. Kept in its own module so that if
the upstream project updates the algorithm, this file can be updated in
isolation without touching the plugin's business logic.

The architecture is auto-detected from the checkpoint's key layout:
  - 3D ("conv_in.weight", "in_blocks.N.in_layers.*"): pure Conv3d backbone.
  - 2D ("resizer.conv_in.weight", "temporal_blocks.N.*"): Conv2d backbone run per
    frame, with Conv1d-style temporal blocks interleaved every other block.

DELIBERATE DIVERGENCE FROM UPSTREAM - do not "sync" this away: the checkpoint
loader here accepts .safetensors only. Upstream also accepts pickle checkpoints
(.pth) through torch.load(..., weights_only=False), which is an arbitrary code
execution risk; weights_only=True is not a reliable mitigation on torch < 2.6
(CVE-2025-32434). See _SUPPORTED_MODEL_EXTENSIONS below.
"""

import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

# Spatial compression factor of the MiniMax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16

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

    def forward(self, x, scale=None, target_size=None, tier=None):
        # tier is the 2D model's sharpness control; the 3D model has none.
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


# ---------------------------------------------------------------------------
# 2D backbone (minimax_h3_latent_upscaler_2d_conv_v1 and siblings)
#
# Ported from upstream nodes/minimax_h3_latent_upscaler_2d.py. Same 24-channel
# latent space and the same normalisation stats as the 3D model, so it drops
# into the same pipeline. Differences:
#   - Conv2d runs per frame (b t folded into the batch); temporal Conv1d-style
#     blocks are interleaved every `temporal_every` blocks to keep motion
#     coherent. This is why it is ~half the parameters of the 3D model.
#   - The Fourier scale embedding may carry a 32-d "sharpness tier" branch
#     (6 tiers) that the 3D model does not have.
#   - Long clips are split into chunks with a halo equal to the temporal
#     receptive field, then stitched, so per-chunk sampling stays consistent.
#
# Blocks are built from the checkpoint's own entry list instead of assuming
# "a large-kernel block every N blocks": the released v1 checkpoint is 12+12
# plain ResBlocks with no large-kernel / SE / post-refine entries at all, and a
# count-based rule would either invent modules the weights do not have or miss
# the ones they do.
# ---------------------------------------------------------------------------

# Frames per chunk when the clip is long enough that chunking pays off. Below
# `chunk + halo` a single pass is both cheaper and exact, so it is used instead.
_CHUNK_FRAMES_2D = 32


class _FourierScaleEmbedding2D(nn.Module):
    """sin/cos scale encoding, optionally followed by a sharpness tier encoding.

    Layout matches the training script: [sin/cos(scale_h) x freq] +
    [sin/cos(scale_w) x freq] (+ [sin/cos(tier-1) x freq]). Phases are not
    multiplied by pi: with a pi factor, scales that differ by an even integer
    alias to the same encoding and the branch cannot tell 1.2x from 3.2x.
    """

    def __init__(self, embed_dim, num_frequencies=16, tier_dim=0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.tier_dim = int(tier_dim)
        in_dim = 4 * num_frequencies + self.tier_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Zero-init the tier columns: without a tier input the branch vanishes,
        # which is exactly the "no tier" behaviour of pre-tier weights.
        if self.tier_dim > 0:
            with torch.no_grad():
                self.linear[0].weight[:, 4 * num_frequencies:].zero_()

    def forward(self, scale_h, scale_w, tier=None):
        dev = self.linear[0].weight.device
        if isinstance(scale_h, (float, int)):
            scale_h = torch.tensor([float(scale_h)], device=dev)
        if isinstance(scale_w, (float, int)):
            scale_w = torch.tensor([float(scale_w)], device=dev)
        scale_h = scale_h.to(device=dev, dtype=torch.float32)
        scale_w = scale_w.to(device=dev, dtype=torch.float32)
        if scale_h.dim() == 0:
            scale_h = scale_h.unsqueeze(0)
        if scale_w.dim() == 0:
            scale_w = scale_w.unsqueeze(0)

        freqs = 2.0 ** torch.arange(self.num_frequencies, device=dev).float()
        emb_h = torch.cat([torch.sin(scale_h.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_h.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
        emb_w = torch.cat([torch.sin(scale_w.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_w.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
        emb = torch.cat([emb_h, emb_w], dim=1)

        if self.tier_dim > 0:
            # Phase is (tier - 1): tier 1 is the canonical zero point. "No tier
            # specified" must mean tier 1 (phase 0), not tier 0, which does not
            # exist in training and would be out of distribution.
            if tier is None:
                t = torch.ones(1, device=dev)
            elif isinstance(tier, (float, int)):
                t = torch.tensor([float(tier)], device=dev)
            else:
                t = tier.to(device=dev, dtype=torch.float32)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
            t = t - 1.0
            emb = torch.cat([emb,
                             torch.sin(t.unsqueeze(1) * freqs.unsqueeze(0)),
                             torch.cos(t.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

        return self.linear(emb.to(self.linear[0].weight.dtype))


class _CoordConv2D(nn.Module):
    """Appends normalised y/x coordinates as two extra channels. No parameters,
    so it never shows up in the checkpoint."""

    def forward(self, x):
        b, _, h, w = x.shape
        yc = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xc = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, yc, xc], dim=1)


class _SEBlock2D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.SiLU(True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class _LargeKernelBlock2D(nn.Module):
    def __init__(self, ch, ks=7):
        super().__init__()
        self.norm = _normalization(ch)
        self.dw = nn.Conv2d(ch, ch, ks, padding=ks // 2, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        return x + self.pw(self.dw(F.silu(self.norm(x))))


class _PostUpsampleRefine2D(nn.Module):
    def __init__(self, ch, n=2):
        super().__init__()
        layers = []
        for i in range(n):
            layers.append(nn.Conv2d(ch, ch, 3, padding=1))
            if i < n - 1:
                layers.append(nn.SiLU(True))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x):
        return x + self.refine(x)


class _MultiScaleTemporalConv2D(nn.Module):
    def __init__(self, ch, kernel_sizes=(3, 5, 9)):
        super().__init__()
        self.norm = _normalization(ch)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch),
                nn.Conv3d(ch, ch, 1),
            ) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv3d(ch * len(kernel_sizes), ch, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        b, _, t, _, _ = x.shape
        h = _flat_bt(x)
        h = _unflat_bt(self.norm(h), b, t)
        return x + self.fusion(torch.cat([br(F.silu(h)) for br in self.branches], dim=1))


class _TemporalConv2D(nn.Module):
    """Depthwise temporal conv + pointwise projection. Names must stay
    dwconv/pwconv to match the checkpoint keys."""

    def __init__(self, ch, ks=5):
        super().__init__()
        self.norm = _normalization(ch)
        self.dwconv = nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch)
        self.pwconv = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        b, _, t, _, _ = x.shape
        h = _flat_bt(x)
        h = _unflat_bt(self.norm(h), b, t)
        return x + self.pwconv(self.dwconv(F.silu(h)))


def _flat_bt(x):
    """(b, c, t, h, w) -> (b*t, c, h, w)"""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _unflat_bt(x, b, t):
    """(b*t, c, h, w) -> (b, c, t, h, w)"""
    c, h, w = x.shape[1:]
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class _ResBlockEmb2D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None,
                 use_se=False, se_reduction=8):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv2d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )
        self.se = _SEBlock2D(self.out_channels, se_reduction) if use_se else None

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        if self.se is not None:
            h = self.se(h)
        return self.skip(x) + h


def _bilinear_taps(k):
    """k 1D taps of length 3 (for input index i-1, i, i+1), align_corners=False."""
    taps = []
    for a in range(k):
        t = (a + 0.5) / k - 0.5
        lo = math.floor(t)
        frac = t - lo
        v = torch.zeros(3)
        v[lo + 1] = 1.0 - frac
        v[lo + 2] = frac
        taps.append(v)
    return taps


class _PSUpsample2D(nn.Module):
    """Single-ratio PixelShuffle(k) learned upsample (exclusive: no bilinear on
    top). Used only by checkpoints that ship a learned_up branch."""

    def __init__(self, channels, k=2):
        super().__init__()
        self.k = int(k)
        # 'replicate' matches training and reproduces the edge clamp of
        # bilinear interpolation ('zeros' would make borders systematically low).
        self.conv = nn.Conv2d(channels, self.k * self.k * channels, 3, padding=1,
                              padding_mode="replicate")
        self.ps = nn.PixelShuffle(self.k)
        self._init_bilinear_kernel()

    def _init_bilinear_kernel(self):
        k, c = self.k, self.conv.in_channels
        taps = _bilinear_taps(k)
        with torch.no_grad():
            w = torch.zeros_like(self.conv.weight)
            for a in range(k):
                for b in range(k):
                    ker = torch.outer(taps[a], taps[b])
                    idx = a * k + b
                    for ch in range(c):
                        w[ch * k * k + idx, ch] = ker
            self.conv.weight.copy_(w)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.ps(self.conv(x))


class _PSUpsampleBank2D(nn.Module):
    """One PS branch per trained integer ratio."""

    def __init__(self, channels, ks):
        super().__init__()
        self.ks = sorted({int(k) for k in ks})
        self.branches = nn.ModuleDict({str(k): _PSUpsample2D(channels, k) for k in self.ks})

    def select_k(self, scale):
        for k in self.ks:                       # exact match first
            if abs(k - scale) <= 1e-3:
                return k
        for k in self.ks:                       # else the smallest k above scale
            if k > scale:
                return k
        return self.ks[-1]

    def forward(self, x, scale):
        return self.branches[str(self.select_k(scale))](x)


def _build_2d_blocks(specs, channels, embed_dim, dropout, use_se, se_reduction):
    blocks = nn.ModuleList()
    for kind, kw in specs:
        if kind == "resblock":
            blocks.append(_ResBlockEmb2D(channels, embed_dim, dropout,
                                         use_se=use_se, se_reduction=se_reduction))
        elif kind == "large_kernel":
            blocks.append(_LargeKernelBlock2D(channels, kw["ks"]))
        else:  # pragma: no cover - guarded by _detect_arch_2d
            raise ValueError(f"unknown 2D block kind: {kind}")
    return blocks


class _LatentResizer2D(nn.Module):
    def __init__(self, in_channels=24, channels=640, dropout=0.1,
                 use_dual_scale=True, use_coord=True, use_se=False, se_reduction=8,
                 embed_dim=128, dual_embed_freq=16, tier_dim=0,
                 in_specs=(), out_specs=(), post_refine_layers=0,
                 learned_upsample_ks=None):
        super().__init__()
        self.use_dual_scale = use_dual_scale
        self.use_coord = use_coord
        self.tier_dim = int(tier_dim)

        if use_dual_scale:
            self.embed = _FourierScaleEmbedding2D(embed_dim, dual_embed_freq, tier_dim=self.tier_dim)
        else:
            self.embed = nn.Sequential(
                nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.coord_conv = _CoordConv2D() if use_coord else None
        self.conv_in = nn.Conv2d(in_channels + 2 if use_coord else in_channels,
                                 channels, 3, padding=1)

        self.learned_up = (_PSUpsampleBank2D(channels, learned_upsample_ks)
                           if learned_upsample_ks else None)
        self.in_blocks = _build_2d_blocks(in_specs, channels, embed_dim, dropout,
                                          use_se, se_reduction)
        self.out_blocks = _build_2d_blocks(out_specs, channels, embed_dim, dropout,
                                           use_se, se_reduction)
        self.post_up = (_PostUpsampleRefine2D(channels, post_refine_layers)
                        if post_refine_layers else None)

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def _upsample_mid(self, x, size):
        """Mid upsample: learned PS and bilinear are mutually exclusive. PS is
        only trained on exact integer ratios, so anything else falls back to
        bilinear (and the result is cropped, never resampled)."""
        if self.learned_up is not None:
            s = min(size[0] / x.shape[-2], size[1] / x.shape[-1])
            k = self.learned_up.select_k(s)
            if (abs(k - s) <= 1e-3
                    and x.shape[-2] * k >= size[0]
                    and x.shape[-1] * k >= size[1]):
                y = self.learned_up(x, s)
                if y.shape[-2] != size[0] or y.shape[-1] != size[1]:
                    y = y[..., :size[0], :size[1]]
                return y
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x, scale_h=None, scale_w=None, target_hw=None, tier=None):
        if target_hw is not None:
            size = target_hw
        elif scale_h is not None and scale_w is not None:
            size = (int(round(x.shape[-2] * scale_h)), int(round(x.shape[-1] * scale_w)))
        else:
            return x

        if size == x.shape[-2:]:
            return x

        if scale_h is None:
            scale_h = size[0] / x.shape[-2]
        if scale_w is None:
            scale_w = size[1] / x.shape[-1]

        if self.use_dual_scale:
            emb = self.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        if self.use_coord:
            x = self.coord_conv(x)
        x = self.conv_in(x)

        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb2D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = self._upsample_mid(x, size)
        if self.post_up:
            x = self.post_up(x)

        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb2D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        return self.conv_out(F.silu(self.norm_out(x)))


class _VideoLatentResizer2D(nn.Module):
    """5D wrapper: folds (b t) for the 2D backbone and interleaves the temporal
    blocks. Exposes the same call signature as the 3D resizer so upscale_video
    can drive either model."""

    def __init__(self, resizer, temporal_every=2, n_temporal_blocks=0,
                 use_multiscale_temporal=False, temporal_kernel_sizes=(3, 5, 9),
                 temporal_kernel=5, chunk_frames=_CHUNK_FRAMES_2D):
        super().__init__()
        self.resizer = resizer
        self.temporal_every = temporal_every
        self.chunk_frames = chunk_frames
        self.temporal_blocks = nn.ModuleList()
        for _ in range(n_temporal_blocks):
            if use_multiscale_temporal:
                self.temporal_blocks.append(
                    _MultiScaleTemporalConv2D(resizer.conv_in.out_channels, temporal_kernel_sizes))
            else:
                self.temporal_blocks.append(
                    _TemporalConv2D(resizer.conv_in.out_channels, temporal_kernel))

    def _temporal_radius(self):
        """Half-width of the total temporal receptive field, in frames."""
        tk = 0
        for b in self.temporal_blocks:
            if isinstance(b, _TemporalConv2D):
                tk = max(tk, b.dwconv.weight.shape[2])
            elif isinstance(b, _MultiScaleTemporalConv2D):
                tk = max(tk, max(br[0].weight.shape[2] for br in b.branches))
        return max(tk // 2, 1)

    def _halo(self):
        if not self.temporal_blocks:
            return 0
        n_temporal = sum(1 for i in range(len(self.resizer.in_blocks))
                         if i % self.temporal_every == 0)
        n_temporal += sum(1 for i in range(len(self.resizer.out_blocks))
                          if i % self.temporal_every == 0)
        return self._temporal_radius() * max(n_temporal, 1)

    def forward(self, x, scale=None, target_size=None, tier=None):
        b, _, t, h, w = x.shape

        # Accept the 3D call convention (t, h, w) as well as (h, w).
        if target_size is not None:
            size = tuple(target_size[-2:])
        elif scale is not None:
            size = (int(round(h * scale)), int(round(w * scale)))
        else:
            size = (h, w)

        if size == (h, w):
            return x

        halo = self._halo()
        # Chunking only pays off once a clip is longer than one chunk plus its
        # halo; before that, two overlapping passes would double the cost for
        # no memory benefit.
        if halo == 0 or t > self.chunk_frames + halo:
            return self._forward_chunked(x, size, tier=tier)
        return self._forward_full(x, size, tier=tier)

    def _forward_chunked(self, x, size, tier=None):
        b, c, t, h, w = x.shape
        halo = self._halo()
        chunk = max(1, int(self.chunk_frames))
        out_full = torch.zeros(b, c, t, size[0], size[1], device=x.device, dtype=x.dtype)

        start = 0
        while start < t:
            seg_start = start
            seg_end = min(t, start + chunk)
            lo = max(0, seg_start - halo)     # real frames, not zero padding
            hi = min(t, seg_end + halo)
            seg = x[:, :, lo:hi].contiguous()
            seg_out = self._forward_full(seg, size, tier=tier)
            s0 = seg_start - lo
            out_full[:, :, seg_start:seg_end] = seg_out[:, :, s0:s0 + (seg_end - seg_start)]
            start += chunk
            del seg, seg_out
        return out_full

    def _forward_full(self, x, size, tier=None):
        b, _, t, h, w = x.shape
        x_flat = _flat_bt(x)
        scale_h = size[0] / h
        scale_w = size[1] / w

        if not self.temporal_blocks:
            out = self.resizer(x_flat, scale_h=scale_h, scale_w=scale_w,
                               target_hw=size, tier=tier)
            return _unflat_bt(out, b, t)

        if self.resizer.use_dual_scale:
            emb = self.resizer.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.resizer.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        feat = self.resizer.coord_conv(x_flat) if self.resizer.use_coord else x_flat
        out = self.resizer.conv_in(feat)

        for i, block in enumerate(self.resizer.in_blocks):
            if isinstance(block, _ResBlockEmb2D):
                out = block(out, emb.expand(b * t, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0:
                out = _flat_bt(self.temporal_blocks[0](_unflat_bt(out, b, t)))

        out = self.resizer._upsample_mid(out, size)
        if self.resizer.post_up:
            out = self.resizer.post_up(out)

        for i, block in enumerate(self.resizer.out_blocks):
            if isinstance(block, _ResBlockEmb2D):
                out = block(out, emb.expand(b * t, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0 and len(self.temporal_blocks) > 1:
                out = _flat_bt(self.temporal_blocks[1](_unflat_bt(out, b, t)))

        out = self.resizer.conv_out(F.silu(self.resizer.norm_out(out)))
        return _unflat_bt(out, b, t)


_MODEL_CACHE = {}

# Only safetensors checkpoints are accepted, and they are rejected *before* the
# file is read: torch.load() on a pickle checkpoint can execute arbitrary code,
# and weights_only=True is not a sufficient defence on torch < 2.6
# (CVE-2025-32434). ".sft" is the other extension ComfyUI core accepts for
# safetensors (comfy.utils.load_torch_file), so it is allowed here as well.
_SUPPORTED_MODEL_EXTENSIONS = (".safetensors", ".sft")

# Formats that used to be accepted and are now refused on purpose. Only used to
# print an actionable hint instead of a silent "no models found".
_REJECTED_MODEL_EXTENSIONS = (".pth", ".pt", ".pt2", ".ckpt", ".bin", ".pkl")

_CONVERT_HINT = (
    "Convert it once with:\n"
    "    import torch, safetensors.torch\n"
    "    sd = torch.load('model.pth', map_location='cpu', weights_only=True)\n"
    "    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd\n"
    "    safetensors.torch.save_file(sd, 'model.safetensors')\n"
    "If weights_only=True fails, the checkpoint is not a plain state dict and "
    "should not be trusted."
)


def _model_ext(name):
    """Lower-cased extension, so the scanner and the loader agree on case."""
    return os.path.splitext(name)[1].lower()


def _model_dirs():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)


def _get_models_dir():
    """First registered models dir. Used for messages only - loading resolves a
    name across every registered dir (see load_upscale_model)."""
    return _model_dirs()[0]


def _scan_models():
    """List usable checkpoints across ALL registered latent_upscale_models dirs.

    Uses folder_paths.get_filename_list (same as the upstream node) rather than a
    glob on a single directory, so models registered via extra_model_paths.yaml
    and sub-folders show up too, and the extension filter is case-insensitive.
    """
    all_names = folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
    names = sorted(n for n in all_names
                   if _model_ext(n) in _SUPPORTED_MODEL_EXTENSIONS)
    rejected = sorted(n for n in all_names
                      if _model_ext(n) in _REJECTED_MODEL_EXTENSIONS)
    if rejected:
        print(f"[MMH3-UltimateUpscale] Ignored {len(rejected)} pickle checkpoint(s) "
              f"in the latent_upscale_models folder (only .safetensors is supported): "
              f"{', '.join(rejected)}")
    if not names:
        return [f"(no .safetensors upscale models found in: {', '.join(_model_dirs())})"]
    return names


def _load_raw_sd(path):
    """Read a checkpoint into a raw state dict. Only safetensors is accepted;
    pickle formats are refused before the file is opened."""
    if _model_ext(path) not in _SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Refusing to load '{os.path.basename(path)}': this node only accepts "
            f"{' / '.join(_SUPPORTED_MODEL_EXTENSIONS)} checkpoints. Pickle formats "
            f"(.pth / .pt / .ckpt) can execute arbitrary code when loaded, so they "
            f"are rejected before the file is read. {_CONVERT_HINT}"
        )
    from safetensors.torch import load_file
    sd = load_file(path, device='cpu')
    if isinstance(sd, dict) and isinstance(sd.get('model'), dict):
        # Pickle checkpoints sometimes wrap the weights in {"model": ...}. Guard
        # on the value type so a tensor literally named "model" can't break this.
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


def _is_2d_sd(sd):
    """2D checkpoints keep the backbone under 'resizer.' and the temporal blocks
    at the top level; 3D checkpoints use bare 'conv_in.' / 'in_blocks.N.'."""
    return any(k.startswith("resizer.") or k.startswith("temporal_blocks.") for k in sd)


def _block_specs_2d(sd, prefix):
    """Ordered block list for one side, read straight off the checkpoint's own
    entry indices (so large-kernel entries land at the exact positions the
    weights were saved with)."""
    idx = sorted({int(m.group(1)) for k in sd
                  for m in [re.match(rf"{re.escape(prefix)}\.(\d+)\.", k)] if m})
    specs = []
    for i in idx:
        if f"{prefix}.{i}.in_layers.0.weight" in sd:
            specs.append(("resblock", {}))
        elif f"{prefix}.{i}.dw.weight" in sd:
            specs.append(("large_kernel", {"ks": int(sd[f"{prefix}.{i}.dw.weight"].shape[2])}))
        else:
            raise ValueError(f"Unrecognised block at {prefix}.{i} in this 2D upscaler checkpoint.")
    return specs


def _detect_arch_2d(sd):
    cfg = {
        "in_channels": 24, "channels": 640, "dropout": 0.1,
        "use_dual_scale": True, "use_coord": False, "use_se": False, "se_reduction": 8,
        "embed_dim": 128, "dual_embed_freq": 16, "tier_dim": 0,
        "in_specs": [], "out_specs": [], "post_refine_layers": 0,
        "learned_upsample_ks": [], "temporal_every": 2, "n_temporal_blocks": 0,
        "use_multiscale_temporal": False, "temporal_kernel_sizes": (3, 5, 9),
        "temporal_kernel": 5,
    }

    w_in = sd.get("resizer.conv_in.weight")
    w_out = sd.get("resizer.conv_out.weight")
    if w_in is None or w_out is None:
        raise ValueError("2D upscaler checkpoint is missing resizer.conv_in/conv_out.")
    cfg["channels"] = int(w_in.shape[0])
    real_in = int(w_out.shape[0])
    # CoordConv2D has no parameters, so it is inferred from the input width:
    # 24 latent channels + 2 coordinate channels.
    cfg["use_coord"] = int(w_in.shape[1]) == real_in + 2
    cfg["in_channels"] = real_in

    cfg["in_specs"] = _block_specs_2d(sd, "resizer.in_blocks")
    cfg["out_specs"] = _block_specs_2d(sd, "resizer.out_blocks")
    cfg["use_se"] = any(".se.fc." in k for k in sd)

    post_idx = {int(m.group(1)) for k in sd
                for m in [re.match(r"resizer\.post_up\.refine\.(\d+)\.weight", k)] if m}
    cfg["post_refine_layers"] = len(post_idx)

    lu_ks = {int(m.group(1)) for k in sd
             for m in [re.match(r"resizer\.learned_up\.branches\.(\d+)\.", k)] if m}
    cfg["learned_upsample_ks"] = sorted(lu_ks)

    cfg["use_dual_scale"] = "resizer.embed.linear.0.weight" in sd
    emb_k = sd.get("resizer.embed.linear.0.weight")
    if emb_k is not None:
        cfg["embed_dim"] = int(emb_k.shape[0])
        in_dim = int(emb_k.shape[1])
        # 64 = 4*16 (scale only), 96 = 4*16 + 32 (scale + sharpness tier).
        if in_dim == 96:
            cfg["dual_embed_freq"], cfg["tier_dim"] = 16, 32
        else:
            cfg["dual_embed_freq"], cfg["tier_dim"] = in_dim // 4, 0

    t_idx = {int(m.group(1)) for k in sd
             for m in [re.match(r"temporal_blocks\.(\d+)\.", k)] if m}
    cfg["n_temporal_blocks"] = len(t_idx)
    cfg["temporal_every"] = 2 if t_idx else 0
    cfg["use_multiscale_temporal"] = "temporal_blocks.0.branches.0.0.weight" in sd
    if cfg["use_multiscale_temporal"]:
        ks = {int(sd[k].shape[2]) for k in sd
              if re.match(r"temporal_blocks\.0\.branches\.\d+\.0\.weight", k)}
        if ks:
            cfg["temporal_kernel_sizes"] = tuple(sorted(ks))
    else:
        # Older weights name these dw/pw, current ones dwconv/pwconv.
        for key in ("temporal_blocks.0.dwconv.weight", "temporal_blocks.0.dw.weight"):
            if key in sd:
                cfg["temporal_kernel"] = int(sd[key].shape[2])
                break
    return cfg


def _build_model_2d(name, sd):
    cfg = _detect_arch_2d(sd)
    if cfg["in_channels"] != 24:
        raise ValueError(
            f"Checkpoint '{name}' is not an H3 latent upscaler "
            f"(expected 24 input channels, got {cfg['in_channels']})."
        )
    resizer = _LatentResizer2D(
        in_channels=cfg["in_channels"], channels=cfg["channels"], dropout=cfg["dropout"],
        use_dual_scale=cfg["use_dual_scale"], use_coord=cfg["use_coord"],
        use_se=cfg["use_se"], se_reduction=cfg["se_reduction"],
        embed_dim=cfg["embed_dim"], dual_embed_freq=cfg["dual_embed_freq"],
        tier_dim=cfg["tier_dim"], in_specs=cfg["in_specs"], out_specs=cfg["out_specs"],
        post_refine_layers=cfg["post_refine_layers"],
        learned_upsample_ks=cfg["learned_upsample_ks"],
    )
    model = _VideoLatentResizer2D(
        resizer, temporal_every=cfg["temporal_every"],
        n_temporal_blocks=cfg["n_temporal_blocks"],
        use_multiscale_temporal=cfg["use_multiscale_temporal"],
        temporal_kernel_sizes=cfg["temporal_kernel_sizes"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(sd, strict=True)
    print(f"[MMH3-UltimateUpscale] 2D architecture: {cfg['channels']} ch, "
          f"{len(cfg['in_specs'])}+{len(cfg['out_specs'])} blocks, "
          f"{cfg['n_temporal_blocks']} temporal, tier_dim={cfg['tier_dim']}")
    return model


def load_upscale_model(name, device, precision):
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key].to(device)

    # Resolve the name against every registered latent_upscale_models dir, not
    # just the first one, so extra_model_paths.yaml entries are usable.
    path = folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {name} (searched: {', '.join(_model_dirs())})")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)

    if _is_2d_sd(up_sd):
        model = _build_model_2d(name, up_sd)
        arch = "2D"
    else:
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
        arch = "3D"

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    _MODEL_CACHE[cache_key] = model
    print(f"[MMH3-UltimateUpscale] Loaded {arch} upscale model: {name}")
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
    """Upscale one chunk's video latent with the H3 latent upscaler (3D or 2D,
    auto-detected). Audio untouched.

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
        raise ValueError(
            "No usable H3 upscale model selected. Put a .safetensors checkpoint "
            "(e.g. minimax_h3_latent_upscaler_3d_fp16.safetensors) into the "
            "latent_upscale_models folder and restart ComfyUI. Pickle checkpoints "
            "(.pth / .pt / .ckpt) are no longer accepted for security reasons."
        )

    s = video.to(device=dev, dtype=compute_dtype, copy=True)
    model = load_upscale_model(model_name, dev, precision)
    norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

    with torch.inference_mode():
        s = s.sub(norm_mean).div(norm_std)
        # tier (1-6) is the 2D model's sharpness control; the 3D model ignores it.
        out = model(s, scale=eff, target_size=(t, h_out, w_out),
                    tier=param.get("sharpness"))
        del s
        out = out.mul(norm_std).add(norm_mean)

    out = out.to(device="cpu", dtype=orig_dtype)
    if param.get("offload_model", True):
        unload_upscale_model(model_name, dev, precision)
    return out, h_out, w_out
