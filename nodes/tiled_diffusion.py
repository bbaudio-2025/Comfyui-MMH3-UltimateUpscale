"""Experimental per-step spatial Tiled Diffusion for MiniMax H3.

Ported from shiimizu/ComfyUI-TiledDiffusion (MultiDiffusion / Mixture of
Diffusers). Instead of sampling each tile to completion and stitching the
latents afterwards (the 'spatial split' path of MMH3 Ultimate Upscale), this
node patches the model with a model_function_wrapper: every denoising step
slices the packed AV latent into spatial tiles, runs the DiT once per tile,
and blends the tile predictions into a shared full-frame buffer with uniform
or gaussian weights. All tiles advance through the same sigmas on the same
noise field, so no freeze/fade overlap masks (and no intermediate denoise-mask
values) are involved.

v1.1: tile windows carry GLOBAL frame positions. v1 gave mutually unrelated
tile contents because every tile's PackedLayout restarted its RoPE grid at
the tile origin, so the DiT treated each tile as a complete standalone frame.
_WindowLayout now derives each tile's layout from the full-frame layout,
keeping the window's true (h, w) coordinates and the full-frame audio
anchors, so a tile's prediction matches what the full-frame model would
predict inside that window - overlap blending then averages consistent
predictions instead of unrelated ones.

Usage: model -> MMH3 Tiled Diffusion -> MMH3 Ultimate Upscale, with the
Ultimate Upscale node's spatial_split_param LEFT UNCONNECTED so each temporal
chunk is sampled whole through the patched model. Temporal split and latent
upscale keep working unchanged.

Guidance:
- keep tile_overlap at a meaningful fraction of the tile size (>= ~25%), or
  the tiles decouple again; e.g. 128px on 1024px tiles is too weak
- lower denoise keeps the source latent anchoring the content

v1 limitations (experimental):
- no controlnet / fun control, no denoise mask, no inpaint chain
- tiles run sequentially inside each step (H3 requires batch size 1), so the
  node trades nothing for speed - its goal is seam quality and tile sync
- temporal tiling is not implemented; long videos still go through the
  Ultimate Upscale node's temporal split
"""

import torch
import comfy.utils
import comfy.ldm.common_dit
from comfy_api.latest import io
from comfy.ldm.minimax.model import PackedLayout


def _ceil2(x):
    """Round up to the next even number (the H3 2x2 spatial patch grid)."""
    return (x + 1) // 2 * 2


def _tile_starts(total, tile, overlap):
    """Start offsets covering [0, total) with `tile`-unit tiles overlapping by
    `overlap`. Tile sizes and overlaps are multiples of 2 latent units (the H3
    2x2 spatial patch grid, i.e. 32 px) so every boundary stays patch-aligned."""
    if tile >= total:
        return [0]
    stride = tile - overlap
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


class _WindowLayout:
    """PackedLayout-compatible layout whose video / keyframe rows keep the
    GLOBAL frame coordinates of the tile window (instead of restarting the
    RoPE grid at the tile origin), and whose audio anchors come from the full
    frame so every tile positions the audio stream identically."""

    def __init__(self, full, keyframes, T, Hp, Wp, h0, w0, h1, w1, audio_t, text_len):
        # padded window bounds on the full frame's patch grid
        hp1, wp1 = h0 + _ceil2(h1 - h0), w0 + _ceil2(w1 - w0)
        rows = torch.arange(h0 // 2, hp1 // 2, dtype=torch.long)[:, None] * (Wp // 2) \
            + torch.arange(w0 // 2, wp1 // 2, dtype=torch.long)[None, :]
        idx = rows.reshape(-1)
        frame_rows = (Hp // 2) * (Wp // 2)
        # temporal token count per keyframe that packs a video latent, in
        # PackedLayout's segment order (one 'cond' segment per such keyframe)
        kf_vts = [kf["latent"].shape[2] for kf in (keyframes or ()) if kf.get("latent") is not None]

        pos, img_pos, img_upd, aud_pos, aud_upd, segments = [], [], [], [], [], []
        kf_i = 0
        row = 0
        for a, b, kind in full.segments:
            p = full.position_ids[a:b]
            if kind == "video":
                p = p.reshape(T, frame_rows, 3)[:, idx, :].reshape(-1, 3)
            elif kind == "cond":
                vt = kf_vts[kf_i]
                kf_i += 1
                p = p.reshape(vt, frame_rows, 3)[:, idx, :].reshape(-1, 3)
            n = p.shape[0]
            pos.append(p)
            segments.append((row, row + n, kind))
            if kind in ("cond", "ref_img", "video"):
                img_pos.append(torch.arange(row, row + n))
                img_upd.append(torch.full((n,), kind == "video", dtype=torch.bool))
            elif kind in ("cond_audio", "ref_audio", "audio"):
                aud_pos.append(torch.arange(row, row + n))
                aud_upd.append(torch.full((n,), kind == "audio", dtype=torch.bool))
            row += n

        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_upd)
        self.audio_pos = torch.cat(aud_pos)
        self.audio_update = torch.cat(aud_upd)
        self.segments = segments
        self.seq_len = row
        self.signature = (text_len, T, hp1 - h0, wp1 - w0, audio_t)


class _H3TiledDiffusionImpl:
    """model_function_wrapper installed via set_model_unet_function_wrapper.

    Receives the flat packed AV latent ([B, 1, video_elems + audio_elems], see
    comfy.utils.pack_latents) plus the cond dict carrying latent_shapes and the
    minimax payload (keyframes / refs / prebuilt packed layout)."""

    def __init__(self, method, tile_h, tile_w, overlap):
        self.method = method
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.overlap = overlap
        # (payload, video shape) -> (tiles, weights, per-tile payloads); rebuilt
        # when the conditioning payload or the chunk shape changes
        self._cache = None

    def _build(self, payload, video, audio, c):
        _, _, T, H, W = video.shape
        rows = _tile_starts(H, self.tile_h, self.overlap)
        cols = _tile_starts(W, self.tile_w, self.overlap)
        text = c.get("c_crossattn")
        text_len = text.shape[1] if text is not None else 0
        audio_t = audio.shape[-1]
        keyframes = payload.get("keyframes")
        refs = payload.get("refs")

        # full-frame layout as the source of global window positions; the
        # payload usually carries the one model_base prebuilt per chunk
        Hp, Wp = _ceil2(H), _ceil2(W)
        full = payload.get("layout")
        if full is None or full.signature != (text_len, T, Hp, Wp, audio_t):
            full = PackedLayout(text_len, T, Hp, Wp, audio_t,
                                keyframes=keyframes, refs=refs)

        tiles, weights, payloads = [], [], []
        for h0 in rows:
            for w0 in cols:
                h1, w1 = min(h0 + self.tile_h, H), min(w0 + self.tile_w, W)
                th, tw = h1 - h0, w1 - w0
                tiles.append((h0, w0, h1, w1))
                if self.method == "gaussian":
                    sigma = max(min(th, tw) / 8.0, 1e-6)
                    yy = torch.arange(th, dtype=torch.float32) - (th - 1) / 2.0
                    xx = torch.arange(tw, dtype=torch.float32) - (tw - 1) / 2.0
                    w = torch.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma * sigma))
                else:
                    w = torch.ones(th, tw, dtype=torch.float32)
                weights.append(w.to(device=video.device, dtype=video.dtype))

                p = dict(payload)
                if keyframes:
                    # keyframe latents share the target spatial grid, so crop
                    # (or resize-then-crop) them to the tile exactly like the
                    # spatial split path's crop_keyframes_to_tile does
                    kfs, cond_v = [], []
                    for kf in keyframes:
                        nkf = dict(kf)
                        lt = nkf.get("latent")
                        if lt is not None:
                            if lt.shape[3] != H or lt.shape[4] != W:
                                Bk, Ck, Tk, Hk, Wk = lt.shape
                                lt = torch.nn.functional.interpolate(
                                    lt.reshape(Bk * Tk, Ck, Hk, Wk).float(),
                                    size=(H, W), mode="bilinear", align_corners=False,
                                ).reshape(Bk, Ck, Tk, H, W).to(lt.dtype)
                            crop = lt[:, :, :, h0:h1, w0:w1].contiguous()
                            nkf["latent"] = comfy.ldm.common_dit.pad_to_patch_size(crop, (1, 2, 2))
                            cond_v.append(nkf["latent"])
                        kfs.append(nkf)
                    p["keyframes"] = kfs
                    # refs keep their own resolution and grid - only keyframes
                    # are cropped; the flattened cond list must mirror that
                    p["cond_video_latents"] = cond_v + [
                        r["latent"] for r in (refs or ()) if "latent" in r]
                # window layout with GLOBAL positions so the tile predicts
                # what the full-frame model would inside this window
                p["layout"] = _WindowLayout(full, keyframes, T, Hp, Wp,
                                            h0, w0, h1, w1, audio_t, text_len)
                payloads.append(p)
        print(f"[MMH3-TiledDiffusion] {len(tiles)} tiles ({len(rows)}x{len(cols)}) "
              f"on {H}x{W} latent, overlap {self.overlap * 16}px, method={self.method}, "
              f"global window positions")
        return tiles, weights, payloads

    def __call__(self, model_function, args):
        x = args["input"]
        t = args["timestep"]
        c = args["c"]

        payload = c.get("minimax_payload")
        if payload is None:
            raise ValueError("MMH3 Tiled Diffusion only works with the MiniMax H3 model.")
        for k in ("control", "denoise_mask", "audio_denoise_mask"):
            if c.get(k) is not None:
                raise ValueError(
                    f"MMH3 Tiled Diffusion (experimental) does not support '{k}'. "
                    "Disconnect the fun controlnet / inpaint chain, or use the "
                    "spatial split path of MMH3 Ultimate Upscale instead.")

        streams = comfy.utils.unpack_latents(x, c["latent_shapes"])
        video, audio = streams[0], streams[1] if len(streams) > 1 else None
        _, _, T, H, W = video.shape

        entry = self._cache
        if entry is None or entry[0] is not payload or entry[1] != (T, H, W):
            entry = (payload, (T, H, W)) + self._build(payload, video, audio, c)
            self._cache = entry
        _, _, tiles, weights, payloads = entry

        buf = torch.zeros_like(video)
        wsum = torch.zeros((1, 1, 1, H, W), device=video.device, dtype=video.dtype)
        audio_acc = None
        for (h0, w0, h1, w1), w, p in zip(tiles, weights, payloads):
            v_t = video[:, :, :, h0:h1, w0:w1]
            c_t = dict(c)
            c_t["minimax_payload"] = p
            if audio is not None:
                x_t, shapes_t = comfy.utils.pack_latents([v_t, audio])
                c_t["latent_shapes"] = shapes_t
                v_out, a_out = comfy.utils.unpack_latents(
                    model_function(x_t, t, **c_t), shapes_t)
            else:
                v_out = model_function(v_t, t, **c_t)
                a_out = None
            buf[:, :, :, h0:h1, w0:w1] += v_out * w[None, None, None]
            wsum[:, :, :, h0:h1, w0:w1] += w[None, None, None]
            if a_out is not None:
                audio_acc = a_out if audio_acc is None else audio_acc + a_out

        out_v = buf / wsum.clamp(min=1e-8)
        if audio is None:
            return out_v
        # every tile saw the full audio stream; average its predictions
        out, _ = comfy.utils.pack_latents([out_v, audio_acc / len(tiles)])
        return out


class MMH3TiledDiffusion(io.ComfyNode):
    """Patch an H3 model for per-step spatial Tiled Diffusion sampling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TiledDiffusion",
            display_name="MMH3 Tiled Diffusion (Experimental)",
            category="model/latent/minimax",
            description=(
                "Experimental per-step spatial Tiled Diffusion for MiniMax H3 "
                "(MultiDiffusion / Mixture of Diffusers style, ported from "
                "shiimizu/ComfyUI-TiledDiffusion). Patches the model so every "
                "denoising step slices the latent into spatial tiles, runs the "
                "DiT per tile, and blends the predictions with uniform or "
                "gaussian weights - all tiles share the same sigmas and noise "
                "field, so no overlap masks are needed. Tiles carry global "
                "frame positions so their predictions stay consistent with the "
                "full frame. Connect between the H3 model and 'MMH3 Ultimate "
                "Upscale' and leave that node's spatial_split_param "
                "unconnected. Use a tile_overlap of at least ~25% of the tile "
                "size, or the tiles decouple. Does not support fun controlnet, "
                "inpaint or denoise masks, and does not save VRAM/time (H3 "
                "runs batch 1); it is an experiment in seam quality."
            ),
            search_aliases=["tiled diffusion", "multidiffusion", "mixture of diffusers", "h3 tiling"],
            inputs=[
                io.Model.Input("model",
                               tooltip="The MiniMax H3 diffusion model to patch."),
                io.Combo.Input("method", options=["gaussian", "uniform"], default="gaussian",
                               tooltip="Overlap blending weight per tile: gaussian (Mixture of Diffusers, default - tiles dominate their center) or uniform (MultiDiffusion - plain averaging in overlaps)."),
                io.Int.Input("tile_width", default=1024, min=64, max=8192, step=32,
                             tooltip="Tile width in PIXELS, snapped to a multiple of 32 (the H3 2x2 latent patch grid)."),
                io.Int.Input("tile_height", default=1024, min=64, max=8192, step=32,
                             tooltip="Tile height in PIXELS, snapped to a multiple of 32 (the H3 2x2 latent patch grid)."),
                io.Int.Input("tile_overlap", default=256, min=0, max=8192, step=32,
                             tooltip="Overlap between neighbouring tiles in PIXELS, snapped to a multiple of 32. Keep at least ~25% of the tile size or the tiles decouple into unrelated content. 0 gives hard tile edges."),
            ],
            outputs=[
                io.Model.Output("model",
                                tooltip="The patched model; feed it into 'MMH3 Ultimate Upscale' (spatial_split_param unconnected) or any H3 sampler."),
            ],
        )

    @classmethod
    def execute(cls, model, method, tile_width, tile_height, tile_overlap) -> io.NodeOutput:
        # pixel -> latent units, snapped to the 2-token (32 px) patch grid
        tw = int(round(tile_width / 32.0)) * 2
        th = int(round(tile_height / 32.0)) * 2
        ov = int(round(tile_overlap / 32.0)) * 2
        if ov >= min(tw, th):
            raise ValueError(
                f"tile_overlap ({ov * 16}px) must be smaller than the tile size "
                f"({th * 16}x{tw * 16}px) so tiles can advance.")
        impl = _H3TiledDiffusionImpl(method, th, tw, ov)
        patched = model.clone()
        patched.set_model_unet_function_wrapper(impl)
        return io.NodeOutput(patched)
