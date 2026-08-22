# Comfyui-MMH3-UltimateUpscale

**Upscale long, high-resolution MiniMax H3 video on a VRAM-limited GPU — in a single node.**

This node re-samples (enhances / upscales) an already-denoised MiniMax H3 AV latent through the full auto pipeline **under tight VRAM constraints**: it processes the clip with **temporal chunking (so arbitrarily long videos fit in memory) + spatial tiling (so arbitrarily high resolutions fit in memory)**, keeping **peak VRAM bounded to a single tile**, while preserving the audio track intact.

MiniMax H3 generates video as a nested latent that bundles 24-channel video **and** 32-channel audio in one tensor. Standard ComfyUI upscale nodes do not understand this structure. `MMH3 Ultimate Upscale` wraps the entire `temporal split -> latent upscale -> spatial split -> per-tile sampling -> spatial stitch -> temporal stitch` loop into one node, so you can upscale a finished H3 clip the same way you would upscale a normal latent — without breaking audio and **without running out of VRAM even on small cards**.

---

## Features

- **Upscale long + high-res video on limited VRAM.** The core design goal: temporal chunking keeps arbitrarily long clips in memory, spatial tiling keeps arbitrarily high resolutions in memory, and only one tile is sampled at a time — so peak VRAM stays at a single tile regardless of video length or output resolution.
- **One node, full pipeline.** Temporal chunking (outer loop), optional latent upscale, spatial tiling (inner loop), per-tile diffusion sampling, then spatial + temporal stitching — all driven by a single `MMH3 Ultimate Upscale` node.
- **Temporal chunking for long videos.** A long clip is cut into overlapping time chunks; each chunk is processed independently and stitched back together.
- **Two upscale modes per chunk:**
  - **H3 3D model-based upscaler** (`MMH3 Latent Upscale with Model Params`) — uses the `minimax_h3_latent_upscaler_3d_*.safetensors` checkpoints from the `latent_upscale_models` folder.
  - **Model-free interpolation** (`MMH3 Latent Upscale Params`) — resizes the video latent spatially (nearest / bilinear / area / bicubic) with no extra model, audio untouched. Mirrors ComfyUI's *Upscale Latent* but keeps the nested AV structure.
- **Spatial tiling for bounded VRAM.** Each chunk is split into tiles and only one tile is sampled at a time, so peak VRAM stays at one tile instead of the whole frame.
- **Audio preserved.** The audio portion of the latent is carried through unchanged on every chunk and stitch — it is never re-sampled.
- **Optional stages.** `latent_upscale_param`, `temporal_split_param`, and `spatial_split_param` are all optional. Leave any of them unconnected to skip that stage (no upscale / single chunk / whole-chunk sampling).
- **VRAM-friendly model management.** The 3D upscaler is offloaded back to CPU after each use, and the diffusion model is unloaded while the upscaler runs, so H3 + upscaler are never resident at the same time (the next sample reloads H3 automatically).
- **Per-piece conditioning.** Conditioning is re-anchored in time per chunk and spatially cropped per tile; keyframe video latents are resized to the (possibly upscaled) chunk grid.

---

## Advantages

### Temporal consistency & smooth time transitions
- **Frame-0 anchor.** At the start of each chunk (except the first), the chunk's frame-0 keyframe is replaced by the previous chunk's re-sampled boundary frame (`anchor_conditioning`, controlled by `anchor_strength`, default `0.999` — mirroring the *Anchor MiniMax H3 Latent* node). This removes detail mismatch at the chunk seam.
- **Cross-fade stitching.** Overlapping chunks are blended with a linear cross-fade (`temporal_append` / `_crossfade`) over the overlap region, so transitions between chunks are smooth rather than hard-cut.

### Pixel-space consistency & smooth spatial transitions
- **Frozen overlap mask.** Each tile is sampled at its true extent, but the overlap strips it shares with already-stitched neighbors are pre-filled from the accumulated result and locked with a `noise_mask` (`spatial_fade_mask`). The re-sample is therefore only allowed to change the *free* interior; the shared seam content is preserved exactly.
- **Masked write-back.** After sampling, the frozen seam region is written back with `torch.where(band, stitched, tile)`, guaranteeing the already-consistent seam is never overwritten.
- **Configurable seam blending.** The overlap band between tiles is blended with `overlap_blend` (`linear` / `smoothstep` / `overwrite` / `midpoint`) under `overlap_mode` (`earlier` wins / `later` wins), giving full control over how adjacent tiles transition into each other — smooth, not blocky.

### Other
- **Peak VRAM bounded to a single tile** thanks to spatial tiling + model offloading.
- **Audio never re-sampled** — no audio artifacts, no extra cost.
- **No forced model download** — pick the model-based 3D upscaler *or* the model-free interpolation path.

---

## Nodes

| Node | Role |
|------|------|
| **MMH3 Ultimate Upscale** | Main node. Runs the whole loop. Inputs: `latent`, `conditioning`, `model`, `noise`, `sampler`, `sigmas`, optional `negative` + `cfg`, and the three optional param inputs. |
| **MMH3 Temporal Split Params** | `chunk_length` (px frames, multiple of 17), `temporal_overlap` (multiple of 17), `anchor_strength`. |
| **MMH3 Spatial Split Params** | `tile_width` / `tile_height` (px, multiple of 32), `spatial_w_overlap` / `spatial_h_overlap` (px, multiple of 32), `fade_width` / `fade_height` (seam mask fade), `min_tile_size`, `overlap_mode`, `overlap_blend`. |
| **MMH3 Latent Upscale with Model Params** | H3 3D model upscaler: `model_name`, `width`, `height` (snapped to a multiple of 32), `device`, `precision`. |
| **MMH3 Latent Upscale Params** | Model-free interpolation: `method`, `width`, `height` (snapped to a multiple of 32). |

### Typical workflow
1. Generate an H3 AV latent with MiniMax H3 (video + audio in one latent).
2. (Optional) `MMH3 Temporal Split Params` → connect to `temporal_split_param`.
3. (Optional) `MMH3 Latent Upscale with Model Params` **or** `MMH3 Latent Upscale Params` → connect to `latent_upscale_param`.
4. (Optional) `MMH3 Spatial Split Params` → connect to `spatial_split_param`.
5. Feed `latent`, `conditioning`, `model`, `noise`, `sampler`, `sigmas` into `MMH3 Ultimate Upscale`.
6. Decode the output latent with the H3 VAE.

> The width/height you set for upscaling must match the **conditioning's generation size** (the size the video was conditioned at, after upscale).

---

## Reference Projects

This node is built on top of two existing community projects:

- **Latent split (temporal / spatial / anchor / append mechanics):**  
  https://github.com/bbaudio-2025/Comfyui-MiniMax-H3-LatentSplit
- **Latent model-based upscaling (H3 3D upscaler checkpoints & inference):**  
  https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

The H3 3D upscaler network code and normalization statistics are adapted from the second project; the temporal/spatial split, anchor and append logic follow the first.

---

## Extra

This project was vibe-coded by AI, If you run into any problems, it's best to search with AI.😂
