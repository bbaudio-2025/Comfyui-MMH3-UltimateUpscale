"""Comfyui-MMH3-UltimateUpscale - one node for the full latent re-enhancement loop."""
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    MMH3UltimateUpscale,
    MMH3LatentUpscaleWithModelParams,
    MMH3LatentUpscaleParams,
    MMH3TemporalSplitParams,
    MMH3SpatialSplitParams,
    MMH3FunControlnetParams,
    MMH3SpatialInpaintParams,
    LTX25UltimateUpscale,
    LTX25LatentUpscaleParams,
    LTX25TemporalSplitParams,
    LTX25SpatialSplitParams,
    LTX25ReferenceParams,
    LTX25ICLoRALoader,
)

NODE_CLASS_MAPPINGS = {
    "MMH3UltimateUpscale": MMH3UltimateUpscale,
    "MMH3LatentUpscaleWithModelParams": MMH3LatentUpscaleWithModelParams,
    "MMH3LatentUpscaleParams": MMH3LatentUpscaleParams,
    "MMH3TemporalSplitParams": MMH3TemporalSplitParams,
    "MMH3SpatialSplitParams": MMH3SpatialSplitParams,
    "MMH3FunControlnetParams": MMH3FunControlnetParams,
    "MMH3SpatialInpaintParams": MMH3SpatialInpaintParams,
    "LTX25UltimateUpscale": LTX25UltimateUpscale,
    "LTX25LatentUpscaleParams": LTX25LatentUpscaleParams,
    "LTX25TemporalSplitParams": LTX25TemporalSplitParams,
    "LTX25SpatialSplitParams": LTX25SpatialSplitParams,
    "LTX25ICLoRALoader": LTX25ICLoRALoader,
    "LTX25ReferenceParams": LTX25ReferenceParams,
}

# front-end JS: auto-show/hide tile size vs rows/cols inputs on the two
# Spatial Split Params nodes based on the tile_size_mode combo
WEB_DIRECTORY = "./web"

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3UltimateUpscale": "MMH3 Ultimate Upscale",
    "MMH3LatentUpscaleWithModelParams": "MMH3 Latent Upscale with Model Params",
    "MMH3LatentUpscaleParams": "MMH3 Latent Upscale Params",
    "MMH3TemporalSplitParams": "MMH3 Temporal Split Params",
    "MMH3SpatialSplitParams": "MMH3 Spatial Split Params",
    "MMH3FunControlnetParams": "MMH3 Fun Controlnet Params",
    "MMH3SpatialInpaintParams": "MMH3 Spatial Inpaint Params",
    "LTX25UltimateUpscale": "LTX25 Ultimate Upscale",
    "LTX25LatentUpscaleParams": "LTX25 Latent Upscale Params",
    "LTX25TemporalSplitParams": "LTX25 Temporal Split Params",
    "LTX25SpatialSplitParams": "LTX25 Spatial Split Params",
    "LTX25ICLoRALoader": "LTX25 IC-LoRA Loader (MSR)",
    "LTX25ReferenceParams": "LTX25 Reference Params",
}


class MMH3UltimateUpscaleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            MMH3UltimateUpscale,
            MMH3LatentUpscaleWithModelParams,
            MMH3LatentUpscaleParams,
            MMH3TemporalSplitParams,
            MMH3SpatialSplitParams,
            MMH3FunControlnetParams,
            MMH3SpatialInpaintParams,
            LTX25UltimateUpscale,
            LTX25LatentUpscaleParams,
            LTX25TemporalSplitParams,
            LTX25SpatialSplitParams,
            LTX25ReferenceParams,
            LTX25ICLoRALoader,
        ]


async def comfy_entrypoint() -> MMH3UltimateUpscaleExtension:
    return MMH3UltimateUpscaleExtension()
