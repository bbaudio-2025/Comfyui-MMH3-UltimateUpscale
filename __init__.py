"""Comfyui-MMH3-UltimateUpscale - one node for the full latent re-enhancement loop."""
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    MMH3UltimateUpscale,
    MMH3LatentUpscaleWithModelParams,
    MMH3LatentUpscaleParams,
    MMH3TemporalSplitParams,
    MMH3SpatialSplitParams,
    LTX25UltimateUpscale,
    LTX25LatentUpscaleParams,
    LTX25TemporalSplitParams,
    LTX25SpatialSplitParams,
)

NODE_CLASS_MAPPINGS = {
    "MMH3UltimateUpscale": MMH3UltimateUpscale,
    "MMH3LatentUpscaleWithModelParams": MMH3LatentUpscaleWithModelParams,
    "MMH3LatentUpscaleParams": MMH3LatentUpscaleParams,
    "MMH3TemporalSplitParams": MMH3TemporalSplitParams,
    "MMH3SpatialSplitParams": MMH3SpatialSplitParams,
    "LTX25UltimateUpscale": LTX25UltimateUpscale,
    "LTX25LatentUpscaleParams": LTX25LatentUpscaleParams,
    "LTX25TemporalSplitParams": LTX25TemporalSplitParams,
    "LTX25SpatialSplitParams": LTX25SpatialSplitParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3UltimateUpscale": "MMH3 Ultimate Upscale",
    "MMH3LatentUpscaleWithModelParams": "MMH3 Latent Upscale with Model Params",
    "MMH3LatentUpscaleParams": "MMH3 Latent Upscale Params",
    "MMH3TemporalSplitParams": "MMH3 Temporal Split Params",
    "MMH3SpatialSplitParams": "MMH3 Spatial Split Params",
    "LTX25UltimateUpscale": "LTX25 Ultimate Upscale",
    "LTX25LatentUpscaleParams": "LTX25 Latent Upscale Params",
    "LTX25TemporalSplitParams": "LTX25 Temporal Split Params",
    "LTX25SpatialSplitParams": "LTX25 Spatial Split Params",
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
            LTX25UltimateUpscale,
            LTX25LatentUpscaleParams,
            LTX25TemporalSplitParams,
            LTX25SpatialSplitParams,
        ]


async def comfy_entrypoint() -> MMH3UltimateUpscaleExtension:
    return MMH3UltimateUpscaleExtension()
