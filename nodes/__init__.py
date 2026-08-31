"""Package holding the MMH3 Ultimate Upscale plugin's node implementations.

`nodes.py` is the MMH3 business logic (shared helpers + MMH3 nodes),
`h3_latent_upscaler.py` is the external H3 3D latent upscaler algorithm and
`ltx.py` is the LTX2.5 block. This module re-exports every node class so the
plugin's root `__init__.py` can keep importing them from `.nodes`.
"""

from .nodes import (
    MMH3UltimateUpscale,
    MMH3LatentUpscaleWithModelParams,
    MMH3LatentUpscaleParams,
    MMH3TemporalSplitParams,
    MMH3SpatialSplitParams,
    MMH3FunControlnetParams,
    MMH3SpatialInpaintParams,
)
from .ltx import (
    LTX25UltimateUpscale,
    LTX25LatentUpscaleParams,
    LTX25TemporalSplitParams,
    LTX25SpatialSplitParams,
    LTX25ICLoRALoader,
    LTX25ReferenceParams,
)
