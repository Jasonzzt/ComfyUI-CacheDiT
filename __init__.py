"""
ComfyUI-CacheDiT
================

Production-ready ComfyUI integration for cache-dit (v1.2.0+).
Accelerate DiT model inference through inter-step residual caching.

Supported Models (2026):
- Qwen-Image Series (2511)
- LTX-2 (T2V, I2V)
- Z-Image / Z-Image-Turbo

Features:
- Adaptive/Static/Dynamic caching strategies
- Skip interval for video temporal consistency
- Noise injection to prevent static artifacts
- Automatic step detection and context refresh
- Rich summary statistics with ASCII dashboard

Repository: https://github.com/your-org/ComfyUI-CacheDiT
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "ComfyUI-CacheDiT Contributors"

from .nodes import NODE_CLASS_MAPPINGS as MAIN_NODE_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as MAIN_NODE_DISPLAY_NAME_MAPPINGS

from .nodes_ltx2 import NODE_CLASS_MAPPINGS as LTX2_NODE_CLASS_MAPPINGS
from .nodes_ltx2 import NODE_DISPLAY_NAME_MAPPINGS as LTX2_NODE_DISPLAY_NAME_MAPPINGS

from .nodes_wan import NODE_CLASS_MAPPINGS as WAN_NODE_CLASS_MAPPINGS
from .nodes_wan import NODE_DISPLAY_NAME_MAPPINGS as WAN_NODE_DISPLAY_NAME_MAPPINGS

# Merge all nodes
NODE_CLASS_MAPPINGS = {
    **MAIN_NODE_CLASS_MAPPINGS,
    **LTX2_NODE_CLASS_MAPPINGS,
    **WAN_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MAIN_NODE_DISPLAY_NAME_MAPPINGS,
    **LTX2_NODE_DISPLAY_NAME_MAPPINGS,
    **WAN_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
