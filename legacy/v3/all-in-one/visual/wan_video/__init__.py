"""
Package for Wan2.2 Video Generation (TI2V-5B, T2V-A14B, I2V-A14B)
"""

from .engine import ComfyUIVideoEngine, get_wan_engine, get_wan22_engine
from .workflow import (
    build_wan22_ti2v_workflow,
    build_wan22_t2v_a14b_workflow,
    build_wan22_i2v_a14b_workflow,
)
from .manifest import get_wan_video_targets

__all__ = [
    "ComfyUIVideoEngine",
    "get_wan_engine",
    "get_wan22_engine",
    "build_wan22_ti2v_workflow",
    "build_wan22_t2v_a14b_workflow",
    "build_wan22_i2v_a14b_workflow",
    "get_wan_video_targets",
]
