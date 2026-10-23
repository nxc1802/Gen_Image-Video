"""
Package for MiniMax H3 Video & Audio Generation
"""

from .engine import ComfyUIMiniMaxEngine, get_minimax_engine
from .workflow import build_minimax_h3_workflow
from .manifest import get_minimax_video_targets

__all__ = [
    "ComfyUIMiniMaxEngine",
    "get_minimax_engine",
    "build_minimax_h3_workflow",
    "get_minimax_video_targets",
]
