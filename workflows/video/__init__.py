"""
workflows/video
ComfyUI Video Subsystem Workflows
"""
from workflows.video.registry import VideoWorkflowRegistry
from workflows.video.parameter_mappings import VideoParameterMapper

__all__ = ["VideoWorkflowRegistry", "VideoParameterMapper"]
