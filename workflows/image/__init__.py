"""
workflows/image
ComfyUI Image Subsystem Workflows
"""
from workflows.image.registry import ImageWorkflowRegistry
from workflows.image.parameter_mappings import ImageParameterMapper

__all__ = ["ImageWorkflowRegistry", "ImageParameterMapper"]
