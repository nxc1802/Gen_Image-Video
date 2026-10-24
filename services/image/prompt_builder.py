"""
services/image/prompt_builder.py
Translates FactoryJob requests into compiled ComfyUI prompt graphs for Image Subsystem.
"""
from typing import Any, Dict, Optional
from workflows.image.registry import ImageWorkflowRegistry
from workflows.image.parameter_mappings import ImageParameterMapper


class ImagePromptBuilder:
    """Builds and parameterizes ComfyUI prompt graphs for image generation tasks."""

    def __init__(self, registry: Optional[ImageWorkflowRegistry] = None):
        self.registry = registry or ImageWorkflowRegistry()

    def build_prompt(
        self,
        task: str,
        inputs: Dict[str, Any],
        staged_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a complete, validated ComfyUI prompt dictionary for the given task and inputs.
        """
        workflow = self.registry.get_workflow(task)
        parameterized = ImageParameterMapper.apply(
            workflow=workflow,
            task=task,
            inputs=inputs,
            staged_files=staged_files,
        )
        return parameterized
