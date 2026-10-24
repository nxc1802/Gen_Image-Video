"""
services/video/prompt_builder.py
Translates FactoryJob requests into compiled ComfyUI prompt graphs for Video Subsystem (Wan2.2 & MiniMax H3).
"""
from typing import Any, Dict, Optional
from workflows.video.registry import VideoWorkflowRegistry
from workflows.video.parameter_mappings import VideoParameterMapper


class VideoPromptBuilder:
    """Builds and parameterizes ComfyUI prompt graphs for Wan2.2 and MiniMax H3 tasks."""

    def __init__(self, registry: Optional[VideoWorkflowRegistry] = None):
        self.registry = registry or VideoWorkflowRegistry()

    def build_prompt(
        self,
        model: str,
        task: str,
        inputs: Dict[str, Any],
        staged_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        workflow = self.registry.get_workflow_for_task(model=model, task=task)
        parameterized = VideoParameterMapper.apply(
            workflow=workflow,
            model=model,
            task=task,
            inputs=inputs,
            staged_files=staged_files,
        )
        return parameterized
