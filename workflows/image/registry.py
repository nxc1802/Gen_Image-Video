"""
workflows/image/registry.py
Workflow registry for ComfyUI Image Subsystem.
Loads, validates, caches, and provides access to versioned workflow JSON definitions.
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from factory_contracts.capabilities import require_canonical

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

TASK_TEMPLATE_MAP = {
    "image.text_to_image": "qwen_image_t2i.json",
    "image.image_to_image": "qwen_image_i2i.json",
    "image.inpaint": "qwen_image_inpaint.json",
    "image.outpaint": "qwen_image_outpaint.json",
    "image.instruction_edit": "qwen_image_edit.json",
    "image.reference_image": "qwen_image_reference.json",
    "image.control_pose": "qwen_image_controlnet.json",
    "image.control_depth": "qwen_image_controlnet.json",
    "image.multi_reference": "qwen_image_reference.json",
}



class ImageWorkflowRegistry:
    """Registry managing versioned ComfyUI image workflow templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Workflow templates directory not found: {self.templates_dir}")

        for task, filename in TASK_TEMPLATE_MAP.items():
            filepath = self.templates_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required workflow template: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._validate_workflow(task, data)
            self._cache[task] = data

    @staticmethod
    def _validate_workflow(task: str, workflow: Dict[str, Any]) -> None:
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError(f"Workflow for {task} must be a non-empty dict")
        for node_id, node in workflow.items():
            if "class_type" not in node or "inputs" not in node:
                raise ValueError(f"Invalid ComfyUI node {node_id} in {task}: missing class_type or inputs")

    def get_workflow(self, task: str) -> Dict[str, Any]:
        """Returns a deep copy of the raw workflow graph for the given task."""
        canonical_task = require_canonical(task, engine="image")
        if canonical_task not in self._cache:
            raise KeyError(f"Unknown image task: {task}. Supported tasks: {list(self._cache.keys())}")
        return copy.deepcopy(self._cache[canonical_task])

    def list_tasks(self) -> List[str]:
        return list(self._cache.keys())
