"""
workflows/video/registry.py
Workflow registry for ComfyUI Video Subsystem.
Loads, validates, caches, and selects versioned workflow JSON definitions for Wan2.2 and MiniMax H3.
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

ALL_TEMPLATES = [
    "wan_t2v_5b.json",
    "wan_t2v_a14b.json",
    "wan_i2v_a14b.json",
    "wan_fl2v.json",
    "wan_extension.json",
    "minimax_t2va.json",
    "minimax_i2va.json",
    "minimax_fl2va.json",
    "minimax_ref2va.json",
]


TASK_ALIASES: Dict[str, str] = {
    "t2v": "text_to_video",
    "i2v": "image_to_video",
    "fl2v": "first_last_frame_to_video",
    "extension": "video_extension",
    "with_audio": "video_with_audio",
    "ref2va": "reference_conditioning",
}


class VideoWorkflowRegistry:
    """Registry managing versioned ComfyUI video workflow templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Workflow templates directory not found: {self.templates_dir}")

        for filename in ALL_TEMPLATES:
            filepath = self.templates_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required workflow template: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._validate_workflow(filename, data)
            template_key = filename.replace(".json", "")
            self._cache[template_key] = data

    @staticmethod
    def _validate_workflow(name: str, workflow: Dict[str, Any]) -> None:
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError(f"Workflow {name} must be a non-empty dict")
        for node_id, node in workflow.items():
            if "class_type" not in node or "inputs" not in node:
                raise ValueError(f"Invalid ComfyUI node {node_id} in {name}: missing class_type or inputs")

    def get_template(self, template_key: str) -> Dict[str, Any]:
        """Returns a deep copy of a specific workflow template by key."""
        if template_key not in self._cache:
            raise KeyError(f"Unknown template: {template_key}. Available: {list(self._cache.keys())}")
        return copy.deepcopy(self._cache[template_key])

    def get_workflow_for_task(self, model: str, task: str) -> Dict[str, Any]:
        """Selects and returns the appropriate workflow template based on model and task."""
        model_lower = (model or "").lower()
        is_minimax = "minimax" in model_lower or "h3" in model_lower
        canonical_task = TASK_ALIASES.get(task, task)

        if is_minimax:
            if canonical_task in ("text_to_video", "t2v"):
                return self.get_template("minimax_t2va")
            elif canonical_task in ("image_to_video", "i2v"):
                return self.get_template("minimax_i2va")
            elif canonical_task in ("first_last_frame_to_video", "fl2v"):
                return self.get_template("minimax_fl2va")
            elif canonical_task in ("video_with_audio", "with_audio"):
                return self.get_template("minimax_t2va")
            elif canonical_task in ("reference_conditioning", "ref2va"):
                return self.get_template("minimax_ref2va")
            else:
                return self.get_template("minimax_t2va")
        else:
            # Wan2.2 family
            if canonical_task in ("text_to_video", "t2v"):
                return self.get_template("wan_t2v_a14b")
            elif canonical_task in ("image_to_video", "i2v"):
                return self.get_template("wan_i2v_a14b")
            elif canonical_task in ("first_last_frame_to_video", "fl2v"):
                return self.get_template("wan_fl2v")
            elif canonical_task in ("video_extension", "extension"):
                return self.get_template("wan_extension")
            elif canonical_task in ("video_with_audio", "with_audio"):
                # Fallback to minimax for video_with_audio if requested
                return self.get_template("minimax_t2va")
            else:
                return self.get_template("wan_t2v_a14b")

    def get_workflow(self, model: str, task: Optional[str] = None) -> Dict[str, Any]:
        """Unified accessor for video workflow templates by model and task (or task only)."""
        if task is None:
            # Handle single argument invocation get_workflow(task)
            task = model
            model = "wan-2.2"
        return self.get_workflow_for_task(model=model, task=task)

    def list_templates(self) -> List[str]:
        return list(self._cache.keys())
