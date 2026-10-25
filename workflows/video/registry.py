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
    # Short aliases
    "t2v": "text_to_video",
    "i2v": "image_to_video",
    "fl2v": "first_last_frame_to_video",
    "extension": "video_extension",
    "with_audio": "video_with_audio",
    "ref2va": "reference_conditioning",
    # Capability IDs
    "video.t2v": "text_to_video",
    "video.i2v": "image_to_video",
    "video.fl2v": "first_last_frame_to_video",
    "video.extend": "video_extension",
    "video.extension": "video_extension",
    "video.audio": "video_with_audio",
    "video.with_audio": "video_with_audio",
    "video.reference": "reference_conditioning",
    "video.ref2va": "reference_conditioning",
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
        """Return only workflows genuinely implemented by the requested model family."""
        model_lower = (model or "").lower()
        canonical_task = TASK_ALIASES.get(task, task)

        if "minimax" in model_lower or "h3" in model_lower:
            mapping = {
                "text_to_video": "minimax_t2va",
                "image_to_video": "minimax_i2va",
                "first_last_frame_to_video": "minimax_fl2va",
                "video_with_audio": "minimax_t2va",
                "reference_conditioning": "minimax_ref2va",
            }
        elif "wan" in model_lower:
            mapping = {
                "text_to_video": "wan_t2v_a14b",
                "image_to_video": "wan_i2v_a14b",
                "first_last_frame_to_video": "wan_fl2v",
                "video_extension": "wan_extension",
            }
        else:
            raise KeyError(f"Unknown video model family: {model}")

        template = mapping.get(canonical_task)
        if template is None:
            raise KeyError(
                f"Task '{canonical_task}' is not implemented for model '{model}'. "
                "Cross-model workflow fallback is forbidden."
            )
        return self.get_template(template)

    def get_workflow(self, model: str, task: Optional[str] = None) -> Dict[str, Any]:
        """Unified accessor for video workflow templates by model and task (or task only)."""
        if task is None:
            # Handle single argument invocation get_workflow(task)
            task = model
            model = "wan-2.2"
        return self.get_workflow_for_task(model=model, task=task)

    def list_templates(self) -> List[str]:
        return list(self._cache.keys())
