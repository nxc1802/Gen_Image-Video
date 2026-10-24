"""
workflows/video/tests/test_video_registry.py
Unit tests for VideoWorkflowRegistry unified get_workflow accessor and task aliases.
"""
import pytest
from workflows.video.registry import VideoWorkflowRegistry, TASK_ALIASES


def test_video_registry_task_aliases():
    assert TASK_ALIASES["t2v"] == "text_to_video"
    assert TASK_ALIASES["i2v"] == "image_to_video"
    assert TASK_ALIASES["fl2v"] == "first_last_frame_to_video"
    assert TASK_ALIASES["extension"] == "video_extension"
    assert TASK_ALIASES["with_audio"] == "video_with_audio"
    assert TASK_ALIASES["ref2va"] == "reference_conditioning"


def test_video_registry_get_workflow_unified():
    registry = VideoWorkflowRegistry()

    # Two-arg call: get_workflow(model, task)
    wf_wan_t2v = registry.get_workflow("wan-2.2", "t2v")
    assert "9" in wf_wan_t2v
    assert wf_wan_t2v == registry.get_workflow_for_task("wan-2.2", "text_to_video")

    wf_minimax_t2v = registry.get_workflow("minimax-h3", "t2v")
    assert "9" in wf_minimax_t2v
    assert wf_minimax_t2v == registry.get_workflow_for_task("minimax-h3", "text_to_video")

    wf_wan_i2v = registry.get_workflow("wan-2.2", "i2v")
    assert wf_wan_i2v == registry.get_workflow_for_task("wan-2.2", "image_to_video")

    wf_wan_fl2v = registry.get_workflow("wan-2.2", "fl2v")
    assert wf_wan_fl2v == registry.get_workflow_for_task("wan-2.2", "first_last_frame_to_video")

    wf_wan_ext = registry.get_workflow("wan-2.2", "extension")
    assert wf_wan_ext == registry.get_workflow_for_task("wan-2.2", "video_extension")

    # Single-arg call: get_workflow(task) defaults to wan-2.2
    wf_single_t2v = registry.get_workflow("t2v")
    assert wf_single_t2v == wf_wan_t2v

    wf_single_canonical = registry.get_workflow("text_to_video")
    assert wf_single_canonical == wf_wan_t2v


def test_video_registry_returns_deep_copy():
    registry = VideoWorkflowRegistry()
    wf1 = registry.get_workflow("wan-2.2", "t2v")
    wf2 = registry.get_workflow("wan-2.2", "t2v")
    assert wf1 == wf2
    assert wf1 is not wf2
    wf1["modified_node"] = {"test": True}
    assert "modified_node" not in wf2
