"""
workflows/video/tests/test_video_registry.py
Unit tests for VideoWorkflowRegistry unified get_workflow accessor and task aliases.
"""
import pytest
from workflows.video.registry import VideoWorkflowRegistry


def test_video_registry_does_not_normalize_request_aliases():
    registry = VideoWorkflowRegistry()
    with pytest.raises(ValueError, match="canonical"):
        registry.get_workflow("t2v")


def test_video_registry_get_workflow_unified():
    registry = VideoWorkflowRegistry()

    # Two-arg call: get_workflow(model, task)
    wf_wan_t2v = registry.get_workflow("wan-2.2", 'video.text_to_video')
    assert "9" in wf_wan_t2v
    assert wf_wan_t2v == registry.get_workflow_for_task("wan-2.2", 'video.text_to_video')

    wf_minimax_t2v = registry.get_workflow("minimax-h3", 'video.text_to_video')
    assert "9" in wf_minimax_t2v
    assert wf_minimax_t2v == registry.get_workflow_for_task("minimax-h3", 'video.text_to_video')

    wf_wan_i2v = registry.get_workflow("wan-2.2", 'video.image_to_video')
    assert wf_wan_i2v == registry.get_workflow_for_task("wan-2.2", 'video.image_to_video')

    wf_wan_fl2v = registry.get_workflow("wan-2.2", 'video.first_last_frame_to_video')
    assert wf_wan_fl2v == registry.get_workflow_for_task("wan-2.2", 'video.first_last_frame_to_video')

    wf_wan_ext = registry.get_workflow("wan-2.2", 'video.video_extension')
    assert wf_wan_ext == registry.get_workflow_for_task("wan-2.2", 'video.video_extension')

    # Single-arg call: get_workflow(task) defaults to wan-2.2
    wf_single_t2v = registry.get_workflow('video.text_to_video')
    assert wf_single_t2v == wf_wan_t2v

    wf_single_canonical = registry.get_workflow('video.text_to_video')
    assert wf_single_canonical == wf_wan_t2v


def test_video_registry_returns_deep_copy():
    registry = VideoWorkflowRegistry()
    wf1 = registry.get_workflow("wan-2.2", 'video.text_to_video')
    wf2 = registry.get_workflow("wan-2.2", 'video.text_to_video')
    assert wf1 == wf2
    assert wf1 is not wf2
    wf1["modified_node"] = {"test": True}
    assert "modified_node" not in wf2
