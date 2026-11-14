"""
workflows/image/tests/test_image_registry.py
Unit tests for ImageWorkflowRegistry task aliases and deep-copy semantics.
"""
import pytest
from workflows.image.registry import ImageWorkflowRegistry


def test_image_registry_does_not_normalize_request_aliases():
    registry = ImageWorkflowRegistry()
    with pytest.raises(ValueError, match="canonical"):
        registry.get_workflow("t2i")


def test_image_registry_unknown_task_raises_key_error():
    registry = ImageWorkflowRegistry()
    with pytest.raises(ValueError, match="canonical"):
        registry.get_workflow("unsupported_image_task")


def test_image_registry_returns_deep_copy():
    registry = ImageWorkflowRegistry()
    wf1 = registry.get_workflow('image.text_to_image')
    wf2 = registry.get_workflow('image.text_to_image')
    assert wf1 == wf2
    assert wf1 is not wf2
    wf1["new_key"] = "modified"
    assert "new_key" not in wf2
