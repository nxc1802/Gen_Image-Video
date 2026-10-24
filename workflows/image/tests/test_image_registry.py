"""
workflows/image/tests/test_image_registry.py
Unit tests for ImageWorkflowRegistry task aliases and deep-copy semantics.
"""
import pytest
from workflows.image.registry import ImageWorkflowRegistry, TASK_ALIASES


def test_image_registry_task_aliases():
    registry = ImageWorkflowRegistry()

    # Test all canonical aliases
    assert TASK_ALIASES["t2i"] == "text_to_image"
    assert TASK_ALIASES["i2i"] == "image_to_image"
    assert TASK_ALIASES["edit"] == "instruction_edit"
    assert TASK_ALIASES["reference"] == "reference_conditioning"

    # Verify alias resolution returns valid workflow graphs
    wf_t2i = registry.get_workflow("t2i")
    wf_canonical_t2i = registry.get_workflow("text_to_image")
    assert wf_t2i == wf_canonical_t2i
    assert "474" in wf_t2i

    wf_i2i = registry.get_workflow("i2i")
    wf_canonical_i2i = registry.get_workflow("image_to_image")
    assert wf_i2i == wf_canonical_i2i
    assert "474" in wf_i2i

    wf_edit = registry.get_workflow("edit")
    wf_canonical_edit = registry.get_workflow("instruction_edit")
    assert wf_edit == wf_canonical_edit

    wf_ref = registry.get_workflow("reference")
    wf_canonical_ref = registry.get_workflow("reference_conditioning")
    assert wf_ref == wf_canonical_ref


def test_image_registry_unknown_task_raises_key_error():
    registry = ImageWorkflowRegistry()
    with pytest.raises(KeyError, match="Unknown image task"):
        registry.get_workflow("unsupported_image_task")


def test_image_registry_returns_deep_copy():
    registry = ImageWorkflowRegistry()
    wf1 = registry.get_workflow("t2i")
    wf2 = registry.get_workflow("t2i")
    assert wf1 == wf2
    assert wf1 is not wf2
    wf1["new_key"] = "modified"
    assert "new_key" not in wf2
