"""
tests/unit/api/test_p0_job_admission_fixes.py
Targeted regression tests for Phase 0 fixes:
- FIX-06: Remove misleading aliases (e.g. 72B mapping to 7B checkpoint).
- FIX-07: Job admission checks implemented capability, backend readiness, variants, and hardware feasibility.
"""
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from unittest.mock import MagicMock

from factory_contracts.errors import FactoryError, InvalidRequestError
from factory_contracts.jobs.job_contract import JobSubmissionRequest
from factory_contracts.tasks import normalize_model
from factory.apps.control_api.src.api.routers.jobs import (
    MODEL_ALIASES,
    _validate_job_capability,
)
from model_registry.loader import ManifestLoader


def test_fix_06_no_misleading_72b_aliases():
    """FIX-06: 72B and 8B models must not be aliased to 7B or 3B models."""
    assert "qwen-2.5-72b" not in MODEL_ALIASES
    assert "qwen-2.5-72b-instruct" not in MODEL_ALIASES
    assert "qwen-vl-max" not in MODEL_ALIASES
    assert "qwen-decision-8b" not in MODEL_ALIASES

    # normalize_model should not silently alias 72b to qwen-text
    assert normalize_model("qwen-2.5-72b") != "qwen-text"
    assert normalize_model("qwen-2.5-vl-72b") != "qwen-vlm"


def test_fix_06_requested_identity_matches_resolved_weights():
    """FIX-06: Submitting qwen-2.5-72b fails early as unknown model instead of secretly using 7B weights."""
    loader = ManifestLoader()
    mock_request = MagicMock()
    mock_request.app.state.manifest_loader = loader

    req = JobSubmissionRequest(
        service="text",
        model="qwen-2.5-72b",
        task="chat",
        inputs={"messages": [{"role": "user", "content": "Hello"}]},
    )

    with pytest.raises(InvalidRequestError) as exc:
        _validate_job_capability(req, mock_request)
    assert "Unknown model 'qwen-2.5-72b'" in str(exc.value)


def test_fix_07_job_admission_checks_variants_and_capabilities():
    """FIX-07: Job admission validates variants, missing capabilities, and hardware feasibility."""
    loader = ManifestLoader()
    mock_request = MagicMock()
    mock_request.app.state.manifest_loader = loader

    # 1. Unknown variant rejected early
    req_bad_variant = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "test", "variant": "nonexistent_super_variant"},
    )
    with pytest.raises(InvalidRequestError) as exc_var:
        _validate_job_capability(req_bad_variant, mock_request)
    assert "Variant 'nonexistent_super_variant' is not supported" in str(exc_var.value)

    # 2. Supported variant accepted
    req_good_variant = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "test", "variant": "turbo"},
    )
    # Should not raise
    _validate_job_capability(req_good_variant, mock_request)

    # 3. Unsupported capability rejected early
    req_bad_task = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_video",  # image model cannot do video
        inputs={"prompt": "test"},
    )
    with pytest.raises(FactoryError) as exc_task:
        _validate_job_capability(req_bad_task, mock_request)
    assert exc_task.value.code == "CAPABILITY_NOT_SUPPORTED"
