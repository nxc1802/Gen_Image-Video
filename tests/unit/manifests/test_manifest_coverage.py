"""Unit tests for Feature 29 (P2.1): Model Manifest Coverage and Explicit Turbo Variant.

Validates that:
1. Exactly 12 model manifests exist (11 standard foundation models + 1 explicit turbo variant manifest).
2. qwen-image-2.1-turbo is properly registered and conforms to all Pydantic & validation contracts.
3. All 12 manifests pass ManifestValidator and ModelManifest schema validation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from factory_contracts.manifests.model import (
    AllocationPolicy,
    ModelManifest,
    ModelStatus,
)
from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = WORKSPACE_ROOT / "manifests"

EXPECTED_11_STANDARD_MODELS = {
    "ace-step-1.5",
    "enhancement-suite-v1",
    "minimax-h3",
    "qwen-decision",
    "qwen-embedding",
    "qwen-image-2.1",
    "qwen-speech",
    "qwen-text",
    "qwen-vlm",
    "wan-2.2",
    "yue2",
}

EXPECTED_12_MODELS = EXPECTED_11_STANDARD_MODELS | {"qwen-image-2.1-turbo"}


@pytest.fixture
def manifest_loader() -> ManifestLoader:
    return ManifestLoader(MANIFESTS_DIR)


def test_manifest_coverage_count_and_ids(manifest_loader: ManifestLoader) -> None:
    """Assert exactly 12 models are discovered and match expected IDs."""
    models = manifest_loader.load_models()
    assert len(models) == 12, f"Expected exactly 12 models in manifests/, found {len(models)}"
    assert set(models.keys()) == EXPECTED_12_MODELS


def test_qwen_image_turbo_manifest_integrity(manifest_loader: ManifestLoader) -> None:
    """Verify detailed contract attributes of the explicit qwen-image-2.1-turbo manifest."""
    models = manifest_loader.load_models()
    assert "qwen-image-2.1-turbo" in models, "qwen-image-2.1-turbo manifest must be present"

    turbo = models["qwen-image-2.1-turbo"]
    assert isinstance(turbo, ModelManifest)

    # Metadata
    assert turbo.metadata.id == "qwen-image-2.1-turbo"
    assert turbo.metadata.family == "qwen-image"
    assert turbo.metadata.display_name == "Qwen-Image 2.1 Turbo"
    assert turbo.metadata.service_id == "image-service"
    assert turbo.metadata.license == "Tongyi Qianwen License"
    assert turbo.metadata.commercial_use is True

    # Hardware & VRAM budget
    assert turbo.hardware.preferred_gpu == "t4"
    assert turbo.hardware.preferred_gpu_index == 0
    assert turbo.hardware.min_vram_gb == 14.0
    assert turbo.hardware.allocation_policy == AllocationPolicy.SINGLE_GPU
    assert turbo.hardware.quantization_format == "q4_k_m"

    # Status & capabilities
    assert turbo.status == ModelStatus.PARTIAL
    assert turbo.coverage_percent == 20.0
    expected_caps = {
        "text_to_image": True,
        "inpaint": False,
        "outpaint": False,
        "controlnet": False,
        "instruction_edit": False,
    }
    for cap, expected_val in expected_caps.items():
        assert turbo.capabilities.get(cap) == expected_val, f"Capability {cap} expected {expected_val}"

    # Weights provenance
    assert turbo.weights.weights_file == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    assert turbo.weights.repository == "unsloth/Qwen-Image-2.1-Turbo-GGUF"
    assert turbo.weights.checkpoint_revision == "d1245e8964af54497e25e83ec9bb892bf32e5190"


def test_all_12_manifests_pass_validator(manifest_loader: ManifestLoader) -> None:
    """Assert all 12 models pass ManifestValidator with zero errors."""
    validator = ManifestValidator(manifest_loader)
    ok, errors = validator.validate_all()
    assert ok, f"ManifestValidator reported validation errors: {errors}"
    assert len(errors) == 0
