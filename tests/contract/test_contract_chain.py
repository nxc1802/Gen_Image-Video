"""
tests/contract/test_contract_chain.py
Contract tests verifying canonical identifier resolution and workflow registry parity.
"""
from pathlib import Path
import pytest

from model_registry.loader import ManifestLoader
from factory_contracts.tasks import (
    CANONICAL_MODEL_MAP,
    CANONICAL_TASK_MAP,
    CANONICAL_TO_CAPABILITY_MAP,
    MODEL_TO_FAMILY_MAP,
    get_model_aliases,
    get_task_aliases,
    model_to_family,
    normalize_model,
    normalize_task,
    task_to_capability_id,
)
from workflows.video.registry import VideoWorkflowRegistry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT_DIR / "manifests"


@pytest.fixture
def manifest_loader():
    return ManifestLoader(MANIFESTS_DIR)


@pytest.mark.tier1
class TestCanonicalIdentifierParity:
    """Contract tests verifying bidirectional resolution between capability IDs, models, and tasks."""

    def test_all_capability_manifests_mapped_in_canonical_task_map(self, manifest_loader):
        """Every capability declared in manifests/capabilities/ must exist in CANONICAL_TASK_MAP."""
        caps = manifest_loader.load_capabilities()
        assert len(caps) > 0, "No capability manifests found"

        for cap_id, cap in caps.items():
            assert cap_id in CANONICAL_TASK_MAP, f"Capability '{cap_id}' missing from CANONICAL_TASK_MAP"
            canonical_task = normalize_task(cap_id)
            assert canonical_task != "", f"Canonical task for '{cap_id}' must not be empty"
            # Bidirectional resolution
            primary_cap = task_to_capability_id(canonical_task)
            assert primary_cap is not None, f"Canonical task '{canonical_task}' has no primary capability ID mapping"

    def test_all_model_manifests_mapped_in_canonical_model_map(self, manifest_loader):
        """Every model manifest in manifests/models/ must resolve canonically."""
        models = manifest_loader.load_models()
        assert len(models) > 0, "No model manifests found"

        for model_id, model_manifest in models.items():
            normalized = normalize_model(model_id)
            assert normalized == model_id, f"Model ID '{model_id}' should be its own canonical name"
            family = model_to_family(model_id)
            assert family is not None, f"Model '{model_id}' must map to a model family"

            # A family is a grouping label, not an implicit checkpoint alias.
            assert family == model_manifest.metadata.family

    def test_model_and_task_aliases_bidirectional(self):
        """Verify alias lookup helpers return valid non-empty lists."""
        image_aliases = get_task_aliases("text_to_image")
        assert "image.t2i" in image_aliases
        assert "t2i" in image_aliases
        assert "text_to_image" in image_aliases

        wan_aliases = get_model_aliases("wan-2.2")
        assert "wan" in wan_aliases
        assert "wan-2.2" in wan_aliases

    def test_cross_model_workflow_fallback_forbidden(self):
        """Cross-model workflow fallback (e.g. attempting video_with_audio on Wan2.2) is strictly forbidden."""
        video_registry = VideoWorkflowRegistry()
        with pytest.raises(KeyError, match="not implemented for model 'wan-2.2'"):
            video_registry.get_workflow_for_task(model="wan-2.2", task='video.video_with_audio')
