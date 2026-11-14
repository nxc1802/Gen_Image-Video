"""Manifest provenance guards; no model loading or inference."""
import pytest
from pydantic import ValidationError

from factory_contracts.manifests.model import CheckpointFileSpec, ModelWeightsSpec


def identity(**updates):
    return {"repository": "owner/model", "revision": "a" * 40, "path": "weights/model.gguf",
            "sha256": "b" * 64, "size_bytes": 123, **updates}


@pytest.mark.parametrize("updates", [
    {"revision": "main"}, {"sha256": "unknown"}, {"path": "../other-model.gguf"},
    {"path": "weights/*.gguf"}, {"provider": "kaggle", "revision": "latest"},
])
def test_unpinned_or_ambiguous_file_identity_rejected(updates):
    with pytest.raises(ValidationError):
        CheckpointFileSpec.model_validate(identity(**updates))


def test_legacy_projection_is_derived_and_conflicting_pin_is_rejected():
    weights = ModelWeightsSpec.model_validate({"files": {"primary": identity()}})
    assert weights.repository == "owner/model"
    assert weights.checkpoint_revision == "a" * 40
    assert weights.sha256 == "b" * 64
    with pytest.raises(ValidationError, match="conflicts"):
        ModelWeightsSpec.model_validate({"files": {"primary": identity()}, "checkpoint_revision": "c" * 40})


def test_file_hash_does_not_invent_missing_primary_identity():
    with pytest.raises(ValidationError, match="primary"):
        ModelWeightsSpec.model_validate({"files": {"vae": identity()}})
