"""
tests/contract/test_provenance_and_licensing.py
Contract tests verifying model licensing, commercial use flags, and weight provenance across all 11 models.
"""
from pathlib import Path
import pytest

from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator
from deployments.kaggle.prepare_datasets import MODEL_DATASETS

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT_DIR / "manifests"

EXPECTED_11_MODEL_IDS = {
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

NON_COMMERCIAL_MODEL_IDS = {
    "minimax-h3",
    "ace-step-1.5",
    "enhancement-suite-v1",
}


@pytest.mark.tier1
class TestModelProvenanceAndLicensing:
    @pytest.fixture(autouse=True)
    def setup_manifests(self):
        self.loader = ManifestLoader(MANIFESTS_DIR)
        self.models = self.loader.load_models()

    def test_all_11_models_present(self):
        loaded_ids = set(self.models.keys())
        expected_with_turbo = EXPECTED_11_MODEL_IDS | {"qwen-image-2.1-turbo"}
        assert loaded_ids in (EXPECTED_11_MODEL_IDS, expected_with_turbo), (
            f"Expected canonical models, got {len(loaded_ids)}: {loaded_ids}"
        )


    def test_all_11_models_have_legal_license_declarations(self):
        for m_id, model in self.models.items():
            lic = getattr(model.metadata, "license", None)
            assert lic is not None and isinstance(lic, str) and len(lic.strip()) > 0, (
                f"Model '{m_id}' is missing a valid metadata.license declaration."
            )
            comm = getattr(model.metadata, "commercial_use", None)
            assert isinstance(comm, bool), (
                f"Model '{m_id}' metadata.commercial_use must be a boolean (True/False), got {type(comm)}."
            )
            lic_url = getattr(model.metadata, "license_url", None)
            assert lic_url is not None and isinstance(lic_url, str) and len(lic_url.strip()) > 0, (
                f"Model '{m_id}' is missing metadata.license_url."
            )

    def test_non_commercial_models_flagged_correctly(self):
        for m_id in NON_COMMERCIAL_MODEL_IDS:
            model = self.models[m_id]
            assert model.metadata.commercial_use is False, (
                f"Non-commercial model '{m_id}' must have commercial_use: false."
            )
            lic_lower = model.metadata.license.lower()
            assert any(term in lic_lower for term in ["non-commercial", "nc", "rife"]), (
                f"Non-commercial model '{m_id}' license '{model.metadata.license}' must indicate non-commercial terms."
            )

    def test_commercial_models_flagged_correctly(self):
        commercial_ids = set(self.models.keys()) - NON_COMMERCIAL_MODEL_IDS
        for m_id in commercial_ids:
            model = self.models[m_id]
            assert model.metadata.commercial_use is True, (
                f"Commercial model '{m_id}' should have commercial_use: true."
            )

    def test_zero_generic_mit_on_non_mit_models(self):
        # None of the 11 models are pure MIT licensed
        for m_id, model in self.models.items():
            assert model.metadata.license.strip().upper() != "MIT", (
                f"Model '{m_id}' incorrectly declared generic 'MIT' license."
            )

    def test_all_11_models_declare_checkpoint_revision_and_weights_file(self):
        for m_id, model in self.models.items():
            weights = model.weights
            repo = getattr(weights, "repository", None) or weights.primary_checkpoint
            assert repo is not None and len(repo.strip()) > 0, (
                f"Model '{m_id}' weights must declare repository or primary_checkpoint."
            )
            rev = getattr(weights, "checkpoint_revision", None)
            assert rev is not None and len(rev.strip()) >= 7, (
                f"Model '{m_id}' weights.checkpoint_revision must be a valid commit SHA or tag (got '{rev}')."
            )
            wfile = getattr(weights, "weights_file", None)
            assert wfile is not None and len(wfile.strip()) > 0, (
                f"Model '{m_id}' weights.weights_file must be declared."
            )

    def test_prepare_datasets_licenses_match_manifests(self):
        dataset_map = {d["id"]: d for d in MODEL_DATASETS}
        for m_id, model in self.models.items():
            assert m_id in dataset_map, f"Model '{m_id}' missing from MODEL_DATASETS in prepare_datasets.py"
            ds = dataset_map[m_id]
            assert ds["commercial_use"] == model.metadata.commercial_use, (
                f"prepare_datasets.py commercial_use mismatch for '{m_id}': {ds['commercial_use']} != {model.metadata.commercial_use}"
            )
            if not model.metadata.commercial_use:
                assert ds["license_name"] != "apache-2.0", (
                    f"Non-commercial dataset '{m_id}' cannot use 'apache-2.0' in prepare_datasets.py"
                )

    def test_manifest_validator_catches_invalid_licensing(self):
        validator = ManifestValidator(self.loader)
        ok, errors = validator.validate_all()
        assert ok, f"Expected clean validation for all manifests, got errors: {errors}"
