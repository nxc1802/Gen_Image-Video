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
    "qwen-asr",
    "qwen-tts",
    "qwen-reranker",
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
        assert expected_with_turbo <= loaded_ids, (
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
        for model in self.models.values():
            if "cc-by-nc" in model.metadata.license.lower():
                assert model.metadata.commercial_use is False

    def test_commercial_models_flagged_correctly(self):
        for model in self.models.values():
            if model.metadata.license.lower() in {"mit", "apache-2.0", "bsd-3-clause"}:
                assert model.metadata.commercial_use is True

    def test_official_mit_checkpoints_are_not_mislabeled(self):
        # ACE-Step's actual upstream MIT metadata supersedes the old invented
        # CC-BY-NC label; license facts must follow the selected checkpoint.
        assert self.models["ace-step-1.5"].metadata.license.lower() == "mit"
        assert self.models["yue2"].metadata.license.lower() == "cc-by-nc-4.0"

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
