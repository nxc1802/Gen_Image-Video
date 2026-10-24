"""
tests/contract/test_manifest_schemas.py
Contract tests verifying all YAML manifests in manifests/ pass full validation.
"""
from pathlib import Path
import pytest

from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT_DIR / "manifests"


@pytest.mark.tier1
class TestManifestContractIntegrity:
    def test_all_manifests_pass_validation(self):
        loader = ManifestLoader(MANIFESTS_DIR)
        validator = ManifestValidator(loader)
        ok, errors = validator.validate_all()
        assert ok, f"Manifest validation failed with errors:\n" + "\n".join(errors)

    def test_services_count(self):
        loader = ManifestLoader(MANIFESTS_DIR)
        services = loader.load_services()
        assert len(services) >= 10, f"Expected at least 10 services, found {len(services)}"

    def test_models_count(self):
        loader = ManifestLoader(MANIFESTS_DIR)
        models = loader.load_models()
        assert len(models) >= 9, f"Expected at least 9 models, found {len(models)}"

    def test_hardware_count(self):
        loader = ManifestLoader(MANIFESTS_DIR)
        hw = loader.load_hardware_topologies()
        assert "kaggle-dual-t4" in hw
