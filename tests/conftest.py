# Root tests conftest
import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory.apps.control_api.src.main import app
from factory_auth.config import load_auth_config


@pytest.fixture(scope="module")
def legacy_catalog():
    """Explicit v1 fixtures for legacy lifecycle tests; never inference evidence.

    Canonical catalog/identity tests use real manifests without this fixture.
    Fixture readiness is simulated and never changes production catalog state.
    """
    fixture_root = Path(__file__).parent / "fixtures/legacy_manifests"
    patch = pytest.MonkeyPatch()
    patch.setenv("FACTORY_MANIFESTS_DIR", str(fixture_root))
    patch.setenv("FACTORY_DEPLOYMENT_FILE", str(fixture_root / "deployment.yaml"))
    patch.setenv("ENVIRONMENT", "test")
    try:
        yield fixture_root
    finally:
        patch.undo()


@pytest.fixture
def legacy_workspace(tmp_path, legacy_catalog):
    """Workspace layout for historical preflight algorithms, not production."""
    (tmp_path / "manifests").symlink_to(legacy_catalog, target_is_directory=True)
    return tmp_path


@pytest.fixture
def routing_registry(tmp_path):
    """Explicit canonical bindings for software queue contracts; no activation."""
    from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
    from model_registry.deployment import ModelCatalog, DeploymentRegistry
    root = Path(__file__).resolve().parent.parent
    def build(entries):
        config = DeploymentConfig(deployments=tuple(DeploymentEntry(model=model, devices=tuple(devices),
            exclusive=len(devices) > 1, port=9200 + i) for i, (model, devices) in enumerate(entries)))
        return DeploymentRegistry(config, ModelCatalog(root / "manifests"), config_path=tmp_path / "deployment.yaml")
    return build


@pytest.fixture
def legacy_routing(monkeypatch):
    """Function-isolated v1 routing for mixed canonical/legacy test modules."""
    root = Path(__file__).parent / "fixtures/legacy_manifests"
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(root))
    monkeypatch.setenv("FACTORY_DEPLOYMENT_FILE", str(root / "deployment.yaml"))
    monkeypatch.setenv("ENVIRONMENT", "test")
    return root


@pytest.fixture(autouse=True)
def reset_auth_config(request):
    """Ensure hermetic auth configuration isolation across all tests."""
    if any("legacy_catalog" in marker.args for marker in request.node.iter_markers("usefixtures")):
        request.getfixturevalue("legacy_catalog")
    auth_keys = ("FACTORY_API_KEY", "ENVIRONMENT", "FACTORY_ENV", "JWT_SECRET_KEY", "JWT_AUDIENCE", "JWT_ISSUER")
    orig_env = {k: os.environ[k] for k in auth_keys if k in os.environ}
    app.state.auth_config = load_auth_config()
    migration=sys.modules.get("factory.apps.control_api.src.legacy")
    if migration is not None:migration.app.state.auth_config=app.state.auth_config
    yield
    for k in auth_keys:
        if k in orig_env:
            os.environ[k] = orig_env[k]
        else:
            os.environ.pop(k, None)
    app.state.auth_config = load_auth_config()
    migration=sys.modules.get("factory.apps.control_api.src.legacy")
    if migration is not None:migration.app.state.auth_config=app.state.auth_config
