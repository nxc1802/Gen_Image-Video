"""Model catalog, immutable configured bindings, and session-local visibility.

Configuration is never activation. Backend lifecycle and evidence authorization
must supply their own current-session proof before a deployment can be public.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import uuid

import yaml
from factory_contracts.deployment import DeploymentConfig, BackendOptions
from factory_contracts.errors import ModelNotLoadedError
from factory_contracts.manifests.catalog import model_identity_index, manifest_root
from .loader import ManifestLoader


class ModelCatalog:
    def __init__(self, manifests_dir=None):
        self.loader = ManifestLoader(manifests_dir)
        self._models = self.loader.load_models()
        self._identities, _ = model_identity_index(self.loader.manifests_dir)

    def resolve(self, identity):
        key = str(identity).strip().lower()
        mid = self._identities.get(key)
        if mid is None:
            raise ValueError(f"Unknown model identity: {identity}; select an explicit catalog model")
        return mid

    def get(self, identity):
        return self._models[self.resolve(identity)].model_copy(deep=True)

    def models(self):
        return {mid: model.model_copy(deep=True) for mid, model in self._models.items()}


def deployment_path(path=None):
    return Path(path or os.environ.get("FACTORY_DEPLOYMENT_FILE", "factory.deployment.yaml")).expanduser().resolve()


def read_config(path):
    return DeploymentConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def write_config(path, config, *, create=False):
    """Atomic replacement; init refuses overwriting an existing configuration."""
    path = Path(path)
    payload = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    if create:
        with path.open("x") as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DeploymentBinding:
    model: str
    backend: str
    adapter: str
    devices: tuple[int, ...]
    exclusive: bool
    port: int
    weights_dir: Path
    checkpoint_graph_sha256: str
    backend_options: BackendOptions

    def to_dict(self):
        return {**self.__dict__, "devices": list(self.devices), "weights_dir": str(self.weights_dir),
                "backend_options": self.backend_options.model_dump(mode="json")}


class DeploymentRegistry:
    def __init__(self, config, catalog, *, config_path=None):
        self.config = config.model_copy(deep=True)
        self.catalog = catalog
        self.session_id = str(uuid.uuid4())
        self.created_at = int(datetime.now(timezone.utc).timestamp())
        self.config_path = deployment_path(config_path)
        bindings = {}
        for index, entry in enumerate(config.deployments):
            model = catalog.get(entry.model)
            mid = model.metadata.id
            if mid in bindings:
                raise ValueError(f"Duplicate deployment after model alias resolution: {mid}")
            profile = model.runtime_profiles.get(config.hardware)
            backend = entry.backend or (profile.preferred_backend if profile else None)
            if not backend:
                raise ValueError(f"Select an explicit backend for {mid}")
            root = entry.weights_dir or (config.models_dir / mid)
            if not root.is_absolute():
                root = self.config_path.parent / root
            graph = {key: spec.model_dump(mode="json") for key, spec in model.weights.files.items()}
            bindings[mid] = DeploymentBinding(mid, backend, model.backend.adapter if model.backend else "legacy_contract",
                entry.devices, entry.exclusive, entry.port or 9000 + index, root.resolve(),
                hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(), entry.backend_options)
        self._bindings = MappingProxyType(bindings)
        self.config_sha256 = hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()
        self._backend_session = None

    @classmethod
    def from_environment(cls, manifests_dir=None):
        path = deployment_path()
        stat = path.stat() if path.is_file() else None
        return _configured_registry(str(path), stat.st_mtime_ns if stat else 0,
                                    stat.st_size if stat else 0, str(manifest_root(manifests_dir)))

    def assignment(self, model):
        try:
            mid = self.catalog.resolve(model)
        except ValueError as exc:
            raise ModelNotLoadedError(message=f"Model '{model}' is not in this deployment's catalog snapshot.") from exc
        if mid not in self._bindings:
            raise ModelNotLoadedError(message=f"Model '{mid}' is in the catalog but is not configured in this deployment.")
        return self._bindings[mid]

    def bindings(self):
        return tuple(self._bindings.values())

    def snapshot(self):
        return {"hardware": self.config.hardware, "api_host": self.config.api_host, "api_port": self.config.api_port,
                "runtime_dir": str(self.config.runtime_dir),
                "deployments": [binding.to_dict() for binding in self._bindings.values()]}

    def active_models(self):
        # There is deliberately no boolean/config-based activation API. Phase 4
        # supplies lifecycle receipts; V3/V4/semantic authorization is required
        # before production visibility. Until then this registry is fail-closed.
        return ()

    def attach_backend_session(self, session):
        if session.registry is not self or self._backend_session is not None:
            raise ValueError("Backend session must belong to this exact registry and may attach only once")
        self._backend_session = session

    def healthy_backends(self):
        return self._backend_session.healthy_models() if self._backend_session is not None else ()

    def require_active(self, model, capability=None):
        self.assignment(model)
        raise ModelNotLoadedError(message=f"Model '{model}' is configured but has no current-session backend/evidence authorization.")

    def status(self):
        return {"session_id": self.session_id, "config_sha256": self.config_sha256,
            "created_at": self.created_at,
            "configured": self.snapshot(), "active_models": [], "production_ready": False,
            "backend_lifecycle": self._backend_session.status() if self._backend_session is not None else "NOT_STARTED",
            "healthy_backends": list(self.healthy_backends()), "native_inference_verified": False}


@lru_cache(maxsize=16)
def _configured_registry(path, mtime, size, manifests_dir):
    """Standalone queue components share one immutable configuration snapshot.

    API lifespans construct their own registry and inject that same instance.
    No request reloads or rewrites deployment placement.
    """
    config = read_config(path) if Path(path).is_file() else DeploymentConfig()
    return DeploymentRegistry(config, ModelCatalog(manifests_dir), config_path=path)
