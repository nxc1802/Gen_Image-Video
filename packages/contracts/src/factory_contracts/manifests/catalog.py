"""Shared manifest reading and model identity resolution; no deployment state."""
from collections.abc import Mapping
from functools import lru_cache
import os
from pathlib import Path

import yaml

from .model import ModelManifest


def manifest_root(path=None):
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get("FACTORY_MANIFESTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if (Path.cwd() / "manifests/models").is_dir():
        return (Path.cwd() / "manifests").resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "manifests/models").is_dir():
            return parent / "manifests"
    raise ValueError("Set FACTORY_MANIFESTS_DIR to the model catalog directory")


def _fingerprint(root):
    paths = sorted([*(root / "models").glob("*.yaml"), *(root / "models").glob("*.yml")])
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


@lru_cache(maxsize=16)
def _read_models(token):
    models, identities = {}, {}
    for filename, _, _ in token:
        # Model identity is literal. Expanding environment variables here would
        # let session environment silently change the selected checkpoint.
        data = yaml.safe_load(Path(filename).read_text())
        if not data:
            raise ValueError(f"Empty model manifest: {filename}")
        model = ModelManifest.model_validate(data)
        model_id = model.metadata.id
        if model_id in models:
            raise ValueError(f"Duplicate model ID: {model_id}")
        models[model_id] = model
        for identity in [model_id, *model.aliases]:
            if identity in identities:
                raise ValueError(f"Duplicate or ambiguous model identity: {identity}")
            identities[identity] = model_id
    return models, identities


def load_model_manifests(path=None):
    models, _ = _read_models(_fingerprint(manifest_root(path)))
    return {key: model.model_copy(deep=True) for key, model in models.items()}


def model_identity_index(path=None):
    models, identities = _read_models(_fingerprint(manifest_root(path)))
    return dict(identities), {key: model.metadata.family for key, model in models.items()}


class CatalogIdentityMapping(Mapping):
    """Compatibility view; the YAML catalog owns every identity and alias."""
    def __init__(self, families=False):
        self.families = families

    def _data(self):
        aliases, families = model_identity_index()
        return families if self.families else aliases

    def __getitem__(self, key):
        return self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())
