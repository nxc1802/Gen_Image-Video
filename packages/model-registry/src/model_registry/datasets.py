"""Pinned dataset preparation views derived only from ModelManifest."""
from collections.abc import Sequence, Mapping
from .loader import ManifestLoader


def dataset_specs():
    models = ManifestLoader().load_models()
    services = ManifestLoader().load_services()
    result = []
    for model_id, model in models.items():
        weights = model.weights
        primary_files = []
        extra_files = []
        for spec in weights.files.values():
            item = {"src": spec.path, "dest": spec.destination or spec.path,
                    "sha256": spec.sha256, "size_bytes": spec.size_bytes}
            if spec.provider != "huggingface":
                raise ValueError("Legacy dataset publisher only supports explicit Hugging Face file sources")
            if spec.repository == weights.repository and spec.revision == weights.checkpoint_revision:
                primary_files.append(item)
            else:
                extra_files.append({**item, "repo": spec.repository, "revision": spec.revision})
        # v1 compatibility is confined to test/migration catalogs, not authored
        # as another hardcoded table of production checkpoint identities.
        if not weights.files:
            primary_files = [{"src": weights.weights_file, "dest": weights.weights_file}] if weights.weights_file else []
        service = services.get(model.metadata.service_id)
        result.append({"id": model_id, "slug": model_id, "title": model.metadata.display_name,
            "repo": weights.repository, "huggingface": weights.repository,
            "revision": weights.checkpoint_revision, "license": model.metadata.license,
            "license_name": model.metadata.license if model.metadata.license in {"mit", "apache-2.0", "bsd-3-clause", "cc-by-nc-4.0"} else "other", "commercial_use": model.metadata.commercial_use,
            "service_id": model.metadata.service_id, "service": service.metadata.subsystem.value if service else None,
            "target_gpu": None if model.apiVersion == "factory.ai/v2" else model.hardware.preferred_gpu_index,
            "canonical": model.apiVersion == "factory.ai/v2", "components": weights.components,
            "format": weights.format.value, "min_vram_gb": model.hardware.min_vram_gb,
            "estimated_gb": sum(f.size_bytes for f in weights.files.values()) / 1024**3,
            "files": primary_files, "extra_downloads": extra_files, "snapshot": False})
    return result


class CatalogDatasetSpecs(Sequence):
    def __init__(self, bootstrap=False):
        self.bootstrap = bootstrap

    def _data(self):
        specs = dataset_specs()
        if self.bootstrap:
            for spec in specs:
                spec["expected_files"] = [*spec["files"], *spec["extra_downloads"]]
                spec["files"] = [item["dest"] for item in spec["expected_files"]] if spec["canonical"] else [spec["files"][0]["src"]] if spec["files"] else []
        return specs

    def __getitem__(self, key):
        return self._data()[key]

    def __len__(self):
        return len(self._data())

    def __iter__(self):
        return iter(self._data())


class CatalogServiceBackends(Mapping):
    def _data(self):
        models = ManifestLoader().load_models()
        services = ManifestLoader().load_services()
        return {mid: {"service_id": m.metadata.service_id,
            "port": services[m.metadata.service_id].network.internal_port,
            "health_endpoint": services[m.metadata.service_id].health_check.endpoint,
            "target_gpu": None if m.apiVersion == "factory.ai/v2" else m.hardware.preferred_gpu_index} for mid, m in models.items()}

    def __getitem__(self, key):
        return self._data()[key]

    def __len__(self):
        return len(self._data())

    def __iter__(self):
        return iter(self._data())
