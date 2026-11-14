"""Compatibility projections for legacy launchers, derived from ModelManifest."""
from .loader import ManifestLoader


def model_binding(model_id):
    models = ManifestLoader().load_models()
    if model_id not in models:
        raise ValueError(f"Unknown explicit model_ref: {model_id}")
    model = models[model_id]
    if model.apiVersion != "factory.ai/v2":
        raise ValueError("Legacy service bindings require a canonical v2 model manifest")
    return model


def project_model(model):
    parameters = model.parameters
    return {
        "name": model.weights.primary_checkpoint if model.weights.format.value == "gguf" else model.weights.repository,
        "alias": model.metadata.id,
        "quantization": model.weights.quantization.value,
        "precision": model.weights.default_precision.value,
        "max_context_tokens": parameters.get("max_context_tokens", {}).get("default"),
        "max_output_tokens": parameters.get("max_output_tokens", {}).get("default"),
        "revision": model.weights.checkpoint_revision,
    }


def project_hardware(model, configured=None):
    configured = configured or {}
    if set(configured) - {"gpu_device_index"}:
        raise ValueError("Model requirements belong in ModelManifest; service hardware may only assign a device")
    profile = model.runtime_profiles["kaggle-dual-t4"]
    return {"gpu_required": True, "gpu_device_index": configured.get("gpu_device_index"),
            "min_vram_gb": profile.min_vram_gb,
            "allocation_policy": "dual_gpu_split" if profile.gpu_count > 1 else "single_gpu"}


def bind_service_config(data):
    if "model" in data:
        raise ValueError("Inline model identity is retired; configure model_ref")
    if not data.get("model_ref"):
        raise ValueError("An explicit model_ref is required")
    model = model_binding(data["model_ref"])
    return {**data, "model": project_model(model), "hardware": project_hardware(model, data.get("hardware")),
            "backend": model.runtime_profiles["kaggle-dual-t4"].preferred_backend}


def bind_vector_config(data):
    if "model" in data:
        raise ValueError("Inline vector checkpoint names are retired; configure model_refs")
    refs = data.get("model_refs", {})
    embedding, reranker = model_binding(refs["embedding"]), model_binding(refs["rerank"])
    return {**data, "model": {
        "embedding_name": embedding.weights.repository, "embedding_alias": embedding.metadata.id,
        "rerank_name": reranker.weights.repository, "rerank_alias": reranker.metadata.id,
        "dimensions": embedding.parameters["dimensions"]["default"],
        "max_context_tokens": embedding.parameters["max_context_tokens"]["default"],
        "embedding_revision": embedding.weights.checkpoint_revision, "rerank_revision": reranker.weights.checkpoint_revision},
        "hardware": project_hardware(embedding, data.get("hardware"))}
