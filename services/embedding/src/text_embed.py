"""
Strict adapter for real vLLM/TEI embeddings.

Pseudo-hash/random/token-projection embeddings are intentionally forbidden.
"""
import os
from typing import Any, Dict, List, Optional

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.embedding.src.schemas import EmbeddingItem, EmbeddingResponse, EmbeddingUsage

OFFLINE_EMBEDDING_MODELS = {
    "failing-embedding-model", "offline-model", "qwen-offline", "failing-model"
}


class VLLMEmbeddingAdapter:
    def __init__(self, upstream_url: Optional[str] = None, timeout: float = 60.0):
        self.upstream_url = (
            upstream_url
            or os.environ.get("VLLM_EMBEDDING_URL")
            or os.environ.get("TEI_EMBEDDING_URL")
            or ""
        ).rstrip("/")
        self.api_key = os.environ.get("VLLM_EMBEDDING_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.upstream_url)

    def embed(self, model: str, texts: List[str], dimensions: Optional[int] = None) -> EmbeddingResponse:
        if not self.is_configured:
            raise ModelUnavailableError(
                message="Real embedding backend is not configured. Fake vectors are disabled.",
                details={"required_env": "VLLM_EMBEDDING_URL or TEI_EMBEDDING_URL"},
            )

        payload: Dict[str, Any] = {"model": model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.upstream_url}/v1/embeddings", headers=headers, json=payload)
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Real embedding backend rejected the request.",
                        details={"status_code": response.status_code, "model": model},
                    )
                data = response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Real embedding backend is unavailable.",
                details={"error_type": type(exc).__name__, "model": model},
            ) from exc

        items = [
            EmbeddingItem(index=item.get("index", idx), embedding=item["embedding"])
            for idx, item in enumerate(data.get("data", []))
        ]
        if len(items) != len(texts):
            raise ModelUnavailableError(
                message="Embedding backend returned an invalid result count.",
                details={"expected": len(texts), "received": len(items)},
            )

        usage = data.get("usage", {})
        return EmbeddingResponse(
            object="list",
            data=items,
            model=data.get("model", model),
            usage=EmbeddingUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", usage.get("prompt_tokens", 0))),
            ),
        )


class TextEmbedder:
    def __init__(self, default_dimensions: int = 1024, upstream_url: Optional[str] = None):
        self.default_dimensions = default_dimensions
        self.adapter = VLLMEmbeddingAdapter(upstream_url=upstream_url)

    def check_model_availability(self, model: str) -> None:
        if not model:
            raise InvalidRequestError("Field 'model' is required.")
        if model in OFFLINE_EMBEDDING_MODELS:
            raise ModelUnavailableError(message=f"Embedding model '{model}' is offline.")

    def embed_texts(self, model: str, texts: List[str], dimensions: Optional[int] = None) -> EmbeddingResponse:
        self.check_model_availability(model)
        if not texts:
            raise InvalidRequestError("At least one text input is required.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise InvalidRequestError("Embedding inputs must be non-empty strings.")
        dim = dimensions or self.default_dimensions
        if dim <= 0:
            raise InvalidRequestError("Dimensions must be a positive integer.")
        return self.adapter.embed(model=model, texts=texts, dimensions=dim)
