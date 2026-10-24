"""
Multimodal embedding adapter.

A text representation of an image/video URL is not a multimodal embedding.
Until a real multimodal embedding backend is configured this path fails closed.
"""
import os
from typing import Any

import httpx

from factory_contracts.errors import ModelUnavailableError
from services.embedding.src.schemas import EmbeddingItem
from services.text.src.multimodal import validate_url_security


class MultimodalEmbedder:
    def __init__(self, text_embedder=None):
        self.upstream_url = os.environ.get("MULTIMODAL_EMBEDDING_URL", "").rstrip("/")
        self.api_key = os.environ.get("MULTIMODAL_EMBEDDING_API_KEY") or os.environ.get("FACTORY_API_KEY", "")

    def embed_multimodal_item(self, item: Any, index: int, dimensions: int) -> EmbeddingItem:
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Real multimodal embedding backend is not configured.",
                details={"required_env": "MULTIMODAL_EMBEDDING_URL"},
            )
        if not isinstance(item, dict):
            raise ModelUnavailableError(message="Multimodal embedding input must be an object.")

        url = item.get("url")
        if not url and isinstance(item.get("image_url"), dict):
            url = item["image_url"].get("url")
        if url and not str(url).startswith("data:"):
            validate_url_security(str(url))

        payload = {"input": item, "dimensions": dimensions}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{self.upstream_url}/v1/embeddings", json=payload, headers=headers)
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Multimodal embedding backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                data = response.json()
                vector = data["data"][0]["embedding"]
                return EmbeddingItem(index=index, embedding=vector)
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Multimodal embedding backend is unavailable or returned an invalid response.",
                details={"error_type": type(exc).__name__},
            ) from exc
