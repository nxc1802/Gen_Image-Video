"""
services/embedding/src/multimodal_embed.py
Multimodal embedding support for image/video inputs with SSRF validation.
"""
from typing import Any, Dict, List
from services.embedding.src.schemas import EmbeddingItem
from services.embedding.src.text_embed import TextEmbedder
from services.text.src.multimodal import validate_url_security


class MultimodalEmbedder:
    """Computes dense vector representations for multimodal content."""

    def __init__(self, text_embedder: TextEmbedder):
        self.text_embedder = text_embedder

    def embed_multimodal_item(self, item: Any, index: int, dimensions: int) -> EmbeddingItem:
        """Parses multimodal item, validates URLs, and generates normalized embedding."""
        if isinstance(item, str):
            vec = self.text_embedder._encode_single_text(item, dimensions)
            return EmbeddingItem(index=index, embedding=vec)

        if isinstance(item, dict):
            url = item.get("url") or item.get("image_url", {}).get("url")
            if url and not url.startswith("data:"):
                validate_url_security(url)
            text_repr = str(item.get("text") or url or "multimodal_asset")
            vec = self.text_embedder._encode_single_text(text_repr, dimensions)
            return EmbeddingItem(index=index, embedding=vec)

        vec = self.text_embedder._encode_single_text(str(item), dimensions)
        return EmbeddingItem(index=index, embedding=vec)
