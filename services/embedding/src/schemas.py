"""Compatibility imports; canonical vector contracts live with VectorEngine."""
from factory.engines.vector.contracts import (
    EmbeddingRequest, EmbeddingItem, EmbeddingUsage, EmbeddingResponse,
    RerankRequest, RerankResultItem, RerankResponse,
)

__all__ = ["EmbeddingRequest", "EmbeddingItem", "EmbeddingUsage", "EmbeddingResponse", "RerankRequest", "RerankResultItem", "RerankResponse"]
