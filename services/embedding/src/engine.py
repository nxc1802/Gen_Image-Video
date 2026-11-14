"""Compatibility import for the canonical, manifest-selected VectorEngine."""
from factory.engines.vector import VectorEngine

EmbeddingEngine = VectorEngine
__all__ = ["EmbeddingEngine", "VectorEngine"]
