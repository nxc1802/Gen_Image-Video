"""
services/embedding/src/config.py
Configuration loader for the Vector Embedding & Rerank daemon.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from model_registry.bindings import bind_vector_config
from pydantic import BaseModel, Field


class EmbeddingModelConfig(BaseModel):
    embedding_name: str
    embedding_alias: str
    rerank_name: str
    rerank_alias: str
    dimensions: int
    max_context_tokens: int

    embedding_revision: str
    rerank_revision: str


class EmbeddingHardwareConfig(BaseModel):
    gpu_required: bool = True
    gpu_device_index: Optional[int] = None
    min_vram_gb: float
    allocation_policy: str


class EmbeddingSettings(BaseModel):
    service_id: str = "embed-service"
    name: str = "vLLM / TEI Vector Embedding & Rerank Daemon"
    host: str = "0.0.0.0"
    port: int = 8083
    static_domain: str = "embed.ai.example.com"
    protocol: str = "http"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    model: EmbeddingModelConfig
    hardware: EmbeddingHardwareConfig


def load_embedding_config(config_path: Optional[str] = None) -> EmbeddingSettings:
    if not config_path:
        default_yaml = Path(__file__).parent.parent / "configs" / "embedding_config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    config_data = bind_vector_config(config_data)
    config = EmbeddingSettings(**config_data)

    if "PORT" in os.environ:
        try:
            config.port = int(os.environ["PORT"])
        except ValueError:
            pass
    if "EMBEDDING_SERVICE_PORT" in os.environ:
        try:
            config.port = int(os.environ["EMBEDDING_SERVICE_PORT"])
        except ValueError:
            pass
    if "HOST" in os.environ:
        config.host = os.environ["HOST"]

    return config
