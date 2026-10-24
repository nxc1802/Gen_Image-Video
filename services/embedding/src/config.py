"""
services/embedding/src/config.py
Configuration loader for the Vector Embedding & Rerank daemon.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class EmbeddingModelConfig(BaseModel):
    embedding_name: str = "Qwen/Qwen-Embedding-0.6B"
    embedding_alias: str = "qwen-embedding-8b"
    rerank_name: str = "Qwen/Qwen-Rerank-0.6B"
    rerank_alias: str = "qwen-rerank-8b"
    dimensions: int = 1024
    max_context_tokens: int = 8192


class EmbeddingHardwareConfig(BaseModel):
    gpu_required: bool = True
    gpu_device_index: int = 1
    min_vram_gb: float = 4.0
    allocation_policy: str = "single_gpu"


class EmbeddingSettings(BaseModel):
    service_id: str = "embed-service"
    name: str = "vLLM / TEI Vector Embedding & Rerank Daemon"
    host: str = "0.0.0.0"
    port: int = 8083
    static_domain: str = "embed.ai.example.com"
    protocol: str = "http"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    hardware: EmbeddingHardwareConfig = Field(default_factory=EmbeddingHardwareConfig)


def load_embedding_config(config_path: Optional[str] = None) -> EmbeddingSettings:
    if not config_path:
        default_yaml = Path(__file__).parent.parent / "configs" / "embedding_config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

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
