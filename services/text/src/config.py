"""
services/text/src/config.py
Configuration loader for the Text generation daemon.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class TextModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    alias: str = "qwen-2.5-72b-instruct"
    quantization: str = "awq"
    precision: str = "fp16"
    max_context_tokens: int = 32768
    max_output_tokens: int = 4096


class TextHardwareConfig(BaseModel):
    gpu_required: bool = True
    gpu_device_index: int = 0
    min_vram_gb: float = 12.0
    allocation_policy: str = "single_gpu"


class TextServiceConfig(BaseModel):
    service_id: str = "text-service"
    name: str = "vLLM Text Generation Daemon"
    host: str = "0.0.0.0"
    port: int = 8080
    static_domain: str = "text.ai.example.com"
    protocol: str = "http"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    model: TextModelConfig = Field(default_factory=TextModelConfig)
    hardware: TextHardwareConfig = Field(default_factory=TextHardwareConfig)


def load_text_config(config_path: Optional[str] = None) -> TextServiceConfig:
    """Loads text service configuration from YAML file or environment overrides."""
    if not config_path:
        default_yaml = Path(__file__).parent.parent / "configs" / "text_config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    config = TextServiceConfig(**config_data)

    # Environment variable overrides
    if "PORT" in os.environ:
        try:
            config.port = int(os.environ["PORT"])
        except ValueError:
            pass
    if "TEXT_SERVICE_PORT" in os.environ:
        try:
            config.port = int(os.environ["TEXT_SERVICE_PORT"])
        except ValueError:
            pass
    if "HOST" in os.environ:
        config.host = os.environ["HOST"]

    return config
