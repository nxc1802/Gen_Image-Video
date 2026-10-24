"""
services/decision/src/config.py
Configuration loader for the Decision routing daemon.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class DecisionModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    alias: str = "qwen-decision-8b"
    quantization: str = "none"
    precision: str = "bf16"
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048


class DecisionHardwareConfig(BaseModel):
    gpu_required: bool = True
    gpu_device_index: int = 0
    min_vram_gb: float = 6.0
    allocation_policy: str = "single_gpu"


class DecisionSettings(BaseModel):
    service_id: str = "decision-service"
    name: str = "vLLM Decision Routing Engine"
    host: str = "0.0.0.0"
    port: int = 8082
    static_domain: str = "decision.ai.example.com"
    protocol: str = "http"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    model: DecisionModelConfig = Field(default_factory=DecisionModelConfig)
    hardware: DecisionHardwareConfig = Field(default_factory=DecisionHardwareConfig)


def load_decision_config(config_path: Optional[str] = None) -> DecisionSettings:
    if not config_path:
        default_yaml = Path(__file__).parent.parent / "configs" / "decision_config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    config = DecisionSettings(**config_data)

    if "PORT" in os.environ:
        try:
            config.port = int(os.environ["PORT"])
        except ValueError:
            pass
    if "DECISION_SERVICE_PORT" in os.environ:
        try:
            config.port = int(os.environ["DECISION_SERVICE_PORT"])
        except ValueError:
            pass
    if "HOST" in os.environ:
        config.host = os.environ["HOST"]

    return config
