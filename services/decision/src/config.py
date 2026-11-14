"""
services/decision/src/config.py
Configuration loader for the Decision routing daemon.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from model_registry.bindings import bind_service_config
from pydantic import BaseModel, Field


class DecisionModelConfig(BaseModel):
    name: str
    alias: str
    quantization: str
    precision: str
    max_context_tokens: int
    max_output_tokens: int

    revision: str


class DecisionHardwareConfig(BaseModel):
    gpu_required: bool = True
    gpu_device_index: Optional[int] = None
    min_vram_gb: float
    allocation_policy: str


class DecisionSettings(BaseModel):
    backend: str
    service_id: str = "decision-service"
    name: str = "vLLM Decision Routing Engine"
    host: str = "0.0.0.0"
    port: int = 8082
    static_domain: str = "decision.ai.example.com"
    protocol: str = "http"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    model: DecisionModelConfig
    hardware: DecisionHardwareConfig


def load_decision_config(config_path: Optional[str] = None) -> DecisionSettings:
    if not config_path:
        default_yaml = Path(__file__).parent.parent / "configs" / "decision_config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    config_data = bind_service_config(config_data)
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
