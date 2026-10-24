"""
factory_contracts/manifests/service.py
Service Manifest Schema
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SubsystemType(str, Enum):
    TEXT = "text"
    VLM = "vlm"
    DECISION = "decision"
    EMBEDDING = "embedding"
    SPEECH = "speech"
    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"
    ENHANCEMENT = "enhancement"
    JOBS = "jobs"


class FrameworkType(str, Enum):
    VLLM = "vllm"
    COMFYUI = "comfyui"
    ACESTEP = "acestep"
    FASTAPI = "fastapi"
    CUSTOM = "custom"


class ProtocolType(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    DECISION = "decision"
    MULTIMODAL = "multimodal"


class ServiceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-z0-9-]+$", description="Unique service ID, e.g. 'image-service'")
    name: str = Field(..., min_length=3, description="Human-readable service title")
    subsystem: SubsystemType = Field(..., description="Subsystem domain categorization")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="SemVer string")


class NetworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    static_domain: str = Field(..., min_length=4, description="Reserved ngrok static domain or template")
    internal_port: int = Field(..., ge=1024, le=65535, description="Local listening port on worker host")
    protocol: ProtocolType = Field(default=ProtocolType.HTTP, description="Primary network protocol")
    auth_required: bool = Field(default=True, description="Enforce authentication at ingress")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")

    @field_validator("auth_required")
    @classmethod
    def validate_auth_required(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Factory v4 forbids unauthenticated services (auth_required must be True)")
        return v


class HealthCheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(default="/healthz", pattern=r"^/", description="Health probe path")
    interval_seconds: int = Field(default=15, ge=1, le=300)
    timeout_seconds: int = Field(default=5, ge=1, le=60)
    unhealthy_threshold: int = Field(default=3, ge=1, le=10)


class FrameworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: FrameworkType = Field(..., description="Underlying inference framework")
    version: str = Field(..., description="Framework version requirement")
    entrypoint: str = Field(..., description="Executable path or launch command")


class HardwareRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_required: bool = Field(default=True)
    gpu_device_index: Optional[int] = Field(default=None, ge=0, le=7, description="Target GPU index on host")
    min_vram_gb: float = Field(default=0.0, ge=0.0, description="Minimum GPU VRAM required in gigabytes")
    min_ram_gb: float = Field(default=4.0, ge=1.0, description="Minimum System RAM required in gigabytes")
    cuda_version_min: Optional[str] = Field(default="12.1", description="Minimum CUDA version")


class AuthPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: str = Field(default="bearer", description="Authentication scheme")
    jwt_audience: Optional[str] = Field(default=None, description="Audience claim for JWT verification")
    rate_limit_rpm: int = Field(default=60, ge=1, description="Requests per minute per client key")
    allow_unauthenticated: bool = Field(default=False, description="Strictly false in v4")

    @field_validator("allow_unauthenticated")
    @classmethod
    def validate_no_bypass(cls, v: bool) -> bool:
        if v:
            raise ValueError("Factory v4 prohibits unauthenticated public bypass (allow_unauthenticated must be False)")
        return v


class ServiceManifest(BaseModel):
    """Declarative definition of a Factory v4 Subsystem Service."""
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="ServiceManifest")
    metadata: ServiceMetadata
    network: NetworkSpec
    health_check: HealthCheckSpec
    framework: FrameworkSpec
    hardware: HardwareRequirements
    supported_modalities: List[ModalityType]
    auth_policy: AuthPolicySpec
