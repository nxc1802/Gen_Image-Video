"""
factory_contracts/manifests/capability.py
Capability Manifest Schema
"""
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ParameterDataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    IMAGE_URL = "image_url"
    AUDIO_URL = "audio_url"
    VIDEO_URL = "video_url"
    MASK_URL = "mask_url"
    JSON = "json"
    ARRAY = "array"


class ParameterConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    max_length: Optional[int] = None
    step: Optional[float] = None


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z0-9_]+$")
    type: ParameterDataType
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[ParameterConstraint] = None


class OutputDataType(str, Enum):
    IMAGE_URL = "image_url"
    VIDEO_URL = "video_url"
    AUDIO_URL = "audio_url"
    TEXT = "text"
    JSON = "json"
    VECTOR = "vector"


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z0-9_]+$")
    type: OutputDataType
    format: str = Field(..., description="MIME type or serialization format, e.g. 'image/png', 'video/mp4'")


class ValidationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_fixture: str = Field(..., description="Relative path or name of test input JSON fixture")
    golden_output_hash: Optional[str] = None
    verification_method: str = Field(default="schema_check", description="schema_check | perceptual_hash | metric_eval")


class CapabilityManifest(BaseModel):
    """Declarative semantic specification for a single capability."""
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="CapabilityManifest")
    id: str = Field(..., pattern=r"^[a-z0-9_.-]+$", description="e.g. 'image.inpaint' or 'image.t2i'")
    subsystem: str = Field(..., description="Subsystem domain (image, video, speech, etc.)")
    display_name: str
    description: str
    inputs: List[ParameterSpec]
    outputs: List[OutputSpec]
    validation: ValidationSpec
    supported_model_families: List[str] = Field(default_factory=list)
