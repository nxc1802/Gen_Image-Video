"""
factory_contracts/manifests/model.py
Model Manifest Schema & Strict Completeness Rule Enforcement
"""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelStatus(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


class ModelWeightFormat(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    DIFFUSERS = "diffusers"
    TORCH = "torch"
    ONNX = "onnx"


class QuantizationType(str, Enum):
    NONE = "none"
    FP8 = "fp8"
    Q4_K_M = "q4_k_m"
    Q8_0 = "q8_0"
    AWQ = "awq"
    GPTQ = "gptq"
    FOUR_BIT = "4bit"


class PrecisionType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AllocationPolicy(str, Enum):
    SINGLE_GPU = "single_gpu"
    DUAL_GPU_SPLIT = "dual_gpu_split"
    OFFLOAD_RAM = "offload_ram"


class ModelWeightsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_checkpoint: str = Field(..., description="Hugging Face repo or local checkpoint path")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Specialized component paths (e.g. clip, vae, inpaint, controlnet)"
    )
    format: ModelWeightFormat = Field(default=ModelWeightFormat.SAFETENSORS)
    quantization: QuantizationType = Field(default=QuantizationType.NONE)
    default_precision: PrecisionType = Field(default=PrecisionType.BF16)


class ModelHardwareSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_vram_gb: float = Field(..., ge=0.0, description="Minimum dedicated VRAM required to load")
    context_window: Optional[int] = Field(default=None, ge=512, description="Max token context for text/VLM")
    preferred_gpu: str = Field(default="t4", description="Target GPU architecture (t4, l4, a100, etc.)")
    allocation_policy: AllocationPolicy = Field(default=AllocationPolicy.SINGLE_GPU)


class ModelLimitsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_context_tokens: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_frames: Optional[int] = None
    max_duration_seconds: Optional[float] = None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-z0-9-.]+$", description="Logical model family ID, e.g. 'qwen-image-2.1'")
    family: str = Field(..., description="Model family grouping, e.g. 'qwen-image'")
    display_name: str = Field(..., description="Human-readable title")
    service_id: str = Field(..., description="Owning service reference matching ServiceManifest.metadata.id")
    version: str = Field(..., description="Model version")


class ModelManifest(BaseModel):
    """Declarative definition of a Model Family with strict capability completeness enforcement."""
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="ModelManifest")
    metadata: ModelMetadata
    weights: ModelWeightsSpec
    hardware: ModelHardwareSpec
    capabilities: Dict[str, bool] = Field(
        ...,
        description="Dictionary mapping standard capability identifiers to boolean support"
    )
    status: ModelStatus = Field(..., description="'full' if 100% backed; otherwise 'partial'")
    coverage_percent: float = Field(..., ge=0.0, le=100.0, description="Calculated percentage of supported capabilities")
    missing_capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities not backed by checkpoints/workflows"
    )
    limits: ModelLimitsSpec = Field(default_factory=ModelLimitsSpec)

    @model_validator(mode="after")
    def verify_completeness_integrity(self) -> "ModelManifest":
        """
        Enforce Non-Negotiable Completeness Rule:
        A model family is ONLY 'full' if coverage_percent == 100.0 AND missing_capabilities is empty
        AND all capabilities in self.capabilities are True.
        """
        false_caps = [k for k, v in self.capabilities.items() if not v]
        
        if self.status == ModelStatus.FULL:
            if self.coverage_percent < 100.0:
                raise ValueError(
                    f"Model '{self.metadata.id}' cannot be marked 'full': "
                    f"coverage_percent is {self.coverage_percent}%, expected 100.0%"
                )
            if len(self.missing_capabilities) > 0:
                raise ValueError(
                    f"Model '{self.metadata.id}' marked 'full' has missing capabilities: "
                    f"{self.missing_capabilities}"
                )
            if false_caps:
                raise ValueError(
                    f"Model '{self.metadata.id}' marked 'full' contains unbacked capabilities: {false_caps}"
                )
        elif self.status == ModelStatus.PARTIAL:
            if self.coverage_percent >= 100.0 and len(self.missing_capabilities) == 0 and not false_caps:
                raise ValueError(
                    f"Model '{self.metadata.id}' has 100% coverage and no missing capabilities; "
                    "it must be marked status: 'full'"
                )
        return self
