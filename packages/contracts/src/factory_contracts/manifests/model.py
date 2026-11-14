"""
factory_contracts/manifests/model.py
Model Manifest Schema & Strict Completeness Rule Enforcement
"""
from enum import Enum
from pathlib import PurePosixPath
import re
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelStatus(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


class ModelValidationStatus(str, Enum):
    PENDING_LIVE = "pending_live"
    CONTRACT_VERIFIED = "contract_verified"
    LIVE_VERIFIED = "live_verified"


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
    INT8_CONVROT = "int8_convrot"


class PrecisionType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AllocationPolicy(str, Enum):
    SINGLE_GPU = "single_gpu"
    DUAL_GPU_SPLIT = "dual_gpu_split"
    OFFLOAD_RAM = "offload_ram"


class ModelVariantSpec(BaseModel):
    """Specification of a quantization, distilled, or fine-tuned variant of a model."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-z0-9-._]+$", description="Variant identifier, e.g. 'turbo' or 'q4_k_m'")
    display_name: str = Field(..., description="Human-readable title of variant")
    checkpoint: str = Field(..., description="Variant checkpoint path, filename, or Hugging Face repo")
    format: ModelWeightFormat = Field(default=ModelWeightFormat.GGUF)
    quantization: QuantizationType = Field(default=QuantizationType.Q4_K_M)
    default_precision: PrecisionType = Field(default=PrecisionType.BF16)
    recommended_steps: Optional[int] = Field(default=None, ge=1, le=100, description="Recommended inference steps (e.g. 8 for Turbo)")
    kaggle_dataset: Optional[str] = Field(
        default=None,
        description="Linked Kaggle dataset slug, e.g. 'cuongnguyen1802/qwen-image-2-1-turbo-gguf'"
    )
    huggingface_repo: Optional[str] = Field(
        default=None,
        description="Source Hugging Face repository, e.g. 'unsloth/Qwen-Image-2.1-Turbo-GGUF'"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Target file name within dataset, e.g. 'qwen-image-2.1-turbo-Q4_K_M.gguf'"
    )
    sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA256 checksum of variant weights"
    )
    size_bytes: Optional[int] = Field(default=None, ge=1, description="Size in bytes")
    min_vram_gb: Optional[float] = Field(default=None, ge=1.0, description="Minimum dedicated VRAM in GB")


class CheckpointFileSpec(BaseModel):
    """A pinned file identity. Expected hashes do not attest runtime readiness."""
    model_config = ConfigDict(extra="forbid")

    provider: Literal["huggingface", "kaggle", "github_release"] = "huggingface"
    repository: str = Field(..., pattern=r"^[^\s/]+/[^\s/]+$")
    revision: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(..., gt=0)
    destination: Optional[str] = None

    @model_validator(mode="after")
    def validate_identity(self) -> "CheckpointFileSpec":
        if self.provider == "huggingface" and not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Hugging Face files require an immutable commit SHA")
        if self.provider == "kaggle" and not re.fullmatch(r"[1-9][0-9]*", self.revision):
            raise ValueError("Kaggle files require an explicit positive dataset version")
        for path in (self.path, self.destination or self.path):
            if path in (".", "..") or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path or any(c in path for c in "*?[]"):
                raise ValueError("Checkpoint paths must be exact relative paths without traversal or wildcards")
        return self


class ModelWeightsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: Dict[str, CheckpointFileSpec] = Field(default_factory=dict,
        description="Canonical pinned file identities. The primary key derives legacy fields during migration.")

    primary_checkpoint: str = Field(..., description="Hugging Face repo or local checkpoint path")
    repository: Optional[str] = Field(default=None, description="Exact Hugging Face or ModelScope repository")
    checkpoint_revision: Optional[str] = Field(default=None, description="Exact Git commit SHA or revision tag")
    weights_file: Optional[str] = Field(default=None, description="Exact weights file name")
    sha256: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{64}$", description="SHA256 checksum of primary weights")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Specialized component paths (e.g. clip, vae, inpaint, controlnet)"
    )
    format: ModelWeightFormat = Field(default=ModelWeightFormat.SAFETENSORS)
    quantization: QuantizationType = Field(default=QuantizationType.NONE)
    default_precision: PrecisionType = Field(default=PrecisionType.BF16)
    component_precisions: Dict[str, PrecisionType] = Field(default_factory=dict,
        description="Explicit native component compute precisions; Adapter interprets component names. No automatic dtype fallback.")
    component_parameter_precisions: Dict[str, Dict[str, PrecisionType]] = Field(default_factory=dict,
        description="Exact native parameter-name precision overrides, owned by Manifest and validated by Adapter. No wildcard matching or automatic fallback.")
    variants: Dict[str, ModelVariantSpec] = Field(
        default_factory=dict,
        description="Dictionary mapping variant IDs (e.g. 'turbo') to variant specifications"
    )

    @model_validator(mode="before")
    @classmethod
    def derive_legacy_file_fields(cls, value):
        """Keep legacy readers working without maintaining duplicate YAML pins."""
        if not isinstance(value, dict) or not value.get("files"):
            return value
        data = dict(value)
        if "primary" not in data["files"]:
            raise ValueError("Canonical checkpoint files require a primary identity")
        files = {key: CheckpointFileSpec.model_validate(spec) for key, spec in data["files"].items()}
        primary = files["primary"]
        projected = {
            "primary_checkpoint": f"{primary.repository}/{primary.path}",
            "repository": primary.repository,
            "checkpoint_revision": primary.revision,
            "weights_file": primary.path,
            "sha256": primary.sha256,
        }
        # Primary-only migration retains explicitly unpinned legacy component
        # claims. Full component migration replaces them with file projections.
        if len(files) > 1:
            projected["components"] = {key: PurePosixPath(spec.path).name for key, spec in files.items() if key != "primary"}
        for key, expected in projected.items():
            if key in data and data[key] != expected:
                raise ValueError(f"Legacy weights.{key} conflicts with canonical file identity")
            data[key] = expected
        return data


class ModelHardwareSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_vram_gb: float = Field(..., ge=0.0, description="Minimum dedicated VRAM required to load")
    context_window: Optional[int] = Field(default=None, ge=512, description="Max token context for text/VLM")
    preferred_gpu: str = Field(default="t4", description="Target GPU architecture (t4, l4, a100, etc.)")
    preferred_gpu_index: Optional[int] = Field(default=None, ge=0, le=7, description="Discrete GPU device index (0 or 1 on Dual-T4)")
    quantization_format: Optional[str] = Field(default=None, description="Quantization profile used for target hardware (e.g. 'awq', 'q4_k_m', '4bit', 'none')")
    offload_ram_gb: Optional[float] = Field(default=None, ge=0.0, description="System RAM offload bound if offloading enabled")
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
    license: Optional[str] = Field(default=None, description="Legal license declaration (e.g. 'Tongyi Qianwen License', 'MiniMax Non-Commercial License', 'Apache-2.0', 'CC-BY-NC-4.0')")
    license_url: Optional[str] = Field(default=None, description="URL or reference to upstream legal terms")
    commercial_use: Optional[bool] = Field(default=None, description="Whether commercial use is legally permitted")


class CapabilityTruthLevel(BaseModel):
    """Represents a model capability through Declared, Implemented, and Live Verified levels."""
    model_config = ConfigDict(extra="forbid")

    declared: bool = Field(default=True, description="Feature declared by upstream model card / specification")
    implemented: bool = Field(default=False, description="Factory has real workflow, worker, adapter and artifact path")
    live_verified: bool = Field(default=False, description="Verified through real model execution on hardware profile")
    evidence_run_id: Optional[str] = Field(default=None, description="Artifact or run proof of verification")


class EngineID(str, Enum):
    RESPONSES = "responses"
    DECISION = "decision"
    VECTOR = "vector"
    SPEECH = "speech"
    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"
    ENHANCEMENT = "enhancement"


class ModelBackendSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adapter: str = Field(..., pattern=r"^[a-z0-9_]+$")
    supported: List[str] = Field(..., min_length=1)


class ComponentTilingSpec(BaseModel):
    """Explicit spatial execution policy; supported components belong to the Adapter."""
    model_config = ConfigDict(extra="forbid")
    tile_height: int = Field(..., gt=0, strict=True)
    tile_width: int = Field(..., gt=0, strict=True)
    stride_height: int = Field(..., gt=0, strict=True)
    stride_width: int = Field(..., gt=0, strict=True)

    @model_validator(mode="after")
    def overlapping_tiles(self):
        if self.stride_height >= self.tile_height or self.stride_width >= self.tile_width:
            raise ValueError("Spatial tiling requires positive overlap in both dimensions")
        return self


class ModelRuntimeProfile(BaseModel):
    """Requirements, never a GPU assignment or hardware certification."""
    model_config = ConfigDict(extra="forbid")
    gpu_count: int = Field(..., ge=1)
    min_vram_gb: float = Field(..., gt=0)
    preferred_backend: str
    exclusive: bool = False
    cpu_offload: bool = False
    offload_strategy: Optional[Literal["model", "sequential"]] = None
    component_tiling: Dict[str, ComponentTilingSpec] = Field(default_factory=dict)
    extra_vram_headroom_gb: float = Field(default=0, ge=0, le=8, multiple_of=0.001, allow_inf_nan=False,
        description="Additional preconfigured native allocator headroom beyond topology reservation; never raises the residency acceptance budget.")
    inference_timeout_seconds: int = Field(default=600, ge=1, le=7200)
    notes: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def multi_gpu_exclusivity(self):
        if self.offload_strategy is not None and not self.cpu_offload:
            raise ValueError("An explicit offload strategy requires CPU offload to be enabled")
        if self.gpu_count > 1 and not self.exclusive:
            raise ValueError("Multi-GPU runtime profiles must declare exclusive access")
        return self


class ModelValidationState(str, Enum):
    READY = "READY"
    EXPERIMENTAL = "EXPERIMENTAL"
    BLOCKED = "BLOCKED"
    DISABLED = "DISABLED"
    DEPRECATED = "DEPRECATED"


class ModelValidationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: ModelValidationState = ModelValidationState.EXPERIMENTAL
    reasons: List[str] = Field(..., min_length=1)
    implemented_capabilities: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_static_certification(self):
        if self.state == ModelValidationState.READY:
            raise ValueError("A static manifest cannot grant READY; V3/V4/semantic evidence authorization is not implemented yet")
        return self


class ModelManifest(BaseModel):
    """Model identity and declared support; deployment/inference readiness is separate."""
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="ModelManifest")
    metadata: ModelMetadata
    engine: Optional[EngineID] = None
    backend: Optional[ModelBackendSpec] = None
    aliases: List[str] = Field(default_factory=list)
    parameters: Dict[str, dict] = Field(default_factory=dict)
    runtime_profiles: Dict[str, ModelRuntimeProfile] = Field(default_factory=dict)
    validation: Optional[ModelValidationSpec] = None
    weights: ModelWeightsSpec
    hardware: ModelHardwareSpec
    capabilities: Dict[str, bool] = Field(
        ...,
        description="Dictionary mapping standard capability identifiers to boolean support"
    )
    status: ModelStatus = Field(..., description="Legacy declared-support coverage; never inference certification")
    coverage_percent: float = Field(..., ge=0.0, le=100.0, description="Calculated percentage of supported capabilities")
    missing_capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities not backed by checkpoints/workflows"
    )
    validation_status: ModelValidationStatus = Field(
        default=ModelValidationStatus.PENDING_LIVE,
        description="Runtime certification state; independent from workflow/checkpoint coverage.",
    )
    validation_notes: Optional[str] = Field(default=None)
    limits: ModelLimitsSpec = Field(default_factory=ModelLimitsSpec)

    # Capability Truth Model & Lifecycle Fields (§5.1)
    revision: Optional[str] = Field(default="main", description="Model revision, Git tag, or checkpoint commit")
    dependencies: List[str] = Field(default_factory=list, description="Required runtime dependencies and packages")
    task_variant_matrix: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Matrix mapping tasks to supported variant IDs (e.g. {'text_to_image': ['standard', 'turbo']})"
    )
    validated_hardware_profiles: List[str] = Field(
        default_factory=list,
        description="Hardware profiles on which model has been verified (e.g. ['kaggle-dual-t4'])"
    )
    installed: bool = Field(default=False, description="Whether model weights/runtime are installed locally")
    checkpoint_valid: bool = Field(default=False, description="Whether checkpoint weights pass checksum/integrity")
    backend_ready: bool = Field(default=False, description="Whether target backend is ready for inference")
    evidence_run_id: Optional[str] = Field(default=None, description="Proof evidence run identifier")
    capabilities_truth: Dict[str, CapabilityTruthLevel] = Field(
        default_factory=dict,
        description="Truth model tracking declared, implemented, and live-verified status per capability"
    )

    @model_validator(mode="before")
    @classmethod
    def canonical_manifest_projection(cls, value):
        """v2 YAML owns truth; old fields are compatibility projections only."""
        if not isinstance(value, dict) or value.get("apiVersion") != "factory.ai/v2":
            return value
        data = dict(value)
        for key in ("engine", "backend", "runtime_profiles", "validation"):
            if not data.get(key):
                raise ValueError(f"Canonical manifest requires {key}")
        if data.get("task_variant_matrix") or data.get("weights", {}).get("variants"):
            raise ValueError("Checkpoint variants require separate model IDs, not task_variant_matrix/weights.variants")
        weights = ModelWeightsSpec.model_validate(data["weights"])
        if not weights.files or "primary" not in weights.files:
            raise ValueError("Canonical manifests require complete pinned checkpoint files")
        destinations = [spec.destination or spec.path for spec in weights.files.values()]
        if len(destinations) != len(set(destinations)):
            raise ValueError("Checkpoint files have colliding destinations")
        backend = ModelBackendSpec.model_validate(data["backend"])
        validation = ModelValidationSpec.model_validate(data["validation"])
        profile = ModelRuntimeProfile.model_validate(data["runtime_profiles"]["kaggle-dual-t4"])
        for runtime in data["runtime_profiles"].values():
            parsed = ModelRuntimeProfile.model_validate(runtime)
            if parsed.preferred_backend not in backend.supported:
                raise ValueError("Runtime preferred backend must be supported by the manifest")
        caps = data["capabilities"]
        from factory_contracts.capabilities import require_canonical
        for key in caps:
            require_canonical(key, engine=data["engine"])
        for spec in data.get("parameters", {}).values():
            for key in spec.get("required_for", []):
                require_canonical(key, engine=data["engine"])
        if not caps:
            raise ValueError("Canonical manifests require an explicit capability disposition")
        if any(type(supported) is not bool for supported in caps.values()):
            raise ValueError("Capability support must be an explicit boolean")
        implemented = validation.implemented_capabilities
        if any(not caps.get(key, False) for key in implemented):
            raise ValueError("Implemented capabilities must be explicitly supported by this checkpoint")
        missing = [key for key, supported in caps.items() if not supported]
        limits = {key: spec["default"] for key, spec in data.get("parameters", {}).items()
                  if key in ModelLimitsSpec.model_fields and "default" in spec}
        derived = {
            "hardware": {"min_vram_gb": profile.min_vram_gb, "preferred_gpu": "t4",
                "preferred_gpu_index": None, "quantization_format": weights.quantization.value,
                "offload_ram_gb": None if profile.cpu_offload else 0.0,
                "allocation_policy": "dual_gpu_split" if profile.gpu_count > 1 else "single_gpu",
                "context_window": limits.get("max_context_tokens")},
            "limits": limits,
            "status": "partial" if missing else "full",
            "coverage_percent": round(100 * sum(caps.values()) / len(caps), 2) if caps else 0.0,
            "missing_capabilities": missing, "installed": False, "backend_ready": False,
            "checkpoint_valid": False, "validation_status": "pending_live",
            "validated_hardware_profiles": [], "evidence_run_id": None,
            "validation_notes": " ".join(validation.reasons),
            "capabilities_truth": {key: {"declared": supported, "implemented": key in implemented,
                "live_verified": False, "evidence_run_id": None} for key, supported in caps.items()},
        }
        for key, expected in derived.items():
            if key in data and data[key] != expected:
                raise ValueError(f"Legacy field {key} conflicts with canonical manifest truth")
            data[key] = expected
        for alias in data.get("aliases", []):
            if alias != alias.strip().lower() or not re.fullmatch(r"[a-z0-9_.-]+", alias):
                raise ValueError("Model aliases must be explicit normalized identifiers")
        return data

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

        # Auto-populate capabilities_truth if not explicitly defined
        # A model-wide legacy label cannot prove per-capability inference.
        # Runtime installation/health likewise must never be inferred here.
        for cap, val in self.capabilities.items():
            if cap not in self.capabilities_truth:
                self.capabilities_truth[cap] = CapabilityTruthLevel(
                    declared=True,
                    implemented=bool(val),
                    live_verified=False,
                    evidence_run_id=None,
                )
            else:
                # Synchronize implemented flag with capabilities dictionary
                truth = self.capabilities_truth[cap]
                if self.apiVersion != "factory.ai/v2":
                    truth.implemented = bool(val)
                if not val:
                    truth.live_verified = False

        return self


    def get_capability_truth(self, task: str) -> CapabilityTruthLevel:
        """Returns the 3-level capability truth state for the given task."""
        if task in self.capabilities_truth:
            return self.capabilities_truth[task]
        val = self.capabilities.get(task, False)
        return CapabilityTruthLevel(
            declared=val,
            implemented=val if self.apiVersion != "factory.ai/v2" else bool(val and self.validation and task in self.validation.implemented_capabilities),
            live_verified=False,
            evidence_run_id=None,
        )

    def is_variant_supported_for_task(self, task: str, variant: Optional[str] = None) -> bool:
        """Checks if the given variant is supported for the specified task."""
        norm_variant = str(variant or "").strip().lower()
        if not norm_variant:
            norm_variant = "standard"
        if self.apiVersion == "factory.ai/v2":
            return bool(self.capabilities.get(task, False)) and norm_variant in ("standard", "default")
        if self.task_variant_matrix and task in self.task_variant_matrix:
            return norm_variant in [v.lower() for v in self.task_variant_matrix[task]]
        if norm_variant in ("standard", "base", "default"):
            return True
        if hasattr(self.weights, "variants") and self.weights.variants:
            return norm_variant in self.weights.variants
        return False

