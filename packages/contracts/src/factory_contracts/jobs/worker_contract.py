"""
factory_contracts/jobs/worker_contract.py
Worker daemon registration, leasing, heartbeats, progress, and completion contracts.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from factory_contracts.tasks import normalize_model, normalize_task
from factory_contracts.jobs.job_contract import FactoryJob, ArtifactReference
from factory_contracts.errors.error_contract import ErrorDetail


class WorkerLeaseRequest(BaseModel):
    """Payload sent by worker daemon to request the next matching job."""
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(..., description="Unique worker daemon identifier")
    subsystem: str = Field(..., description="Subsystem type, e.g. 'image', 'video', 'text'")
    supported_models: List[str] = Field(..., description="List of model IDs worker can execute")
    supported_tasks: List[str] = Field(..., description="List of task names worker can execute")
    gpu_device: Optional[int] = Field(default=None, description="Physical GPU device index")
    lease_duration_seconds: int = Field(default=300, ge=30, le=1800)

    @model_validator(mode="after")
    def normalize_registration(self):
        self.supported_tasks = list(dict.fromkeys(normalize_task(task, engine=self.subsystem) for task in self.supported_tasks))
        self.supported_models = list(dict.fromkeys(normalize_model(mid) for mid in self.supported_models))
        return self


class WorkerLeaseResponse(BaseModel):
    """Response returned to worker on lease attempt."""
    model_config = ConfigDict(extra="forbid")

    job: Optional[FactoryJob] = Field(default=None, description="Leased job, or null if queue empty")
    lease_token: Optional[str] = Field(default=None, description="Cryptographic lease token held by worker")
    lease_expires_at: Optional[datetime] = Field(default=None, description="Lease expiration timestamp")


class WorkerHeartbeatRequest(BaseModel):
    """Periodic heartbeat sent by worker daemon every 30s."""
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(..., description="Unique worker daemon identifier")
    active_job_id: Optional[str] = Field(default=None, description="Active job ID currently being processed")
    lease_token: Optional[str] = Field(default=None, description="Active lease token")
    telemetry_status: str = Field(default="UNAVAILABLE", description="'AVAILABLE' or 'UNAVAILABLE'")
    gpu_utilization_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    vram_used_mb: Optional[float] = Field(default=None, ge=0.0)
    vram_total_mb: Optional[float] = Field(default=None, ge=0.0)
    system_vram_used_mb: Optional[float] = Field(default=None, ge=0.0)
    process_vram_used_mb: Optional[float] = Field(default=None, ge=0.0)
    gpu_name: Optional[str] = Field(default=None, description="Name of GPU device")
    temperature_c: Optional[float] = Field(default=None, description="GPU temperature in Celsius")


class WorkerHeartbeatResponse(BaseModel):
    """Heartbeat response instructing worker on status and cancel signals."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="acknowledged")
    cancel_requested: bool = Field(default=False, description="True if active job was cancelled")
    lease_extended_until: Optional[datetime] = Field(default=None)


class ProgressUpdateRequest(BaseModel):
    """Progress reported by worker daemon during execution."""
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(..., description="e.g. 'loading', 'sampling', 'encoding'")
    step: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    preview_url: Optional[str] = None
    worker_id: Optional[str] = Field(default=None, description="Worker daemon identifier")
    lease_token: Optional[str] = Field(default=None, description="Active lease token")


class JobCompletionRequest(BaseModel):
    """Payload sent by worker upon successful generation."""
    model_config = ConfigDict(extra="forbid")

    artifacts: List[ArtifactReference] = Field(default_factory=list, description="Generated artifacts")
    result: Optional[Any] = Field(default=None, description="In-memory inference result payload")
    artifact_path: Optional[str] = Field(default=None, description="Scratch disk path to artifact")
    inference_duration_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    fps: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    worker_id: Optional[str] = Field(default=None, description="Worker daemon identifier")
    lease_token: Optional[str] = Field(default=None, description="Active lease token")


class JobFailureRequest(BaseModel):
    """Payload sent by worker upon unrecoverable error."""
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(..., description="Sanitized error description")
    retryable: bool = Field(default=True, description="Whether job may be retried")
    worker_id: Optional[str] = Field(default=None, description="Worker daemon identifier")
    lease_token: Optional[str] = Field(default=None, description="Active lease token")
