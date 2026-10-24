"""
factory_contracts/jobs/worker_contract.py
Worker daemon registration, leasing, heartbeats, progress, and completion contracts.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
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
    gpu_utilization_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    vram_used_mb: Optional[float] = Field(default=None, ge=0.0)


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


class JobCompletionRequest(BaseModel):
    """Payload sent by worker upon successful generation."""
    model_config = ConfigDict(extra="forbid")

    artifacts: List[ArtifactReference] = Field(..., description="Generated artifacts")
    inference_duration_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    fps: Optional[float] = None
    peak_vram_mb: Optional[float] = None


class JobFailureRequest(BaseModel):
    """Payload sent by worker upon unrecoverable error."""
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(..., description="Sanitized error description")
    retryable: bool = Field(default=True, description="Whether job may be retried")
