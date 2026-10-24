"""
factory_contracts/jobs/job_contract.py
Factory Job Lifecycle, Leases, and Progress Contract
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from factory_contracts.errors.error_contract import ErrorDetail


class JobState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    STREAMING = "STREAMING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class JobProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(default="queued", description="Lifecycle stage: queued, loading, sampling, encoding, completed")
    step: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    preview_url: Optional[str] = None


class WorkerLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(..., description="Unique worker daemon identifier")
    gpu_device: Optional[int] = Field(default=None, description="Assigned GPU device index")
    lease_token: str = Field(default="", description="Cryptographic lease token held by worker")
    leased_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lease_expires_at: datetime = Field(..., description="Lease expiration threshold (300s TTL)")


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    artifact_id: str = Field(..., description="Unique artifact identifier, e.g. art-...")
    type: str = Field(default="media", description="Artifact type: media, audio, video, image")
    mime_type: str = Field(..., description="MIME type, e.g. 'image/png', 'video/mp4'")
    size_bytes: int = Field(..., ge=0)
    storage_path: str = Field(..., description="Internal S3/MinIO key")
    download_url: Optional[str] = Field(default=None, description="Presigned S3 GET URL (expiring in 24h)")
    url: Optional[str] = Field(default=None, description="Alias for download_url for E2E and client compatibility")
    sha256: str = Field(..., min_length=64, max_length=64, description="Hex-encoded SHA256 checksum")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def sync_url_fields(self) -> "ArtifactReference":
        if not self.url and self.download_url:
            self.url = self.download_url
        elif not self.download_url and self.url:
            self.download_url = self.url
        elif not self.url and not self.download_url:
            raise ValueError("Either 'url' or 'download_url' must be provided.")
        return self


class JobTelemetry(BaseModel):
    model_config = ConfigDict(extra="allow")

    queue_wait_ms: Optional[float] = None
    inference_duration_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    fps: Optional[float] = None
    peak_vram_mb: Optional[float] = None


class JobSubmissionRequest(BaseModel):
    """Client request payload for POST /v1/jobs."""
    model_config = ConfigDict(extra="forbid")

    service: str = Field(..., description="Target service subsystem, e.g. 'image'")
    model: str = Field(..., description="Model family ID, e.g. 'qwen-image-2.1'")
    task: str = Field(..., description="Capability task name, e.g. 'inpaint' or 'text_to_image'")
    inputs: Dict[str, Any] = Field(..., description="Pre-validated task input dictionary")
    priority: int = Field(default=10, ge=1, le=100, description="1=low, 10=standard, 100=critical")
    idempotency_key: Optional[str] = Field(default=None, max_length=128)


class FactoryJob(BaseModel):
    """Authoritative Factory Job entity tracked in PostgreSQL and Redis."""
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., pattern=r"^job-[0-9a-f-]+$", description="Job UUID v4")
    request_id: str = Field(..., pattern=r"^req-[0-9a-f-]+$", description="Originating request UUID v4")
    idempotency_key: Optional[str] = None
    service: str
    model: str
    task: str
    inputs: Dict[str, Any]
    priority: int = 10
    state: JobState = JobState.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: JobProgress = Field(default_factory=JobProgress)
    worker: Optional[WorkerLease] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    error: Optional[ErrorDetail] = None
    artifacts: List[ArtifactReference] = Field(default_factory=list)
    telemetry: JobTelemetry = Field(default_factory=JobTelemetry)
