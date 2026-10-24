"""
factory/apps/control-api/src/queue/repository/base.py
Abstract base class for Job and Worker persistence.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from factory_contracts.jobs.job_contract import FactoryJob, JobState


class BaseJobRepository(ABC):
    """Abstract Repository interface for jobs, worker registrations, and events."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initializes tables, indexes, and connection pool."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Closes connections."""
        ...

    @abstractmethod
    async def create_job(self, job: FactoryJob) -> FactoryJob:
        """Persists a new job record."""
        ...

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[FactoryJob]:
        """Retrieves a job by its unique ID."""
        ...

    @abstractmethod
    async def get_by_idempotency_key(self, key: str) -> Optional[FactoryJob]:
        """Retrieves a job by idempotency key if present."""
        ...

    @abstractmethod
    async def update_job(self, job: FactoryJob) -> FactoryJob:
        """Updates job record in storage."""
        ...

    @abstractmethod
    async def find_eligible_jobs_for_worker(
        self,
        subsystem: str,
        supported_models: List[str],
        supported_tasks: List[str],
        limit: int = 1,
    ) -> List[FactoryJob]:
        """Finds eligible QUEUED jobs ordered by priority DESC, created_at ASC."""
        ...

    @abstractmethod
    async def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        expires_at: datetime,
        gpu_device: Optional[int] = None,
    ) -> bool:
        """Atomically transitions job to RUNNING and stamps worker_lease."""
        ...

    @abstractmethod
    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        new_expires_at: datetime,
    ) -> bool:
        """Extends worker lease expiration."""
        ...

    @abstractmethod
    async def release_lease(self, job_id: str) -> bool:
        """Clears worker lease from job."""
        ...

    @abstractmethod
    async def find_expired_leases(self, cutoff: datetime) -> List[FactoryJob]:
        """Finds RUNNING or STREAMING jobs where lease_expires_at < cutoff."""
        ...

    @abstractmethod
    async def record_event(
        self,
        job_id: str,
        event_type: str,
        data: Dict[str, Any],
        stage: Optional[str] = None,
        percent: Optional[float] = None,
    ) -> int:
        """Records historical event for SSE reconnection."""
        ...

    @abstractmethod
    async def get_events_since(self, job_id: str, since_event_id: int) -> List[Dict[str, Any]]:
        """Retrieves historical events with event_id > since_event_id."""
        ...

    @abstractmethod
    async def upsert_worker_heartbeat(
        self,
        worker_id: str,
        subsystem: Optional[str] = None,
        framework: Optional[str] = None,
        supported_models: Optional[List[str]] = None,
        supported_tasks: Optional[List[str]] = None,
        gpu_device: Optional[int] = None,
        active_job_id: Optional[str] = None,
        gpu_utilization: Optional[float] = None,
        vram_used_mb: Optional[float] = None,
    ) -> None:
        """Updates worker liveness and status."""
        ...

    @abstractmethod
    async def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves worker registration metadata."""
        ...

    @abstractmethod
    async def count_active_and_queued(self) -> Tuple[int, int]:
        """Returns (active_jobs_count, queued_jobs_count)."""
        ...
