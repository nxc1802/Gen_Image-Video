"""
factory/apps/control-api/src/queue/repository/memory.py
Thread-safe and coroutine-safe in-memory job store for Factory v4 Lite.
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from factory_contracts.errors import FactoryError
from factory_contracts.jobs.job_contract import (
    AssetRecord,
    FactoryJob,
    JobProgress,
    JobState,
    WorkerLease,
)
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensures datetime is UTC timezone-aware; converts naive datetimes to UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class CapacityExceededError(FactoryError):
    def __init__(self, message: str = "Job store capacity reached; retry later."):
        super().__init__(
            message=message,
            code="QUEUE_FULL",
            status_code=429,
            details={"reason": "capacity_exceeded"},
        )


class InMemoryJobStore(BaseJobRepository):
    """
    Pure in-memory repository for Factory v4 Lite.
    - Thread-safe and coroutine-safe with asyncio.Lock.
    - Strictly capacity-bounded (default 1000 jobs).
    - Fast O(1) idempotency lookup.
    - Automatic TTL eviction for terminal jobs (SUCCEEDED, FAILED, CANCELLED).
    - Zero external database dependencies (no PostgreSQL, SQLite, or Alembic).
    """

    def __init__(
        self,
        max_capacity: Optional[int] = None,
        default_ttl_seconds: Optional[int] = None,
        db_path: Optional[str] = None,  # Backward compatibility for legacy tests
    ):
        cap = max_capacity or int(os.environ.get("FACTORY_MAX_JOBS", "1000"))
        ttl = default_ttl_seconds or int(os.environ.get("JOB_TTL_SECONDS", "7200"))
        self._capacity = max(1, cap)
        self._ttl_seconds = max(1, ttl)

        self._jobs: Dict[str, FactoryJob] = {}
        self._idempotency_map: Dict[str, str] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._event_counter: int = 0
        self._assets: Dict[str, AssetRecord] = {}
        self._assets_by_sha256: Dict[str, str] = {}
        self._workers: Dict[str, Dict[str, Any]] = {}

        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    async def initialize(self) -> None:
        """No-op for in-memory store."""
        pass

    async def close(self) -> None:
        """Clears all stored data."""
        async with self._lock:
            self._jobs.clear()
            self._idempotency_map.clear()
            self._events.clear()
            self._assets.clear()
            self._assets_by_sha256.clear()
            self._workers.clear()
            self._event_counter = 0

    def _evict_expired_or_overflow_locked(self) -> None:
        """
        Evicts terminal jobs older than TTL.
        If still at or above capacity, evicts oldest terminal jobs first.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._ttl_seconds)

        terminal_states = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}

        # 1. Evict expired terminal jobs
        expired_ids = [
            jid for jid, job in self._jobs.items()
            if job.state in terminal_states
            and (_ensure_utc(job.completed_at) or _ensure_utc(job.created_at)) < cutoff
        ]
        for jid in expired_ids:
            self._remove_job_locked(jid)

        # 2. If still at/over capacity, evict oldest terminal jobs regardless of TTL
        if len(self._jobs) >= self._capacity:
            terminal_jobs = [
                job for job in self._jobs.values()
                if job.state in terminal_states
            ]
            terminal_jobs.sort(key=lambda j: _ensure_utc(j.completed_at) or _ensure_utc(j.created_at))
            for job in terminal_jobs:
                self._remove_job_locked(job.job_id)
                if len(self._jobs) < self._capacity:
                    break

    def _remove_job_locked(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if job and job.idempotency_key:
            self._idempotency_map.pop(job.idempotency_key, None)
        self._events.pop(job_id, None)

    async def cleanup_expired_jobs(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Public API to actively sweep and evict expired terminal jobs.
        Returns the count of deleted jobs.
        """
        ttl = max_age_seconds if max_age_seconds is not None else self._ttl_seconds
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=max(0, ttl))
        terminal_states = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}

        async with self._lock:
            expired_ids = [
                jid for jid, job in self._jobs.items()
                if job.state in terminal_states
                and (_ensure_utc(job.completed_at) or _ensure_utc(job.created_at)) < cutoff
            ]
            for jid in expired_ids:
                self._remove_job_locked(jid)
            return len(expired_ids)

    async def create_job(self, job: FactoryJob) -> FactoryJob:
        async with self._lock:
            # Storage-level idempotency deduplication
            if job.idempotency_key and job.idempotency_key in self._idempotency_map:
                existing_id = self._idempotency_map[job.idempotency_key]
                if existing_id in self._jobs:
                    return self._jobs[existing_id].model_copy(deep=True)

            self._evict_expired_or_overflow_locked()
            if len(self._jobs) >= self._capacity:
                raise CapacityExceededError(
                    f"Job store capacity limit reached ({self._capacity} jobs)."
                )

            # Store deep copy or validated instance to prevent external mutation
            stored_job = job.model_copy(deep=True)
            self._jobs[stored_job.job_id] = stored_job
            if stored_job.idempotency_key:
                self._idempotency_map[stored_job.idempotency_key] = stored_job.job_id
            return stored_job.model_copy(deep=True)

    async def get_job(self, job_id: str) -> Optional[FactoryJob]:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def get_by_idempotency_key(self, key: str) -> Optional[FactoryJob]:
        async with self._lock:
            job_id = self._idempotency_map.get(key)
            if not job_id:
                return None
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def update_job(self, job: FactoryJob) -> FactoryJob:
        async with self._lock:
            if job.job_id in self._jobs:
                updated = job.model_copy(deep=True)
                self._jobs[job.job_id] = updated
                return updated.model_copy(deep=True)
            return job.model_copy(deep=True)

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        service: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[FactoryJob]:
        async with self._lock:
            jobs = list(self._jobs.values())
            if service:
                jobs = [j for j in jobs if j.service == service]
            if state:
                state_val = state.value if hasattr(state, "value") else str(state)
                jobs = [
                    j for j in jobs
                    if (j.state.value if hasattr(j.state, "value") else str(j.state)) == state_val
                ]
            jobs.sort(key=lambda j: _ensure_utc(j.created_at), reverse=True)
            sliced = jobs[offset:offset + limit]
            return [j.model_copy(deep=True) for j in sliced]

    async def find_eligible_jobs_for_worker(
        self,
        subsystem: str,
        supported_models: List[str],
        supported_tasks: List[str],
        limit: int = 1,
    ) -> List[FactoryJob]:
        async with self._lock:
            eligible = [
                j for j in self._jobs.values()
                if j.state == JobState.QUEUED
                and j.service == subsystem
                and j.model in supported_models
                and j.task in supported_tasks
            ]
            # Higher priority first, then older created_at first (FIFO within priority)
            eligible.sort(key=lambda j: (-j.priority, _ensure_utc(j.created_at)))
            return [j.model_copy(deep=True) for j in eligible[:limit]]

    async def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        expires_at: datetime,
        gpu_device: Optional[int] = None,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state not in (JobState.QUEUED, JobState.ADMITTED):
                return False

            now = datetime.now(timezone.utc)
            lease = WorkerLease(
                worker_id=worker_id,
                gpu_device=gpu_device,
                lease_token=lease_token,
                leased_at=now,
                lease_expires_at=expires_at,
            )
            job.state = JobState.RUNNING
            job.worker = lease
            if not job.started_at:
                job.started_at = now
            return True

    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        new_expires_at: datetime,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or not job.worker:
                return False
            if job.worker.worker_id != worker_id:
                return False
            job.worker.lease_expires_at = new_expires_at
            return True

    async def release_lease(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.worker = None
            return True

    async def find_expired_leases(self, cutoff: datetime) -> List[FactoryJob]:
        async with self._lock:
            expired = [
                j.model_copy(deep=True) for j in self._jobs.values()
                if j.state in (JobState.RUNNING, JobState.STREAMING)
                and j.worker is not None
                and j.worker.lease_expires_at < cutoff
            ]
            return expired

    async def record_event(
        self,
        job_id: str,
        event_type: str,
        data: Dict[str, Any],
        stage: Optional[str] = None,
        percent: Optional[float] = None,
    ) -> int:
        async with self._lock:
            self._event_counter += 1
            eid = self._event_counter
            now_iso = datetime.now(timezone.utc).isoformat()
            entry = {
                "event_id": eid,
                "job_id": job_id,
                "event_type": event_type,
                "stage": stage,
                "percent": percent,
                "data": data,
                "created_at": now_iso,
            }
            self._events[job_id].append(entry)
            return eid

    async def get_events_since(self, job_id: str, since_event_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            events = self._events.get(job_id, [])
            return [
                dict(e) for e in events
                if e["event_id"] > since_event_id
            ]

    async def create_asset(self, asset: AssetRecord) -> AssetRecord:
        async with self._lock:
            stored = asset.model_copy(deep=True)
            self._assets[stored.asset_id] = stored
            if stored.sha256:
                self._assets_by_sha256[stored.sha256] = stored.asset_id
            return stored.model_copy(deep=True)

    async def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        async with self._lock:
            asset = self._assets.get(asset_id)
            return asset.model_copy(deep=True) if asset else None

    async def get_asset_by_sha256(self, sha256: str) -> Optional[AssetRecord]:
        async with self._lock:
            asset_id = self._assets_by_sha256.get(sha256)
            if not asset_id:
                return None
            asset = self._assets.get(asset_id)
            return asset.model_copy(deep=True) if asset else None

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
        telemetry_status: Optional[str] = None,
        temperature_c: Optional[float] = None,
    ) -> None:
        async with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker["last_heartbeat"] = now_iso
                worker["current_job_id"] = active_job_id
                worker["status"] = "ONLINE"
                if gpu_utilization is not None:
                    worker["gpu_utilization"] = gpu_utilization
                if vram_used_mb is not None:
                    worker["vram_used_mb"] = vram_used_mb
                if temperature_c is not None:
                    worker["temperature_c"] = temperature_c
                if telemetry_status is not None:
                    worker["telemetry_status"] = telemetry_status
                if subsystem and subsystem != "unknown":
                    worker["subsystem"] = subsystem
                if framework and framework != "unknown":
                    worker["framework"] = framework
                if supported_models is not None:
                    worker["supported_models"] = list(supported_models)
                if supported_tasks is not None:
                    worker["supported_tasks"] = list(supported_tasks)
                if gpu_device is not None:
                    worker["gpu_device"] = gpu_device
            else:
                self._workers[worker_id] = {
                    "worker_id": worker_id,
                    "subsystem": subsystem or "unknown",
                    "framework": framework or "unknown",
                    "supported_models": list(supported_models or []),
                    "supported_tasks": list(supported_tasks or []),
                    "gpu_device": gpu_device,
                    "status": "ONLINE",
                    "current_job_id": active_job_id,
                    "last_heartbeat": now_iso,
                    "gpu_utilization": gpu_utilization,
                    "vram_used_mb": vram_used_mb,
                    "temperature_c": temperature_c,
                    "telemetry_status": telemetry_status or "UNAVAILABLE",
                    "registered_at": now_iso,
                }

    async def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return None
            return json.loads(json.dumps(worker))

    async def clear_worker_job(self, worker_id: str, job_id: Optional[str] = None) -> bool:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            if job_id is None or worker.get("current_job_id") == job_id:
                worker["current_job_id"] = None
                return True
            return False

    async def count_active_and_queued(self) -> Tuple[int, int]:
        async with self._lock:
            active = sum(
                1 for j in self._jobs.values()
                if j.state in (JobState.RUNNING, JobState.STREAMING, JobState.ADMITTED)
            )
            queued = sum(
                1 for j in self._jobs.values()
                if j.state == JobState.QUEUED
            )
            return active, queued
