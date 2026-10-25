"""
factory/apps/control-api/src/queue/repository/sqlite.py
SQLite implementation of BaseJobRepository using aiosqlite for local dev & testing.
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
from pydantic import ValidationError

from factory_contracts.errors.error_contract import ErrorDetail
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobTelemetry,
    WorkerLease,
)
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    service TEXT NOT NULL,
    model TEXT NOT NULL,
    task TEXT NOT NULL,
    inputs TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 10,
    state TEXT NOT NULL DEFAULT 'CREATED',
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    worker_id TEXT,
    worker_lease TEXT,
    progress TEXT NOT NULL,
    error TEXT,
    artifacts TEXT NOT NULL DEFAULT '[]',
    telemetry TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_priority ON jobs (state, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_service_model_task ON jobs (service, model, task);
CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs (idempotency_key);

CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    stage TEXT,
    percent REAL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events (job_id, event_id ASC);

CREATE TABLE IF NOT EXISTS worker_registrations (
    worker_id TEXT PRIMARY KEY,
    subsystem TEXT NOT NULL,
    framework TEXT NOT NULL,
    supported_models TEXT NOT NULL,
    supported_tasks TEXT NOT NULL,
    gpu_device INTEGER,
    status TEXT NOT NULL DEFAULT 'ONLINE',
    current_job_id TEXT,
    lease_expires_at TEXT,
    last_heartbeat TEXT NOT NULL,
    gpu_utilization REAL,
    vram_used_mb REAL,
    registered_at TEXT NOT NULL
);
"""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return None


class SqliteJobRepository(BaseJobRepository):
    """Zero-dependency local and in-memory SQLite repository."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SQLITE_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _row_to_job(self, row: aiosqlite.Row) -> FactoryJob:
        d = dict(row)
        inputs = json.loads(d["inputs"]) if d["inputs"] else {}
        worker_lease = WorkerLease.model_validate(json.loads(d["worker_lease"])) if d["worker_lease"] else None
        progress = JobProgress.model_validate(json.loads(d["progress"])) if d["progress"] else JobProgress()
        error = ErrorDetail.model_validate(json.loads(d["error"])) if d["error"] else None
        artifacts_raw = json.loads(d["artifacts"]) if d["artifacts"] else []
        artifacts = [ArtifactReference.model_validate(a) for a in artifacts_raw]
        telemetry = JobTelemetry.model_validate(json.loads(d["telemetry"])) if d["telemetry"] else JobTelemetry()

        return FactoryJob(
            job_id=d["job_id"],
            request_id=d["request_id"],
            idempotency_key=d["idempotency_key"],
            service=d["service"],
            model=d["model"],
            task=d["task"],
            inputs=inputs,
            priority=d["priority"],
            state=JobState(d["state"]),
            retry_count=d["retry_count"],
            max_retries=d["max_retries"],
            worker=worker_lease,
            progress=progress,
            error=error,
            artifacts=artifacts,
            telemetry=telemetry,
            created_at=_parse_dt(d["created_at"]) or datetime.now(timezone.utc),
            started_at=_parse_dt(d["started_at"]),
            completed_at=_parse_dt(d["completed_at"]),
        )

    async def create_job(self, job: FactoryJob) -> FactoryJob:
        async with self._lock:
            now = _iso_now()
            inputs_json = json.dumps(job.inputs)
            progress_json = json.dumps(job.progress.model_dump(mode="json"))
            artifacts_json = json.dumps([a.model_dump(mode="json") for a in job.artifacts])
            telemetry_json = json.dumps(job.telemetry.model_dump(mode="json"))
            worker_lease_json = json.dumps(job.worker.model_dump(mode="json")) if job.worker else None
            error_json = json.dumps(job.error.model_dump(mode="json")) if job.error else None

            await self._db.execute(
                """
                INSERT INTO jobs (
                    job_id, request_id, idempotency_key, service, model, task,
                    inputs, priority, state, retry_count, max_retries,
                    worker_id, worker_lease, progress, error, artifacts,
                    telemetry, created_at, started_at, completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.request_id,
                    job.idempotency_key,
                    job.service,
                    job.model,
                    job.task,
                    inputs_json,
                    job.priority,
                    job.state.value,
                    job.retry_count,
                    job.max_retries,
                    job.worker.worker_id if job.worker else None,
                    worker_lease_json,
                    progress_json,
                    error_json,
                    artifacts_json,
                    telemetry_json,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    now,
                ),
            )
            await self._db.commit()
            return job

    async def get_job(self, job_id: str) -> Optional[FactoryJob]:
        async with self._lock:
            async with self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_job(row)
                return None

    async def get_by_idempotency_key(self, key: str) -> Optional[FactoryJob]:
        async with self._lock:
            async with self._db.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_job(row)
                return None

    async def update_job(self, job: FactoryJob) -> FactoryJob:
        async with self._lock:
            now = _iso_now()
            progress_json = json.dumps(job.progress.model_dump(mode="json"))
            artifacts_json = json.dumps([a.model_dump(mode="json") for a in job.artifacts])
            telemetry_json = json.dumps(job.telemetry.model_dump(mode="json"))
            worker_lease_json = json.dumps(job.worker.model_dump(mode="json")) if job.worker else None
            error_json = json.dumps(job.error.model_dump(mode="json")) if job.error else None

            await self._db.execute(
                """
                UPDATE jobs SET
                    state = ?,
                    priority = ?,
                    retry_count = ?,
                    worker_id = ?,
                    worker_lease = ?,
                    progress = ?,
                    error = ?,
                    artifacts = ?,
                    telemetry = ?,
                    started_at = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    job.state.value,
                    job.priority,
                    job.retry_count,
                    job.worker.worker_id if job.worker else None,
                    worker_lease_json,
                    progress_json,
                    error_json,
                    artifacts_json,
                    telemetry_json,
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    now,
                    job.job_id,
                ),
            )
            await self._db.commit()
            return job

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        service: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[FactoryJob]:
        async with self._lock:
            conditions = []
            params: List[Any] = []
            if service:
                conditions.append("service = ?")
                params.append(service)
            if state:
                state_val = state.value if hasattr(state, "value") else str(state)
                conditions.append("state = ?")
                params.append(state_val)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"""
                SELECT * FROM jobs
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            async with self._db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_job(row) for row in rows]

    async def find_eligible_jobs_for_worker(
        self,
        subsystem: str,
        supported_models: List[str],
        supported_tasks: List[str],
        limit: int = 1,
    ) -> List[FactoryJob]:
        async with self._lock:
            sql_limit = max(limit * 5, 100)
            async with self._db.execute(
                """
                SELECT * FROM jobs
                WHERE state = 'QUEUED' AND service = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (subsystem, sql_limit),
            ) as cursor:
                rows = await cursor.fetchall()
                eligible = []
                for row in rows:
                    job = self._row_to_job(row)
                    if job.model in supported_models and job.task in supported_tasks:
                        eligible.append(job)
                        if len(eligible) >= limit:
                            break
                return eligible

    async def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        expires_at: datetime,
        gpu_device: Optional[int] = None,
    ) -> bool:
        async with self._lock:
            lease = WorkerLease(
                worker_id=worker_id,
                gpu_device=gpu_device,
                lease_token=lease_token,
                leased_at=datetime.now(timezone.utc),
                lease_expires_at=expires_at,
            )
            lease_json = json.dumps(lease.model_dump(mode="json"))
            started_at = _iso_now()

            cursor = await self._db.execute(
                """
                UPDATE jobs SET
                    state = 'RUNNING',
                    worker_id = ?,
                    worker_lease = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE job_id = ? AND state IN ('QUEUED', 'ADMITTED')
                """,
                (worker_id, lease_json, started_at, started_at, job_id),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        new_expires_at: datetime,
    ) -> bool:
        async with self._lock:
            async with self._db.execute("SELECT worker_lease FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
                row = await cursor.fetchone()
                if not row or not row["worker_lease"]:
                    return False

                lease_data = json.loads(row["worker_lease"])
                lease_data["lease_expires_at"] = new_expires_at.isoformat()
                now = _iso_now()

                c = await self._db.execute(
                    "UPDATE jobs SET worker_lease = ?, updated_at = ? WHERE job_id = ? AND worker_id = ?",
                    (json.dumps(lease_data), now, job_id, worker_id),
                )
                await self._db.commit()
                return c.rowcount > 0

    async def release_lease(self, job_id: str) -> bool:
        async with self._lock:
            now = _iso_now()
            cursor = await self._db.execute(
                "UPDATE jobs SET worker_id = NULL, worker_lease = NULL, updated_at = ? WHERE job_id = ?",
                (now, job_id),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def find_expired_leases(self, cutoff: datetime) -> List[FactoryJob]:
        async with self._lock:
            async with self._db.execute(
                "SELECT * FROM jobs WHERE state IN ('RUNNING', 'STREAMING') AND worker_lease IS NOT NULL"
            ) as cursor:
                rows = await cursor.fetchall()
                expired = []
                for row in rows:
                    job = self._row_to_job(row)
                    if job.worker and job.worker.lease_expires_at < cutoff:
                        expired.append(job)
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
            now = _iso_now()
            cursor = await self._db.execute(
                """
                INSERT INTO job_events (job_id, event_type, stage, percent, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, event_type, stage, percent, json.dumps(data), now),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def get_events_since(self, job_id: str, since_event_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            async with self._db.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND event_id > ? ORDER BY event_id ASC",
                (job_id, since_event_id),
            ) as cursor:
                rows = await cursor.fetchall()
                events = []
                for row in rows:
                    d = dict(row)
                    events.append({
                        "event_id": d["event_id"],
                        "job_id": d["job_id"],
                        "event_type": d["event_type"],
                        "stage": d["stage"],
                        "percent": d["percent"],
                        "data": json.loads(d["data"]),
                        "created_at": d["created_at"],
                    })
                return events

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
        async with self._lock:
            now = _iso_now()
            async with self._db.execute("SELECT worker_id FROM worker_registrations WHERE worker_id = ?", (worker_id,)) as cursor:
                row = await cursor.fetchone()

            if row:
                await self._db.execute(
                    """
                    UPDATE worker_registrations SET
                        current_job_id = ?,
                        last_heartbeat = ?,
                        gpu_utilization = COALESCE(?, gpu_utilization),
                        vram_used_mb = COALESCE(?, vram_used_mb),
                        subsystem = COALESCE(?, subsystem),
                        framework = COALESCE(?, framework),
                        supported_models = COALESCE(?, supported_models),
                        supported_tasks = COALESCE(?, supported_tasks),
                        gpu_device = COALESCE(?, gpu_device),
                        status = 'ONLINE'
                    WHERE worker_id = ?
                    """,
                    (
                        active_job_id,
                        now,
                        gpu_utilization,
                        vram_used_mb,
                        subsystem if subsystem and subsystem != "unknown" else None,
                        framework if framework and framework != "unknown" else None,
                        json.dumps(supported_models) if supported_models is not None else None,
                        json.dumps(supported_tasks) if supported_tasks is not None else None,
                        gpu_device,
                        worker_id,
                    ),
                )
            else:
                await self._db.execute(
                    """
                    INSERT INTO worker_registrations (
                        worker_id, subsystem, framework, supported_models,
                        supported_tasks, gpu_device, status, current_job_id,
                        last_heartbeat, gpu_utilization, vram_used_mb, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ONLINE', ?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        subsystem or "unknown",
                        framework or "unknown",
                        json.dumps(supported_models or []),
                        json.dumps(supported_tasks or []),
                        gpu_device,
                        active_job_id,
                        now,
                        gpu_utilization,
                        vram_used_mb,
                        now,
                    ),
                )
            await self._db.commit()

    async def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            async with self._db.execute("SELECT * FROM worker_registrations WHERE worker_id = ?", (worker_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    d = dict(row)
                    d["supported_models"] = json.loads(d["supported_models"])
                    d["supported_tasks"] = json.loads(d["supported_tasks"])
                    return d
                return None

    async def count_active_and_queued(self) -> Tuple[int, int]:
        async with self._lock:
            async with self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('RUNNING', 'STREAMING', 'ADMITTED')"
            ) as c1:
                active = (await c1.fetchone())[0]

            async with self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state = 'QUEUED'"
            ) as c2:
                queued = (await c2.fetchone())[0]

            return active, queued
