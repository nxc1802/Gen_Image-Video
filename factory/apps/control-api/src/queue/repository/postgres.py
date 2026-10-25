"""
factory/apps/control-api/src/queue/repository/postgres.py
PostgreSQL implementation of BaseJobRepository using asyncpg.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

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

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    idempotency_key VARCHAR(128) UNIQUE,
    service VARCHAR(32) NOT NULL,
    model VARCHAR(64) NOT NULL,
    task VARCHAR(64) NOT NULL,
    inputs JSONB NOT NULL,
    priority INT NOT NULL DEFAULT 10,
    state VARCHAR(20) NOT NULL DEFAULT 'CREATED',
    retry_count INT NOT NULL DEFAULT 0,
    max_retries INT NOT NULL DEFAULT 3,
    worker_id VARCHAR(64),
    worker_lease JSONB,
    progress JSONB NOT NULL,
    error JSONB,
    artifacts JSONB NOT NULL DEFAULT '[]',
    telemetry JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_priority ON jobs (state, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_service_model_task ON jobs (service, model, task);
CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS job_events (
    event_id BIGSERIAL PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    event_type VARCHAR(32) NOT NULL,
    stage VARCHAR(32),
    percent FLOAT,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events (job_id, event_id ASC);

CREATE TABLE IF NOT EXISTS worker_registrations (
    worker_id VARCHAR(64) PRIMARY KEY,
    subsystem VARCHAR(32) NOT NULL,
    framework VARCHAR(32) NOT NULL,
    supported_models JSONB NOT NULL,
    supported_tasks JSONB NOT NULL,
    gpu_device INT,
    status VARCHAR(20) NOT NULL DEFAULT 'ONLINE',
    current_job_id VARCHAR(64) REFERENCES jobs(job_id) ON DELETE SET NULL,
    lease_expires_at TIMESTAMPTZ,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    gpu_utilization FLOAT,
    vram_used_mb FLOAT,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class PostgresJobRepository(BaseJobRepository):
    """PostgreSQL production repository implementation."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=20)
        async with self.pool.acquire() as conn:
            await conn.execute(POSTGRES_SCHEMA)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    def _record_to_job(self, record: asyncpg.Record) -> FactoryJob:
        d = dict(record)
        inputs = json.loads(d["inputs"]) if isinstance(d["inputs"], str) else d["inputs"]
        worker_lease = WorkerLease.model_validate(json.loads(d["worker_lease"]) if isinstance(d["worker_lease"], str) else d["worker_lease"]) if d["worker_lease"] else None
        progress = JobProgress.model_validate(json.loads(d["progress"]) if isinstance(d["progress"], str) else d["progress"]) if d["progress"] else JobProgress()
        error = ErrorDetail.model_validate(json.loads(d["error"]) if isinstance(d["error"], str) else d["error"]) if d["error"] else None
        raw_artifacts = json.loads(d["artifacts"]) if isinstance(d["artifacts"], str) else d["artifacts"]
        artifacts = [ArtifactReference.model_validate(a) for a in raw_artifacts]
        telemetry = JobTelemetry.model_validate(json.loads(d["telemetry"]) if isinstance(d["telemetry"], str) else d["telemetry"]) if d["telemetry"] else JobTelemetry()

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
            created_at=d["created_at"],
            started_at=d["started_at"],
            completed_at=d["completed_at"],
        )

    async def create_job(self, job: FactoryJob) -> FactoryJob:
        query = """
        INSERT INTO jobs (
            job_id, request_id, idempotency_key, service, model, task,
            inputs, priority, state, retry_count, max_retries,
            worker_id, worker_lease, progress, error, artifacts,
            telemetry, created_at, started_at, completed_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13::jsonb, $14::jsonb, $15::jsonb, $16::jsonb, $17::jsonb, $18, $19, $20, NOW())
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                job.job_id,
                job.request_id,
                job.idempotency_key,
                job.service,
                job.model,
                job.task,
                json.dumps(job.inputs),
                job.priority,
                job.state.value,
                job.retry_count,
                job.max_retries,
                job.worker.worker_id if job.worker else None,
                json.dumps(job.worker.model_dump(mode="json")) if job.worker else None,
                json.dumps(job.progress.model_dump(mode="json")),
                json.dumps(job.error.model_dump(mode="json")) if job.error else None,
                json.dumps([a.model_dump(mode="json") for a in job.artifacts]),
                json.dumps(job.telemetry.model_dump(mode="json")),
                job.created_at,
                job.started_at,
                job.completed_at,
            )
        return job

    async def get_job(self, job_id: str) -> Optional[FactoryJob]:
        async with self.pool.acquire() as conn:
            rec = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
            if rec:
                return self._record_to_job(rec)
            return None

    async def get_by_idempotency_key(self, key: str) -> Optional[FactoryJob]:
        async with self.pool.acquire() as conn:
            rec = await conn.fetchrow("SELECT * FROM jobs WHERE idempotency_key = $1", key)
            if rec:
                return self._record_to_job(rec)
            return None

    async def update_job(self, job: FactoryJob) -> FactoryJob:
        query = """
        UPDATE jobs SET
            state = $1,
            priority = $2,
            retry_count = $3,
            worker_id = $4,
            worker_lease = $5::jsonb,
            progress = $6::jsonb,
            error = $7::jsonb,
            artifacts = $8::jsonb,
            telemetry = $9::jsonb,
            started_at = $10,
            completed_at = $11,
            updated_at = NOW()
        WHERE job_id = $12
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                job.state.value,
                job.priority,
                job.retry_count,
                job.worker.worker_id if job.worker else None,
                json.dumps(job.worker.model_dump(mode="json")) if job.worker else None,
                json.dumps(job.progress.model_dump(mode="json")),
                json.dumps(job.error.model_dump(mode="json")) if job.error else None,
                json.dumps([a.model_dump(mode="json") for a in job.artifacts]),
                json.dumps(job.telemetry.model_dump(mode="json")),
                job.started_at,
                job.completed_at,
                job.job_id,
            )
        return job

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        service: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[FactoryJob]:
        conditions = []
        params: List[Any] = []
        idx = 1
        if service:
            conditions.append(f"service = ${idx}")
            params.append(service)
            idx += 1
        if state:
            state_val = state.value if hasattr(state, "value") else str(state)
            conditions.append(f"state = ${idx}")
            params.append(state_val)
            idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])
        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)
            return [self._record_to_job(rec) for rec in records]

    async def find_eligible_jobs_for_worker(
        self,
        subsystem: str,
        supported_models: List[str],
        supported_tasks: List[str],
        limit: int = 1,
    ) -> List[FactoryJob]:
        query = """
        SELECT * FROM jobs
        WHERE state = 'QUEUED'
          AND service = $1
          AND model = ANY($2::varchar[])
          AND task = ANY($3::varchar[])
        ORDER BY priority DESC, created_at ASC
        LIMIT $4
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, subsystem, supported_models, supported_tasks, limit)
            return [self._record_to_job(r) for r in records]

    async def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        expires_at: datetime,
        gpu_device: Optional[int] = None,
    ) -> bool:
        lease = WorkerLease(
            worker_id=worker_id,
            gpu_device=gpu_device,
            lease_token=lease_token,
            leased_at=datetime.now(timezone.utc),
            lease_expires_at=expires_at,
        )
        query = """
        UPDATE jobs SET
            state = 'RUNNING',
            worker_id = $1,
            worker_lease = $2::jsonb,
            started_at = COALESCE(started_at, NOW()),
            updated_at = NOW()
        WHERE job_id = $3 AND state IN ('QUEUED', 'ADMITTED')
        """
        async with self.pool.acquire() as conn:
            res = await conn.execute(query, worker_id, json.dumps(lease.model_dump(mode="json")), job_id)
            return res == "UPDATE 1"

    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        new_expires_at: datetime,
    ) -> bool:
        query = """
        UPDATE jobs SET
            worker_lease = jsonb_set(worker_lease, '{lease_expires_at}', to_jsonb($1::text)),
            updated_at = NOW()
        WHERE job_id = $2 AND worker_id = $3
        """
        async with self.pool.acquire() as conn:
            res = await conn.execute(query, new_expires_at.isoformat(), job_id, worker_id)
            return res == "UPDATE 1"

    async def release_lease(self, job_id: str) -> bool:
        query = """
        UPDATE jobs SET
            worker_id = NULL,
            worker_lease = NULL,
            updated_at = NOW()
        WHERE job_id = $1
        """
        async with self.pool.acquire() as conn:
            res = await conn.execute(query, job_id)
            return res == "UPDATE 1"

    async def find_expired_leases(self, cutoff: datetime) -> List[FactoryJob]:
        query = """
        SELECT * FROM jobs
        WHERE state IN ('RUNNING', 'STREAMING')
          AND (worker_lease->>'lease_expires_at')::timestamptz < $1
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, cutoff)
            return [self._record_to_job(r) for r in records]

    async def record_event(
        self,
        job_id: str,
        event_type: str,
        data: Dict[str, Any],
        stage: Optional[str] = None,
        percent: Optional[float] = None,
    ) -> int:
        query = """
        INSERT INTO job_events (job_id, event_type, stage, percent, data, created_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
        RETURNING event_id
        """
        async with self.pool.acquire() as conn:
            val = await conn.fetchval(query, job_id, event_type, stage, percent, json.dumps(data))
            return int(val)

    async def get_events_since(self, job_id: str, since_event_id: int) -> List[Dict[str, Any]]:
        query = """
        SELECT event_id, job_id, event_type, stage, percent, data, created_at
        FROM job_events
        WHERE job_id = $1 AND event_id > $2
        ORDER BY event_id ASC
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, job_id, since_event_id)
            return [
                {
                    "event_id": r["event_id"],
                    "job_id": r["job_id"],
                    "event_type": r["event_type"],
                    "stage": r["stage"],
                    "percent": r["percent"],
                    "data": json.loads(r["data"]) if isinstance(r["data"], str) else r["data"],
                    "created_at": r["created_at"].isoformat(),
                }
                for r in records
            ]

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
        query = """
        INSERT INTO worker_registrations (
            worker_id, subsystem, framework, supported_models,
            supported_tasks, gpu_device, status, current_job_id,
            last_heartbeat, gpu_utilization, vram_used_mb, registered_at
        ) VALUES (
            $1, $2, $3, $4::jsonb, $5::jsonb, $6, 'ONLINE', $7, NOW(), $8, $9, NOW()
        )
        ON CONFLICT (worker_id) DO UPDATE SET
            current_job_id = $7,
            last_heartbeat = NOW(),
            gpu_utilization = COALESCE($8, worker_registrations.gpu_utilization),
            vram_used_mb = COALESCE($9, worker_registrations.vram_used_mb),
            subsystem = CASE WHEN $2 IS NOT NULL AND $2 <> 'unknown' THEN $2 ELSE worker_registrations.subsystem END,
            framework = CASE WHEN $3 IS NOT NULL AND $3 <> 'unknown' THEN $3 ELSE worker_registrations.framework END,
            supported_models = CASE WHEN $4 IS NOT NULL AND $4 <> '[]' THEN $4::jsonb ELSE worker_registrations.supported_models END,
            supported_tasks = CASE WHEN $5 IS NOT NULL AND $5 <> '[]' THEN $5::jsonb ELSE worker_registrations.supported_tasks END,
            gpu_device = COALESCE($6, worker_registrations.gpu_device),
            status = 'ONLINE'
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                worker_id,
                subsystem or "unknown",
                framework or "unknown",
                json.dumps(supported_models or []),
                json.dumps(supported_tasks or []),
                gpu_device,
                active_job_id,
                gpu_utilization,
                vram_used_mb,
            )

    async def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rec = await conn.fetchrow("SELECT * FROM worker_registrations WHERE worker_id = $1", worker_id)
            if rec:
                d = dict(rec)
                d["supported_models"] = json.loads(d["supported_models"]) if isinstance(d["supported_models"], str) else d["supported_models"]
                d["supported_tasks"] = json.loads(d["supported_tasks"]) if isinstance(d["supported_tasks"], str) else d["supported_tasks"]
                return d
            return None

    async def count_active_and_queued(self) -> Tuple[int, int]:
        async with self.pool.acquire() as conn:
            active = await conn.fetchval(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('RUNNING', 'STREAMING', 'ADMITTED')"
            )
            queued = await conn.fetchval(
                "SELECT COUNT(*) FROM jobs WHERE state = 'QUEUED'"
            )
            return int(active), int(queued)
