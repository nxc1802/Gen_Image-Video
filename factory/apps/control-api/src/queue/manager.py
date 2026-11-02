"""
factory/apps/control-api/src/queue/manager.py
Central JobLifecycleManager coordinating job submission, states, retries, and events.
"""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from factory_contracts.errors import (
    ErrorCode,
    ErrorDetail,
    FactoryError,
    InvalidRequestError,
    NotFoundError,
    UnauthorizedError,
)
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    JobTelemetry,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
)
from factory_contracts.tasks import normalize_model, normalize_task
from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.queue.state_machine import JobStateMachine
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.metrics import metrics


class JobLifecycleManager:
    """Orchestrates job lifecycle across persistence repository, broker, and event streams."""

    def __init__(
        self,
        repository: BaseJobRepository,
        broker: BaseQueueBroker,
        stream_manager: StreamManager,
    ):
        self.repository = repository
        self.broker = broker
        self.stream_manager = stream_manager
        self._last_metrics_update: float = 0.0

    def _schedule_telemetry_update(self) -> None:
        """Throttles and schedules asynchronous background update of queue/active metrics."""
        now = time.monotonic()
        if now - self._last_metrics_update < 1.0:
            return
        self._last_metrics_update = now
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._update_telemetry_async())
        except RuntimeError:
            pass

    async def _update_telemetry_async(self) -> None:
        try:
            active, queued = await self.repository.count_active_and_queued()
            metrics.set_active_jobs(active)
            metrics.set_queue_depth(queued)
        except Exception:
            pass

    async def submit_job(
        self,
        req: JobSubmissionRequest,
        request_id: str,
        trace_id: Optional[str] = None,
    ) -> Tuple[FactoryJob, bool]:
        """
        Submits a new job.
        Returns (job, created):
        - If idempotency_key matches existing job, returns (existing_job, False).
        - If newly created, returns (new_job, True).
        """
        if req.idempotency_key:
            existing = await self.repository.get_by_idempotency_key(req.idempotency_key)
            if existing:
                return existing, False

        import re
        valid_req_id = request_id
        if not valid_req_id or not re.match(r"^req-[0-9a-f-]+$", valid_req_id):
            valid_req_id = f"req-{uuid.uuid4()}"

        job_id = f"job-{uuid.uuid4()}"
        now = datetime.now(timezone.utc)
        job = FactoryJob(
            job_id=job_id,
            request_id=valid_req_id,
            trace_id=trace_id,
            idempotency_key=req.idempotency_key,
            service=req.service,
            model=normalize_model(req.model),
            task=normalize_task(req.task),
            inputs=req.inputs,
            priority=req.priority,
            state=JobState.QUEUED,
            created_at=now,
            progress=JobProgress(stage="queued", step=0, total=0, percent=0.0),
        )

        await self.repository.create_job(job)
        await self.broker.enqueue(job.job_id, job.service, job.priority)

        # Update telemetry metrics asynchronously off the hot request path
        metrics.increment_requests()
        self._schedule_telemetry_update()

        # Emit initial queued event
        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "progress",
            {
                "job_id": job_id,
                "trace_id": trace_id,
                "stage": "queued",
                "percent": 0.0,
                "step": 0,
                "total": 0,
            },
        )

        return job, True

    async def get_job(self, job_id: str) -> FactoryJob:
        job = await self.repository.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job '{job_id}' not found.")
        return job

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        service: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[FactoryJob]:
        """Lists jobs ordered by created_at DESC with optional pagination and filtering."""
        return await self.repository.list_jobs(
            limit=limit,
            offset=offset,
            service=service,
            state=state,
        )

    async def cancel_job(self, job_id: str) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state == JobState.SUCCEEDED:
            raise InvalidRequestError("Cannot cancel a job that has already completed successfully.")

        if job.state == JobState.CANCELLED:
            # Idempotent cancel
            return job

        # Validate state transition
        JobStateMachine.validate_transition(job.state, JobState.CANCELLED, job_id)

        job.state = JobState.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.worker = None

        await self.repository.update_job(job)
        await self.broker.remove(job_id, job.service)

        # Broadcast cancellation event
        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "cancelled",
            {"job_id": job_id, "state": "CANCELLED", "message": "Job successfully cancelled."},
        )

        # Update metrics
        self._schedule_telemetry_update()

        return job

    async def retry_job(self, job_id: str) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state == JobState.SUCCEEDED:
            raise InvalidRequestError("Cannot retry a job that has completed successfully.")

        if job.retry_count >= job.max_retries:
            raise FactoryError(
                message=f"Job '{job_id}' has reached maximum retry limit ({job.max_retries}).",
                code="MAX_RETRIES_EXCEEDED",
                status_code=400,
                details={"job_id": job_id, "retry_count": job.retry_count, "max_retries": job.max_retries},
            )

        JobStateMachine.validate_transition(job.state, JobState.RETRYING, job_id)

        job.retry_count += 1
        job.state = JobState.QUEUED
        job.progress = JobProgress(stage="queued_retry", step=0, total=20, percent=0.0)
        job.worker = None
        job.completed_at = None

        await self.repository.update_job(job)
        await self.broker.enqueue(job.job_id, job.service, job.priority)

        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "progress",
            {
                "job_id": job_id,
                "stage": "queued_retry",
                "percent": 0.0,
                "step": 0,
                "total": 20,
                "message": f"Job re-enqueued for retry ({job.retry_count}/{job.max_retries})",
            },
        )

        return job

    def _verify_worker_lease(
        self,
        job: FactoryJob,
        worker_id: Optional[str],
        lease_token: Optional[str],
        caller_worker_id: Optional[str] = None,
        caller_role: Optional[str] = None,
    ) -> None:
        """
        Enforces strict worker lease authorization.
        - Enforces that caller owns worker, worker owns lease, lease belongs to job,
          and lease is unexpired (job.worker.lease_expires_at >= now).
        - Strictly blocks cross-worker job mutation (HTTP 403 UnauthorizedError).
        - Rejects mutations on unleased jobs if worker credentials are provided or missing when
          called by an authenticated worker identity.
        """
        if job.worker is None:
            if caller_worker_id is not None or worker_id is not None or lease_token is not None:
                raise UnauthorizedError(
                    message=f"Job '{job.job_id}' has no active worker lease.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id or caller_worker_id,
                    },
                )
            return

        # 1. Caller identity verification (JWT subject / worker_id bound to request)
        if caller_worker_id and caller_role != "admin":
            if worker_id and worker_id != caller_worker_id:
                raise UnauthorizedError(
                    message=f"Caller '{caller_worker_id}' is not authorized to act as worker '{worker_id}'.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "caller_worker_id": caller_worker_id,
                        "worker_id": worker_id,
                    },
                )

        # 2. Worker owns lease and token matches active lease
        effective_worker_id = worker_id or caller_worker_id
        if (
            not effective_worker_id
            or not lease_token
            or effective_worker_id != job.worker.worker_id
            or lease_token != job.worker.lease_token
        ):
            raise UnauthorizedError(
                message=f"Worker '{effective_worker_id}' with token '{lease_token}' does not hold active lease for job '{job.job_id}'.",
                status_code=403,
                details={
                    "job_id": job.job_id,
                    "worker_id": effective_worker_id,
                    "lease_worker_id": job.worker.worker_id,
                },
            )

        # 3. Lease is unexpired
        now = datetime.now(timezone.utc)
        expires_at = getattr(job.worker, "expires_at", None) or getattr(job.worker, "lease_expires_at", None)
        if expires_at and expires_at < now:
            raise UnauthorizedError(
                message=f"Lease for job '{job.job_id}' expired at {expires_at.isoformat()}.",
                status_code=403,
                details={
                    "job_id": job.job_id,
                    "worker_id": effective_worker_id,
                    "expires_at": expires_at.isoformat(),
                },
            )

    async def update_progress(
        self,
        job_id: str,
        req: ProgressUpdateRequest,
        caller_worker_id: Optional[str] = None,
        caller_role: Optional[str] = None,
    ) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
            return job

        self._verify_worker_lease(
            job,
            req.worker_id,
            req.lease_token,
            caller_worker_id=caller_worker_id,
            caller_role=caller_role,
        )

        job.progress = JobProgress(
            stage=req.stage,
            step=req.step,
            total=req.total,
            percent=req.percent,
            message=req.message,
            preview_url=req.preview_url,
        )

        if job.state != JobState.RUNNING and job.state != JobState.STREAMING:
            job.state = JobState.RUNNING

        await self.repository.update_job(job)

        payload = {
            "job_id": job_id,
            "stage": req.stage,
            "step": req.step,
            "total": req.total,
            "percent": req.percent,
            "message": req.message,
            "preview_url": req.preview_url,
        }

        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "progress",
            payload,
        )

        return job

    async def complete_job(
        self,
        job_id: str,
        req: JobCompletionRequest,
        caller_worker_id: Optional[str] = None,
        caller_role: Optional[str] = None,
    ) -> FactoryJob:
        job = await self.get_job(job_id)

        # Idempotent no-op for already completed jobs (protects artifacts and telemetry from mutation)
        if job.state == JobState.SUCCEEDED:
            return job

        JobStateMachine.validate_transition(job.state, JobState.SUCCEEDED, job_id)

        self._verify_worker_lease(
            job,
            req.worker_id,
            req.lease_token,
            caller_worker_id=caller_worker_id,
            caller_role=caller_role,
        )

        now = datetime.now(timezone.utc)
        job.state = JobState.SUCCEEDED
        job.completed_at = now
        job.artifacts = req.artifacts
        job.progress = JobProgress(stage="completed", step=job.progress.total, total=job.progress.total, percent=100.0)
        job.telemetry = JobTelemetry(
            inference_duration_ms=req.inference_duration_ms,
            tokens_per_sec=req.tokens_per_sec,
            fps=req.fps,
            peak_vram_mb=req.peak_vram_mb,
        )
        job.worker = None

        await self.repository.update_job(job)

        # Broadcast completion event with full artifacts
        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "completed",
            {
                "job_id": job_id,
                "state": "SUCCEEDED",
                "artifacts": [a.model_dump(mode="json") for a in req.artifacts],
            },
        )

        # Update metrics
        self._schedule_telemetry_update()

        return job

    async def fail_job(
        self,
        job_id: str,
        req: JobFailureRequest,
        caller_worker_id: Optional[str] = None,
        caller_role: Optional[str] = None,
    ) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state in (JobState.SUCCEEDED, JobState.CANCELLED):
            return job

        self._verify_worker_lease(
            job,
            req.worker_id,
            req.lease_token,
            caller_worker_id=caller_worker_id,
            caller_role=caller_role,
        )

        if req.retryable and job.retry_count < job.max_retries:
            return await self.retry_job(job_id)

        JobStateMachine.validate_transition(job.state, JobState.FAILED, job_id)

        now = datetime.now(timezone.utc)
        job.state = JobState.FAILED
        job.completed_at = now
        job.error = req.error
        job.worker = None

        await self.repository.update_job(job)

        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "error",
            {
                "job_id": job_id,
                "state": "FAILED",
                "error": req.error.model_dump(),
            },
        )

        metrics.increment_failures()
        self._schedule_telemetry_update()

        return job
