"""
factory/apps/control-api/src/queue/manager.py
Central JobLifecycleManager coordinating job submission, states, retries, and events.
"""
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
from factory_contracts.tasks import normalize_task
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

    async def submit_job(
        self,
        req: JobSubmissionRequest,
        request_id: str,
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
            idempotency_key=req.idempotency_key,
            service=req.service,
            model=req.model,
            task=normalize_task(req.task),
            inputs=req.inputs,
            priority=req.priority,
            state=JobState.QUEUED,
            created_at=now,
            progress=JobProgress(stage="queued", step=0, total=0, percent=0.0),
        )

        await self.repository.create_job(job)
        await self.broker.enqueue(job.job_id, job.service, job.priority)

        # Update telemetry metrics
        active, queued = await self.repository.count_active_and_queued()
        metrics.set_active_jobs(active)
        metrics.set_queue_depth(queued)
        metrics.increment_requests()

        # Emit initial queued event
        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "progress",
            {"job_id": job_id, "stage": "queued", "percent": 0.0, "step": 0, "total": 0},
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
        active, queued = await self.repository.count_active_and_queued()
        metrics.set_active_jobs(active)
        metrics.set_queue_depth(queued)

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
    ) -> None:
        """
        Enforces worker lease authorization.
        If the job has an active worker lease:
        - Rejects requests from unauthorized workers (mismatched worker_id or lease_token) with HTTP 403 UnauthorizedError.
        If the job is unleased (e.g. direct unit tests):
        - Permits modification only when worker_id is None and lease_token is None.
        """
        if job.worker is not None:
            if not worker_id or not lease_token or worker_id != job.worker.worker_id or lease_token != job.worker.lease_token:
                raise UnauthorizedError(
                    message=f"Worker '{worker_id}' is not authorized to update job '{job.job_id}' under active lease.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id,
                    },
                )
        else:
            if worker_id is not None or lease_token is not None:
                raise UnauthorizedError(
                    message=f"Job '{job.job_id}' is not leased to worker '{worker_id}'.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id,
                    },
                )

    async def update_progress(self, job_id: str, req: ProgressUpdateRequest) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
            return job

        self._verify_worker_lease(job, req.worker_id, req.lease_token)

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

    async def complete_job(self, job_id: str, req: JobCompletionRequest) -> FactoryJob:
        job = await self.get_job(job_id)

        # Idempotent no-op for already completed jobs (protects artifacts and telemetry from mutation)
        if job.state == JobState.SUCCEEDED:
            return job

        JobStateMachine.validate_transition(job.state, JobState.SUCCEEDED, job_id)

        self._verify_worker_lease(job, req.worker_id, req.lease_token)

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
        active, queued = await self.repository.count_active_and_queued()
        metrics.set_active_jobs(active)
        metrics.set_queue_depth(queued)

        return job

    async def fail_job(self, job_id: str, req: JobFailureRequest) -> FactoryJob:
        job = await self.get_job(job_id)

        if job.state in (JobState.SUCCEEDED, JobState.CANCELLED):
            return job

        self._verify_worker_lease(job, req.worker_id, req.lease_token)

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
        active, queued = await self.repository.count_active_and_queued()
        metrics.set_active_jobs(active)
        metrics.set_queue_depth(queued)

        return job
