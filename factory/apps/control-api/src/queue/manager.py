"""
factory/apps/control-api/src/queue/manager.py
Central JobLifecycleManager coordinating job submission, states, retries, and events.
"""
import asyncio
import inspect
import mimetypes
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    JobTelemetry,
    WorkerLease,
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
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.metrics import metrics

from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP, get_subsystem_gpu


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
        gpu_device = SUBSYSTEM_GPU_MAP.get(job.service, 0)
        enqueue_gpu_fn = getattr(self.broker, "enqueue_for_gpu", None)
        if enqueue_gpu_fn and inspect.iscoroutinefunction(enqueue_gpu_fn):
            await enqueue_gpu_fn(job.job_id, job.service, gpu_device, job.priority)
        else:
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

        gpu_device = job.worker.gpu_device if job.worker and job.worker.gpu_device is not None else None
        if hasattr(self.broker, "release_gpu"):
            self.broker.release_gpu(gpu_device, job_id)

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
        gpu_device = get_subsystem_gpu(job.service)
        enqueue_gpu_fn = getattr(self.broker, "enqueue_for_gpu", None)
        if enqueue_gpu_fn and inspect.iscoroutinefunction(enqueue_gpu_fn):
            await enqueue_gpu_fn(job.job_id, job.service, gpu_device, job.priority)
        else:
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
        worker_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        caller_worker_id: Optional[str] = None,
        caller_role: Optional[str] = None,
    ) -> None:
        """
        Enforces worker lease authorization.
        If the job has an active worker lease:
        - Rejects requests from unauthorized workers (mismatched worker_id or lease_token) with HTTP 403 UnauthorizedError.
        - Rejects expired leases with HTTP 403 UnauthorizedError.
        If the job is unleased (e.g. direct unit tests or local calls):
        - Permits modification only when worker_id is None, lease_token is None, and caller_worker_id is None.
        """
        if caller_worker_id and caller_role != "admin":
            if job.worker is None:
                raise UnauthorizedError(
                    message=f"Caller '{caller_worker_id}' cannot mutate unleased job '{job.job_id}'.",
                    status_code=403,
                    details={"job_id": job.job_id, "caller_worker_id": caller_worker_id},
                )
            if caller_worker_id != worker_id:
                raise UnauthorizedError(
                    message=f"Caller worker identity '{caller_worker_id}' does not match request worker '{worker_id}'.",
                    status_code=403,
                    details={"job_id": job.job_id, "caller_worker_id": caller_worker_id, "worker_id": worker_id},
                )

        if job.worker is not None:
            if worker_id and worker_id != job.worker.worker_id:
                raise UnauthorizedError(
                    message=f"Worker '{worker_id}' is not authorized to update job '{job.job_id}' under active lease.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id,
                    },
                )
            if not worker_id and caller_role != "admin":
                raise UnauthorizedError(
                    message=f"Worker '{worker_id}' is not authorized to update job '{job.job_id}' under active lease.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id,
                    },
                )
            # Strict lease_token verification for non-admin callers
            if caller_role != "admin" and job.worker.lease_token:
                if caller_worker_id or caller_role == "worker" or lease_token:
                    if not lease_token or lease_token != job.worker.lease_token:
                        raise UnauthorizedError(
                            message=f"Invalid or missing lease token for job '{job.job_id}'.",
                            status_code=403,
                            details={
                                "job_id": job.job_id,
                                "worker_id": worker_id,
                            },
                        )
            expires_at = getattr(job.worker, "lease_expires_at", None) or getattr(job.worker, "expires_at", None)
            if expires_at:
                now = datetime.now(timezone.utc)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if now > expires_at:
                    raise UnauthorizedError(
                        message=f"Lease for job '{job.job_id}' has expired.",
                        status_code=403,
                        details={"job_id": job.job_id, "worker_id": worker_id},
                    )
        else:
            if caller_role != "admin" and (worker_id is not None or lease_token is not None):
                raise UnauthorizedError(
                    message=f"Job '{job.job_id}' is not leased to worker '{worker_id}'.",
                    status_code=403,
                    details={
                        "job_id": job.job_id,
                        "worker_id": worker_id,
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

        self._verify_worker_lease(job, req.worker_id, req.lease_token, caller_worker_id, caller_role)

        if job.worker is None and req.worker_id:
            now = datetime.now(timezone.utc)
            job.worker = WorkerLease(
                worker_id=req.worker_id,
                leased_at=now,
                lease_expires_at=now + timedelta(seconds=3600),
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
        self._verify_worker_lease(job, req.worker_id, req.lease_token, caller_worker_id, caller_role)

        now = datetime.now(timezone.utc)
        job.state = JobState.SUCCEEDED
        job.completed_at = now
        job.result = req.result
        job.artifacts = req.artifacts or []

        # FIX-01 & FIX-02: Validate artifact existence, size, and real checksum
        if req.artifact_path and not (req.artifacts or job.artifacts):
            art_path = Path(req.artifact_path)
            if not art_path.is_file() or art_path.stat().st_size <= 0:
                return await self.fail_job(
                    job_id,
                    JobFailureRequest(
                        error=ErrorDetail(
                            code=ErrorCode.ARTIFACT_MISSING,
                            message=f"Required output artifact '{req.artifact_path}' is missing, empty, or inaccessible.",
                            request_id=job.request_id,
                        ),
                        retryable=False,
                        worker_id=req.worker_id,
                        lease_token=req.lease_token,
                    ),
                    caller_worker_id=caller_worker_id,
                    caller_role=caller_role,
                )
            try:
                raw_data = art_path.read_bytes()
                size_bytes = len(raw_data)
                sha256 = compute_bytes_sha256(raw_data)
            except Exception as e:
                return await self.fail_job(
                    job_id,
                    JobFailureRequest(
                        error=ErrorDetail(
                            code=ErrorCode.ARTIFACT_MISSING,
                            message=f"Failed to read output artifact '{req.artifact_path}': {e}",
                            request_id=job.request_id,
                        ),
                        retryable=False,
                        worker_id=req.worker_id,
                        lease_token=req.lease_token,
                    ),
                    caller_worker_id=caller_worker_id,
                    caller_role=caller_role,
                )

            mime = mimetypes.guess_type(art_path.name)[0] or "application/octet-stream"
            art_type = "media"
            if mime.startswith("image/"):
                art_type = "image"
            elif mime.startswith("video/"):
                art_type = "video"
            elif mime.startswith("audio/"):
                art_type = "audio"
            elif mime.startswith("text/"):
                art_type = "text"

            art_ref = ArtifactReference(
                artifact_id=f"art-{uuid.uuid4()}",
                type=art_type,
                mime_type=mime,
                size_bytes=size_bytes,
                storage_path=str(req.artifact_path),
                download_url=f"/v1/jobs/{job_id}/result",
                url=f"/v1/jobs/{job_id}/result",
                sha256=sha256,
                created_at=now,
            )
            job.artifacts = [art_ref]

        elif req.artifacts:
            # FIX-02: Ensure genuine hashes and non-empty sizes
            for art in req.artifacts:
                found_file = None
                if art.storage_path:
                    raw_sp = str(art.storage_path).strip()
                    clean_sp = raw_sp.lstrip("/")
                    scratch_root = os.environ.get("SCRATCH_ARTIFACTS_DIR") or os.environ.get("FACTORY_ARTIFACT_DIR") or "/tmp/factory_lite/artifacts"
                    local_storage = os.environ.get("LOCAL_STORAGE_DIR", "/tmp/factory_storage")
                    candidates = [
                        Path(raw_sp),
                        Path(scratch_root) / job_id / Path(raw_sp).name,
                        Path(scratch_root) / clean_sp,
                        Path(scratch_root) / "factory-artifacts" / clean_sp,
                        Path(local_storage) / clean_sp,
                        Path(local_storage) / "factory-artifacts" / clean_sp,
                    ]
                    for cand in candidates:
                        try:
                            if cand.is_file() and not cand.is_symlink():
                                found_file = cand
                                break
                        except Exception:
                            pass

                # If candidate paths did not yield a file, check scratch job directory
                if found_file is None:
                    scratch_root = os.environ.get("SCRATCH_ARTIFACTS_DIR") or os.environ.get("FACTORY_ARTIFACT_DIR") or "/tmp/factory_lite/artifacts"
                    job_dir = Path(scratch_root) / job_id
                    if job_dir.is_dir():
                        found_scratch = [f for f in sorted(job_dir.iterdir()) if f.is_file() and not f.is_symlink() and not f.name.startswith(".")]
                        if found_scratch:
                            found_file = found_scratch[0]
                            art.storage_path = str(found_file)

                if found_file is not None:
                    file_bytes = found_file.read_bytes()
                    if len(file_bytes) == 0:
                        return await self.fail_job(
                            job_id,
                            JobFailureRequest(
                                error=ErrorDetail(
                                    code=ErrorCode.ARTIFACT_MISSING,
                                    message=f"Artifact file '{art.storage_path}' is empty.",
                                    request_id=job.request_id,
                                ),
                                retryable=False,
                                worker_id=req.worker_id,
                                lease_token=req.lease_token,
                            ),
                            caller_worker_id=caller_worker_id,
                            caller_role=caller_role,
                        )
                    computed_sha = compute_bytes_sha256(file_bytes)
                    if art.sha256 in ("0" * 64, None, ""):
                        art.sha256 = computed_sha
                        if not art.size_bytes or art.size_bytes <= 0:
                            art.size_bytes = len(file_bytes)
                    elif art.sha256 != computed_sha:
                        return await self.fail_job(
                            job_id,
                            JobFailureRequest(
                                error=ErrorDetail(
                                    code=ErrorCode.ARTIFACT_MISSING,
                                    message=f"Artifact SHA-256 mismatch for '{art.storage_path}'.",
                                    request_id=job.request_id,
                                ),
                                retryable=False,
                                worker_id=req.worker_id,
                                lease_token=req.lease_token,
                            ),
                            caller_worker_id=caller_worker_id,
                            caller_role=caller_role,
                        )
                else:
                    is_media_service = job.service in {"image", "video", "music"} or (
                        job.service == "speech" and job.task in {"speech_synthesis", "text_to_speech", "voice_clone"}
                    )
                    is_remote_storage = bool(
                        art.download_url
                        and str(art.download_url).startswith(("http://", "https://", "s3://"))
                        and "testserver" not in str(art.download_url)
                        and "localhost" not in str(art.download_url)
                        and "127.0.0.1" not in str(art.download_url)
                        and not (art.storage_path and Path(str(art.storage_path).strip()).is_absolute())
                    )
                    if is_media_service and not is_remote_storage:
                        return await self.fail_job(
                            job_id,
                            JobFailureRequest(
                                error=ErrorDetail(
                                    code=ErrorCode.ARTIFACT_MISSING,
                                    message=f"Required media artifact file '{art.storage_path}' is missing or inaccessible on disk.",
                                    request_id=job.request_id,
                                ),
                                retryable=False,
                                worker_id=req.worker_id,
                                lease_token=req.lease_token,
                            ),
                            caller_worker_id=caller_worker_id,
                            caller_role=caller_role,
                        )
                    elif art.sha256 == "0" * 64 or not art.sha256 or art.size_bytes <= 0:
                        return await self.fail_job(
                            job_id,
                            JobFailureRequest(
                                error=ErrorDetail(
                                    code=ErrorCode.ARTIFACT_MISSING,
                                    message="Artifact output is invalid, empty, or has fabricated SHA-256 hash.",
                                    request_id=job.request_id,
                                ),
                                retryable=False,
                                worker_id=req.worker_id,
                                lease_token=req.lease_token,
                            ),
                            caller_worker_id=caller_worker_id,
                            caller_role=caller_role,
                        )
            job.artifacts = req.artifacts

        if not job.artifacts:
            scratch_root = os.environ.get("SCRATCH_ARTIFACTS_DIR") or os.environ.get("FACTORY_ARTIFACT_DIR") or "/tmp/factory_lite/artifacts"
            job_dir = Path(scratch_root) / job_id
            if job_dir.is_dir():
                found_files = [f for f in sorted(job_dir.iterdir()) if f.is_file() and not f.is_symlink() and not f.name.startswith(".")]
                if found_files:
                    found_file = found_files[0]
                    raw_data = found_file.read_bytes()
                    if len(raw_data) > 0:
                        sha256 = compute_bytes_sha256(raw_data)
                        mime = mimetypes.guess_type(found_file.name)[0] or "application/octet-stream"
                        art_ref = ArtifactReference(
                            artifact_id=f"art-{uuid.uuid4()}",
                            type=job.service,
                            mime_type=mime,
                            size_bytes=len(raw_data),
                            storage_path=str(found_file),
                            download_url=f"/v1/jobs/{job_id}/result",
                            url=f"/v1/jobs/{job_id}/result",
                            sha256=sha256,
                            created_at=now,
                        )
                        job.artifacts = [art_ref]

        # FIX-01: Media jobs cannot succeed without required artifacts or result payload
        is_media_service = job.service in {"image", "video", "music"} or (
            job.service == "speech" and job.task in {"speech_synthesis", "text_to_speech", "voice_clone"}
        )
        if is_media_service and not job.artifacts and not req.result:
            return await self.fail_job(
                job_id,
                JobFailureRequest(
                    error=ErrorDetail(
                        code=ErrorCode.ARTIFACT_MISSING,
                        message=f"Artifact file not found: job for service '{job.service}' completed without required output artifact.",
                        request_id=job.request_id,
                    ),
                    retryable=False,
                    worker_id=req.worker_id,
                    lease_token=req.lease_token,
                ),
                caller_worker_id=caller_worker_id,
                caller_role=caller_role,
            )

        job.progress = JobProgress(stage="completed", step=job.progress.total, total=job.progress.total, percent=100.0)
        job.telemetry = JobTelemetry(
            inference_duration_ms=req.inference_duration_ms,
            tokens_per_sec=req.tokens_per_sec,
            fps=req.fps,
            peak_vram_mb=req.peak_vram_mb,
        )
        gpu_device = job.worker.gpu_device if job.worker and job.worker.gpu_device is not None else None
        if hasattr(self.broker, "release_gpu"):
            self.broker.release_gpu(gpu_device, job_id)

        job.worker = None

        await self.repository.update_job(job)

        # Broadcast completion event with full artifacts
        await self.stream_manager.publish(
            f"factory:events:job:{job_id}",
            "completed",
            {
                "job_id": job_id,
                "state": "SUCCEEDED",
                "artifacts": [a.model_dump(mode="json") for a in job.artifacts],
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

        self._verify_worker_lease(job, req.worker_id, req.lease_token, caller_worker_id, caller_role)

        if req.retryable and job.retry_count < job.max_retries:
            return await self.retry_job(job_id)

        JobStateMachine.validate_transition(job.state, JobState.FAILED, job_id)

        now = datetime.now(timezone.utc)
        job.state = JobState.FAILED
        job.completed_at = now
        job.error = req.error
        gpu_device = job.worker.gpu_device if job.worker and job.worker.gpu_device is not None else None
        if hasattr(self.broker, "release_gpu"):
            self.broker.release_gpu(gpu_device, job_id)

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
