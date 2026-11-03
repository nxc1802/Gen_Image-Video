"""
factory/apps/control-api/src/queue/watchdog.py
Deterministic background watchdog for Factory v4 Lite.
Detects disconnected workers (>60s heartbeat timeout) and job execution timeouts.
Transitions jobs directly to FAILED with retryable=False and releases GPU allocation.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from factory_contracts.errors import ErrorDetail
from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import JobFailureRequest

logger = logging.getLogger(__name__)

SUBSYSTEM_GPU_MAP = {
    "text": 0, "vlm": 0, "vision": 0, "decision": 0, "image": 0, "speech": 0,
    "video": 1, "music": 1, "enhancement": 1, "embedding": 1, "embed": 1, "rerank": 1,
}

SERVICE_EXECUTION_TIMEOUTS = {
    "text": 300.0,
    "vlm": 300.0,
    "vision": 300.0,
    "decision": 60.0,
    "embedding": 60.0,
    "embed": 60.0,
    "rerank": 60.0,
    "image": 300.0,
    "speech": 300.0,
    "music": 600.0,
    "video": 900.0,
    "enhancement": 600.0,
}


async def check_active_jobs_watchdog(
    repository,
    scheduler,
    job_manager,
    heartbeat_timeout_seconds: float = 60.0,
    default_job_timeout_seconds: float = 300.0,
    service_timeouts: Optional[dict] = None,
) -> int:
    """
    Scans active jobs (RUNNING or STREAMING).
    If worker heartbeat has not arrived in > 60s:
      - Mark job FAILED with code="WORKER_DISCONNECTED" (HTTP 502), retryable=False.
      - Release GPU partition immediately via scheduler.release_gpu.
    If job execution has exceeded timeout:
      - Mark job FAILED with code="TIMEOUT" (HTTP 504), retryable=False.
      - Release GPU partition immediately via scheduler.release_gpu.
    Returns the count of timed-out / failed jobs.
    """
    failed_count = 0
    now = datetime.now(timezone.utc)

    if hasattr(repository, "_jobs") and hasattr(repository, "_lock"):
        async with repository._lock:
            active_jobs = [
                j.model_copy(deep=True)
                for j in repository._jobs.values()
                if j.state in (JobState.RUNNING, JobState.STREAMING)
            ]
    else:
        running = await repository.list_jobs(limit=1000, state="RUNNING")
        streaming = await repository.list_jobs(limit=1000, state="STREAMING")
        active_jobs = running + streaming

    for job in active_jobs:
        gpu_device = None
        if job.worker and job.worker.gpu_device is not None:
            gpu_device = job.worker.gpu_device
        else:
            gpu_device = SUBSYSTEM_GPU_MAP.get(job.service, 0)

        # 1. Check worker heartbeat
        worker_id = job.worker.worker_id if job.worker else None
        last_hb = None
        if worker_id:
            worker_data = await repository.get_worker(worker_id)
            if worker_data and worker_data.get("last_heartbeat"):
                try:
                    hb_str = worker_data["last_heartbeat"]
                    last_hb = datetime.fromisoformat(hb_str.replace("Z", "+00:00"))
                except Exception:
                    last_hb = None

        if last_hb is None:
            if job.worker and job.worker.leased_at:
                last_hb = job.worker.leased_at
            elif job.started_at:
                last_hb = job.started_at
            else:
                last_hb = job.created_at

        if last_hb.tzinfo is None:
            last_hb = last_hb.replace(tzinfo=timezone.utc)

        hb_elapsed = (now - last_hb).total_seconds()

        # Check job start / execution timeout
        started_at = (
            job.started_at
            or (job.worker.leased_at if (job.worker and job.worker.leased_at) else None)
            or job.created_at
        )
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        exec_elapsed = (now - started_at).total_seconds()

        # Task/modality specific execution timeout budget
        service_key = (job.service or "").lower()
        if service_timeouts and service_key in service_timeouts:
            timeout_threshold = service_timeouts[service_key]
        else:
            timeout_threshold = default_job_timeout_seconds

        fail_code = None
        fail_msg = None
        if hb_elapsed > heartbeat_timeout_seconds:
            fail_code = "WORKER_DISCONNECTED"
            fail_msg = f"Worker heartbeat timed out after {int(hb_elapsed)}s (threshold: {int(heartbeat_timeout_seconds)}s)."
        elif exec_elapsed > timeout_threshold:
            fail_code = "TIMEOUT"
            fail_msg = f"Job execution duration {int(exec_elapsed)}s exceeded timeout threshold {int(timeout_threshold)}s."

        if fail_code:
            logger.warning(
                f"Watchdog terminating job '{job.job_id}' on GPU {gpu_device}: {fail_code} - {fail_msg}"
            )
            # Release GPU partition immediately
            if scheduler and hasattr(scheduler, "release_gpu"):
                scheduler.release_gpu(gpu_device, job.job_id)

            # Transition job directly to FAILED with retryable=False
            try:
                await job_manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(
                            code=fail_code,
                            message=fail_msg,
                            request_id=job.request_id,
                        ),
                        retryable=False,
                        worker_id=worker_id,
                    ),
                    caller_role="admin",
                )
                failed_count += 1
            except Exception as e:
                logger.error(f"Failed to transition timed-out job '{job.job_id}': {e}", exc_info=True)

    return failed_count


async def watchdog_loop(
    app,
    interval_seconds: float = 10.0,
    heartbeat_timeout_seconds: float = 60.0,
    service_timeouts: Optional[dict] = None,
) -> None:
    """Continuous background loop executing watchdog sweep every interval_seconds."""
    timeouts = service_timeouts if service_timeouts is not None else SERVICE_EXECUTION_TIMEOUTS
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            repository = getattr(app.state, "repository", None)
            scheduler = getattr(app.state, "scheduler", getattr(app.state, "broker", None))
            job_manager = getattr(app.state, "job_manager", None)
            if repository and scheduler and job_manager:
                await check_active_jobs_watchdog(
                    repository=repository,
                    scheduler=scheduler,
                    job_manager=job_manager,
                    heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                    service_timeouts=timeouts,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in watchdog loop: {e}", exc_info=True)
