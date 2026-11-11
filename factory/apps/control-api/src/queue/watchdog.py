"""
factory/apps/control-api/src/queue/watchdog.py
Deterministic background watchdog for Factory v4 Lite.
Detects disconnected workers (>60s heartbeat timeout) and job execution timeouts.
Transitions jobs directly to FAILED with retryable=False and releases GPU allocation.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from factory_contracts.errors import ErrorDetail
from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import JobFailureRequest

from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP, get_subsystem_gpu

logger = logging.getLogger(__name__)

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

TASK_EXECUTION_TIMEOUTS = {
    "video_extension": 1200.0,
    "video.extension": 1200.0,
    "extension": 1200.0,
    "first_last_frame_to_video": 900.0,
    "video.fl2v": 900.0,
    "fl2v": 900.0,
    "text_to_video": 900.0,
    "video.t2v": 900.0,
    "t2v": 900.0,
    "image_to_video": 900.0,
    "video.i2v": 900.0,
    "i2v": 900.0,
    "cover": 900.0,
    "repaint": 900.0,
    "text_to_music": 600.0,
    "lyrics_to_music": 600.0,
    "instruction_edit": 450.0,
    "controlnet": 450.0,
}

MODEL_EXECUTION_TIMEOUTS = {
    "wan-2.2": 1200.0,
    "minimax-h3": 900.0,
    "ace-step-1.5": 600.0,
    "qwen-image-2.1": 300.0,
    "qwen-image-2.1-turbo": 200.0,
}


async def check_active_jobs_watchdog(
    repository,
    scheduler,
    job_manager,
    heartbeat_timeout_seconds: float = 60.0,
    default_job_timeout_seconds: float = 300.0,
    service_timeouts: Optional[dict] = None,
    runtime_manager: Optional[Any] = None,
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

        # Task/modality/model specific execution timeout budget
        service_key = (job.service or "").lower()
        task_key = (job.task or "").lower()
        model_key = (job.model or "").lower()

        # Priority 1: job.inputs timeout overrides
        timeout_threshold = None
        if job.inputs and isinstance(job.inputs, dict):
            for t_field in ("timeout", "timeout_seconds", "execution_timeout"):
                if job.inputs.get(t_field) is not None:
                    try:
                        timeout_threshold = float(job.inputs[t_field])
                        break
                    except (ValueError, TypeError):
                        pass

        # Priority 2: Custom service_timeouts (if provided): task_key, then model_key, then service_key
        if timeout_threshold is None and service_timeouts is not None and service_timeouts is not SERVICE_EXECUTION_TIMEOUTS:
            if task_key and task_key in service_timeouts:
                timeout_threshold = float(service_timeouts[task_key])
            elif model_key and model_key in service_timeouts:
                timeout_threshold = float(service_timeouts[model_key])
            elif service_key and service_key in service_timeouts:
                if task_key and task_key in TASK_EXECUTION_TIMEOUTS:
                    timeout_threshold = float(TASK_EXECUTION_TIMEOUTS[task_key])
                elif model_key and model_key in MODEL_EXECUTION_TIMEOUTS:
                    timeout_threshold = float(MODEL_EXECUTION_TIMEOUTS[model_key])
                else:
                    timeout_threshold = float(service_timeouts[service_key])

        # Priority 3: Module constants in order: TASK -> MODEL -> SERVICE
        if timeout_threshold is None:
            if task_key and task_key in TASK_EXECUTION_TIMEOUTS:
                timeout_threshold = float(TASK_EXECUTION_TIMEOUTS[task_key])
            elif model_key and model_key in MODEL_EXECUTION_TIMEOUTS:
                timeout_threshold = float(MODEL_EXECUTION_TIMEOUTS[model_key])
            elif service_key and service_key in SERVICE_EXECUTION_TIMEOUTS:
                timeout_threshold = float(SERVICE_EXECUTION_TIMEOUTS[service_key])

        # Priority 4: Default job timeout seconds
        if timeout_threshold is None:
            timeout_threshold = float(default_job_timeout_seconds)

        # Heartbeat-aware deadline: if worker is actively heartbeating and making progress,
        # ensure slow valid video/music generation is not killed prematurely.
        if hb_elapsed <= heartbeat_timeout_seconds and (job.progress and (job.progress.percent or 0.0) > 0.0):
            progress_ratio = max(0.0, min(1.0, float(job.progress.percent) / 100.0))
            effective_timeout = timeout_threshold * (1.0 + progress_ratio * 0.5)
        else:
            effective_timeout = timeout_threshold

        fail_code = None
        fail_msg = None
        if hb_elapsed > heartbeat_timeout_seconds:
            fail_code = "WORKER_DISCONNECTED"
            fail_msg = f"Worker heartbeat timed out after {int(hb_elapsed)}s (threshold: {int(heartbeat_timeout_seconds)}s)."
        elif exec_elapsed > effective_timeout:
            fail_code = "TIMEOUT"
            fail_msg = f"Job execution duration {int(exec_elapsed)}s exceeded timeout threshold {int(effective_timeout)}s."

        if fail_code:
            logger.warning(
                f"Watchdog terminating job '{job.job_id}' on GPU {gpu_device}: {fail_code} - {fail_msg}"
            )
            # Release GPU partition immediately (releasing single GPU and multi-GPU [0, 1] allocations)
            if scheduler and hasattr(scheduler, "release_gpus"):
                scheduler.release_gpus(job_id=job.job_id)
            if scheduler and hasattr(scheduler, "release_gpu"):
                scheduler.release_gpu(gpu_device, job.job_id)

            if runtime_manager and hasattr(runtime_manager, "cancel"):
                try:
                    await runtime_manager.cancel(job.job_id)
                except Exception:
                    pass

            if worker_id and repository and hasattr(repository, "clear_worker_job"):
                try:
                    await repository.clear_worker_job(worker_id, job.job_id)
                except Exception:
                    pass

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
    timeouts = service_timeouts
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            repository = getattr(app.state, "repository", None)
            scheduler = getattr(app.state, "scheduler", getattr(app.state, "broker", None))
            job_manager = getattr(app.state, "job_manager", None)
            runtime_manager = getattr(app.state, "model_runtime_manager", getattr(app.state, "runtime_manager", None))
            if repository and scheduler and job_manager:
                await check_active_jobs_watchdog(
                    repository=repository,
                    scheduler=scheduler,
                    job_manager=job_manager,
                    heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                    service_timeouts=timeouts,
                    runtime_manager=runtime_manager,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in watchdog loop: {e}", exc_info=True)
