"""
factory/apps/control-api/src/queue/reaper.py
Dead-Job Reaper: recovers unheartbeated or crashed worker jobs and re-enqueues or fails them.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from factory_contracts.errors import ErrorCode, ErrorDetail
from factory_contracts.jobs.job_contract import JobProgress, JobState
from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.queue.state_machine import JobStateMachine
from factory.apps.control_api.src.streaming.manager import StreamManager

logger = logging.getLogger("control_api.reaper")


class DeadJobReaper:
    """Background task detecting expired worker leases and recovering stalled jobs."""

    def __init__(
        self,
        repository: BaseJobRepository,
        broker: BaseQueueBroker,
        stream_manager: Optional[StreamManager] = None,
        interval_seconds: float = 15.0,
    ):
        self.repository = repository
        self.broker = broker
        self.stream_manager = stream_manager
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def sweep_once(self) -> int:
        """Executes a single sweep cycle over expired jobs. Returns number of recovered jobs."""
        now = datetime.now(timezone.utc)
        expired_jobs = await self.repository.find_expired_leases(cutoff=now)
        recovered_count = 0

        for job in expired_jobs:
            recovered_count += 1
            if job.retry_count < job.max_retries:
                # Auto-retry
                new_retry_count = job.retry_count + 1
                job.retry_count = new_retry_count
                job.state = JobState.QUEUED
                job.progress = JobProgress(
                    stage="queued_retry",
                    step=0,
                    total=0,
                    percent=0.0,
                    message=f"Auto-retrying after worker lease expired (attempt {new_retry_count}/{job.max_retries})",
                )
                job.worker = None

                await self.repository.update_job(job)
                await self.broker.enqueue(job.job_id, job.service, job.priority)

                if self.stream_manager:
                    await self.stream_manager.publish(
                        f"factory:events:job:{job.job_id}",
                        "progress",
                        {
                            "job_id": job.job_id,
                            "stage": "queued_retry",
                            "percent": 0.0,
                            "message": f"Worker lease expired. Re-enqueued for retry {new_retry_count}/{job.max_retries}.",
                        },
                    )
                logger.warning(
                    "Job %s lease expired; auto-retried (%d/%d)",
                    job.job_id,
                    new_retry_count,
                    job.max_retries,
                )
            else:
                # Mark as FAILED
                job.state = JobState.FAILED
                job.completed_at = now
                job.error = ErrorDetail(
                    code=ErrorCode.TIMEOUT,
                    message="Worker lease expired and maximum retry limit reached.",
                    request_id=job.request_id,
                )
                job.worker = None

                await self.repository.update_job(job)

                if self.stream_manager:
                    await self.stream_manager.publish(
                        f"factory:events:job:{job.job_id}",
                        "failed",
                        {
                            "job_id": job.job_id,
                            "state": "FAILED",
                            "error": job.error.model_dump(),
                        },
                    )
                logger.error(
                    "Job %s failed permanently after lease expired and max retries exhausted",
                    job.job_id,
                )

        return recovered_count

    async def _reaper_loop(self) -> None:
        while self._running:
            try:
                await self.sweep_once()
            except Exception as e:
                logger.error("Error during dead-job reaper sweep: %s", e)
            await asyncio.sleep(self.interval)
