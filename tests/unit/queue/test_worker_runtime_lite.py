"""
tests/unit/queue/test_worker_runtime_lite.py
Unit tests verifying Phase 3 (Worker & GPU Runtime Simplification):
1. Service-specific execution timeouts (video tasks up to 900s when heartbeats are fresh).
2. Worker heartbeat timeout separation from task execution timeout.
3. Worker leasing and processing with optional / empty lease_token.
"""
from datetime import datetime, timedelta, timezone
import pytest

from factory_contracts.jobs.job_contract import JobState, JobSubmissionRequest
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest, WorkerHeartbeatRequest
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository import InMemoryJobStore
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.watchdog import (
    SERVICE_EXECUTION_TIMEOUTS,
    check_active_jobs_watchdog,
)
from factory.apps.control_api.src.streaming.manager import StreamManager


@pytest.mark.asyncio
class TestWorkerRuntimeLite:
    async def _setup(self):
        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        scheduler = InMemoryScheduler()
        await scheduler.initialize()
        stream_mgr = StreamManager(repository=repo)
        manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=scheduler)
        return repo, scheduler, stream_mgr, manager, coordinator

    async def test_video_job_allowed_up_to_900s_with_fresh_heartbeats(self):
        """
        Video generation (e.g. Wan 2.2) can take > 300s.
        With SERVICE_EXECUTION_TIMEOUTS configured, a job running 400s with active heartbeats
        must NOT be killed by the watchdog.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "15-minute video job"},
                ),
                request_id="req-video-900",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-video-01",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease.job is not None

            # 400s execution duration (> 300s standard default, but < 900s video budget)
            now = datetime.now(timezone.utc)
            old_started = now - timedelta(seconds=400)
            fresh_hb = now.isoformat()

            await repo.upsert_worker_heartbeat(
                worker_id="worker-video-01",
                active_job_id=job.job_id,
                subsystem="video",
                gpu_device=1,
            )
            async with repo._lock:
                repo._workers["worker-video-01"]["last_heartbeat"] = fresh_hb
                repo._jobs[job.job_id].started_at = old_started

            # Run watchdog with service_timeouts
            failed_count = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
                service_timeouts=SERVICE_EXECUTION_TIMEOUTS,
            )
            assert failed_count == 0

            current_job = await repo.get_job(job.job_id)
            assert current_job.state == JobState.RUNNING

            # Now simulate exceeding the 900s video budget (950s)
            very_old_started = now - timedelta(seconds=950)
            async with repo._lock:
                repo._jobs[job.job_id].started_at = very_old_started

            failed_after_budget = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
                service_timeouts=SERVICE_EXECUTION_TIMEOUTS,
            )
            assert failed_after_budget == 1

            timed_out_job = await repo.get_job(job.job_id)
            assert timed_out_job.state == JobState.FAILED
            assert timed_out_job.error.code == "TIMEOUT"
            assert scheduler.is_gpu_busy(1) is False
        finally:
            await repo.close()
            await scheduler.close()

    async def test_worker_disconnected_heartbeat_timeout(self):
        """
        When worker stops sending heartbeats (> 60s), watchdog fails the job as WORKER_DISCONNECTED.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Crash test"},
                ),
                request_id="req-crash",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-crashed",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease.job is not None

            # Heartbeat is 70s in the past
            now = datetime.now(timezone.utc)
            stale_hb = (now - timedelta(seconds=70)).isoformat()
            async with repo._lock:
                repo._workers["worker-crashed"]["last_heartbeat"] = stale_hb

            failed = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
            )
            assert failed == 1

            failed_job = await repo.get_job(job.job_id)
            assert failed_job.state == JobState.FAILED
            assert failed_job.error.code == "WORKER_DISCONNECTED"
            assert scheduler.is_gpu_busy(0) is False
        finally:
            await repo.close()
            await scheduler.close()
