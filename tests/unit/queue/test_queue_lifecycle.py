"""
tests/unit/queue/test_queue_lifecycle.py
Comprehensive unit tests for Job State Machine, SQLite Repository, Memory Broker,
Leasing Coordinator, Dead-Job Reaper, and Job Lifecycle Manager.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from factory_contracts.errors import FactoryError, InvalidRequestError, NotFoundError
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
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.reaper import DeadJobReaper
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.queue.state_machine import (
    InvalidStateTransitionError,
    JobStateMachine,
)
from factory.apps.control_api.src.streaming.manager import StreamManager


class TestJobStateMachine:
    def test_valid_forward_transitions(self):
        sm = JobStateMachine
        assert sm.can_transition(JobState.CREATED, JobState.QUEUED)
        assert sm.can_transition(JobState.QUEUED, JobState.RUNNING)
        assert sm.can_transition(JobState.RUNNING, JobState.STREAMING)
        assert sm.can_transition(JobState.STREAMING, JobState.SUCCEEDED)

    def test_valid_cancellations(self):
        sm = JobStateMachine
        assert sm.can_transition(JobState.QUEUED, JobState.CANCELLED)
        assert sm.can_transition(JobState.RUNNING, JobState.CANCELLED)
        assert sm.can_transition(JobState.STREAMING, JobState.CANCELLED)
        # Idempotent cancel
        assert sm.can_transition(JobState.CANCELLED, JobState.CANCELLED)

    def test_invalid_transitions_raise_error(self):
        sm = JobStateMachine
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            sm.validate_transition(JobState.SUCCEEDED, JobState.CANCELLED, "job-1")
        assert exc_info.value.code == "INVALID_REQUEST"
        assert exc_info.value.status_code == 400

        with pytest.raises(InvalidStateTransitionError):
            sm.validate_transition(JobState.SUCCEEDED, JobState.QUEUED, "job-2")

        with pytest.raises(InvalidStateTransitionError):
            sm.validate_transition(JobState.CREATED, JobState.STREAMING, "job-3")

    def test_terminal_states(self):
        sm = JobStateMachine
        assert sm.is_terminal(JobState.SUCCEEDED)
        assert sm.is_terminal(JobState.FAILED)
        assert sm.is_terminal(JobState.CANCELLED)
        assert not sm.is_terminal(JobState.RUNNING)
        assert not sm.is_terminal(JobState.QUEUED)


@pytest.mark.asyncio
class TestSqliteRepository:
    async def test_crud_and_idempotency(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()

        job = FactoryJob(
            job_id="job-11111111-2222-3333-4444-555555555555",
            request_id="req-99999999-8888-7777-6666-555555555555",
            idempotency_key="unique-key-123",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "cyberpunk city"},
            priority=20,
            state=JobState.QUEUED,
        )

        created = await repo.create_job(job)
        assert created.job_id == job.job_id

        fetched = await repo.get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.inputs["prompt"] == "cyberpunk city"
        assert fetched.priority == 20

        by_idem = await repo.get_by_idempotency_key("unique-key-123")
        assert by_idem is not None
        assert by_idem.job_id == job.job_id

        # Update
        fetched.state = JobState.RUNNING
        updated = await repo.update_job(fetched)
        assert updated.state == JobState.RUNNING

        await repo.close()

    async def test_worker_leasing_and_matching(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()

        job = FactoryJob(
            job_id="job-11111111-2222-3333-4444-555555555555",
            request_id="req-99999999-8888-7777-6666-555555555555",
            service="video",
            model="wan-2.2",
            task="text_to_video",
            inputs={"prompt": "ocean waves"},
            priority=15,
            state=JobState.QUEUED,
        )
        await repo.create_job(job)

        # Ineligible worker
        ineligible = await repo.find_eligible_jobs_for_worker(
            subsystem="image",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
        )
        assert len(ineligible) == 0

        # Eligible worker
        eligible = await repo.find_eligible_jobs_for_worker(
            subsystem="video",
            supported_models=["wan-2.2"],
            supported_tasks=["text_to_video"],
        )
        assert len(eligible) == 1
        assert eligible[0].job_id == job.job_id

        # Acquire lease
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
        acquired = await repo.acquire_lease(
            job_id=job.job_id,
            worker_id="worker-gpu-1",
            lease_token="lease-abc",
            expires_at=expires_at,
            gpu_device=1,
        )
        assert acquired is True

        leased_job = await repo.get_job(job.job_id)
        assert leased_job.state == JobState.RUNNING
        assert leased_job.worker is not None
        assert leased_job.worker.worker_id == "worker-gpu-1"
        assert leased_job.worker.gpu_device == 1

        # Extend lease
        new_expires = expires_at + timedelta(seconds=300)
        extended = await repo.extend_lease("worker-gpu-1", job.job_id, new_expires)
        assert extended is True

        # Release lease
        released = await repo.release_lease(job.job_id)
        assert released is True

        reloaded = await repo.get_job(job.job_id)
        assert reloaded.worker is None

        await repo.close()


@pytest.mark.asyncio
class TestMemoryBroker:
    async def test_priority_ordering(self):
        broker = MemoryQueueBroker()
        await broker.initialize()

        await broker.enqueue("job-low", "image", priority=5)
        await broker.enqueue("job-high", "image", priority=50)
        await broker.enqueue("job-med", "image", priority=20)

        assert await broker.get_queue_depth("image") == 3

        # Higher priority popped first
        first = await broker.dequeue("image")
        second = await broker.dequeue("image")
        third = await broker.dequeue("image")
        fourth = await broker.dequeue("image")

        assert first == "job-high"
        assert second == "job-med"
        assert third == "job-low"
        assert fourth is None

        await broker.close()

    async def test_remove_job(self):
        broker = MemoryQueueBroker()
        await broker.enqueue("job-1", "video", priority=10)
        await broker.enqueue("job-2", "video", priority=10)

        assert await broker.get_queue_depth("video") == 2
        removed = await broker.remove("job-1", "video")
        assert removed is True
        assert await broker.get_queue_depth("video") == 1

        popped = await broker.dequeue("video")
        assert popped == "job-2"


@pytest.mark.asyncio
class TestLeaseCoordinatorAndReaper:
    async def test_leasing_flow(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        coordinator = LeaseCoordinator(repository=repo)

        job = FactoryJob(
            job_id="job-11111111-2222-3333-4444-555555555555",
            request_id="req-99999999-8888-7777-6666-555555555555",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test"},
            priority=10,
            state=JobState.QUEUED,
        )
        await repo.create_job(job)

        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-gpu-0",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=300,
            )
        )
        assert lease_resp.job is not None
        assert lease_resp.job.job_id == job.job_id
        assert lease_resp.lease_token is not None

        # Heartbeat
        hb_resp = await coordinator.process_heartbeat(
            WorkerHeartbeatRequest(
                worker_id="worker-gpu-0",
                active_job_id=job.job_id,
                lease_token=lease_resp.lease_token,
                gpu_utilization_percent=85.0,
                vram_used_mb=12000.0,
            )
        )
        assert hb_resp.status == "acknowledged"
        assert hb_resp.cancel_requested is False

        # Cancel job and verify next heartbeat flags cancel_requested
        leased_job = await repo.get_job(job.job_id)
        leased_job.state = JobState.CANCELLED
        await repo.update_job(leased_job)

        hb_cancel_resp = await coordinator.process_heartbeat(
            WorkerHeartbeatRequest(
                worker_id="worker-gpu-0",
                active_job_id=job.job_id,
            )
        )
        assert hb_cancel_resp.cancel_requested is True

        await repo.close()

    async def test_dead_job_reaper_auto_retry_and_exhaustion(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        reaper = DeadJobReaper(repository=repo, broker=broker, stream_manager=stream_mgr)

        # Job with expired lease but retry_count < max_retries
        expired_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        job = FactoryJob(
            job_id="job-11111111-2222-3333-4444-555555555555",
            request_id="req-99999999-8888-7777-6666-555555555555",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "stalled job"},
            priority=10,
            state=JobState.RUNNING,
            retry_count=0,
            max_retries=2,
            worker=WorkerLease(
                worker_id="crashed-worker",
                lease_token="lease-expired",
                lease_expires_at=expired_time,
            ),
        )
        await repo.create_job(job)

        recovered = await reaper.sweep_once()
        assert recovered == 1

        reloaded = await repo.get_job(job.job_id)
        assert reloaded.state == JobState.QUEUED
        assert reloaded.retry_count == 1
        assert reloaded.worker is None

        # Expire it again past max_retries
        reloaded.state = JobState.RUNNING
        reloaded.retry_count = 2
        reloaded.worker = WorkerLease(
            worker_id="crashed-worker",
            lease_token="lease-expired-again",
            lease_expires_at=expired_time,
        )
        await repo.update_job(reloaded)

        recovered2 = await reaper.sweep_once()
        assert recovered2 == 1

        failed_job = await repo.get_job(job.job_id)
        assert failed_job.state == JobState.FAILED
        assert failed_job.error is not None
        assert failed_job.error.code == "TIMEOUT"

        await repo.close()

    async def test_lease_coordinator_evicts_job_from_broker(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        job = FactoryJob(
            job_id="job-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            request_id="req-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test evict"},
            priority=10,
            state=JobState.QUEUED,
        )
        await repo.create_job(job)
        await broker.enqueue(job.job_id, job.service, priority=job.priority)

        # Confirm queue depth is 1
        depth_before = await broker.get_queue_depth("image")
        assert depth_before == 1

        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-gpu-0",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=300,
            )
        )
        assert lease_resp.job is not None
        assert lease_resp.job.job_id == job.job_id

        # Verify job was evicted from broker active queue
        depth_after = await broker.get_queue_depth("image")
        assert depth_after == 0

        await repo.close()
        await broker.close()



@pytest.mark.asyncio
class TestJobLifecycleManager:
    async def test_full_job_lifecycle(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        # 1. Submit
        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "a sunset over mountains"},
            priority=10,
            idempotency_key="key-lifecycle-1",
        )
        job, created = await manager.submit_job(sub_req, request_id="req-test-1")
        assert created is True
        assert job.state == JobState.QUEUED

        # Idempotent submit
        job_dup, created_dup = await manager.submit_job(sub_req, request_id="req-test-2")
        assert created_dup is False
        assert job_dup.job_id == job.job_id

        # 2. Progress
        prog_req = ProgressUpdateRequest(stage="sampling", step=5, total=20, percent=25.0)
        job_prog = await manager.update_progress(job.job_id, prog_req)
        assert job_prog.progress.percent == 25.0
        assert job_prog.state == JobState.RUNNING

        # 3. Complete
        comp_req = JobCompletionRequest(
            artifacts=[
                ArtifactReference(
                    artifact_id="art-test-1",
                    mime_type="image/png",
                    size_bytes=204800,
                    storage_path="image/test/output.png",
                    download_url="https://s3.ai.example.com/artifacts/output.png",
                    sha256="a" * 64,
                )
            ],
            inference_duration_ms=4500.0,
            peak_vram_mb=8500.0,
        )
        job_done = await manager.complete_job(job.job_id, comp_req)
        assert job_done.state == JobState.SUCCEEDED
        assert len(job_done.artifacts) == 1
        assert job_done.telemetry.inference_duration_ms == 4500.0

        # Cannot cancel completed job
        with pytest.raises(InvalidRequestError):
            await manager.cancel_job(job.job_id)

        await repo.close()

    async def test_complete_job_idempotent_no_artifact_mutation(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "mountain landscape"},
            priority=10,
        )
        job, _ = await manager.submit_job(sub_req, request_id="req-idem-comp-1")

        # Advance state to RUNNING
        await manager.update_progress(
            job.job_id,
            ProgressUpdateRequest(stage="sampling", step=1, total=10, percent=10.0),
        )

        # 1. Complete with first artifact
        art1 = ArtifactReference(
            artifact_id="art-1",
            mime_type="image/png",
            size_bytes=1024,
            storage_path="image/test/art1.png",
            download_url="https://s3.ai.example.com/art1.png",
            sha256="1" * 64,
        )
        req1 = JobCompletionRequest(artifacts=[art1], inference_duration_ms=1000.0)
        completed1 = await manager.complete_job(job.job_id, req1)
        assert completed1.state == JobState.SUCCEEDED
        assert len(completed1.artifacts) == 1
        assert completed1.artifacts[0].artifact_id == "art-1"
        assert completed1.telemetry.inference_duration_ms == 1000.0

        # 2. Complete again with different artifact (must be idempotent no-op)
        art2 = ArtifactReference(
            artifact_id="art-2",
            mime_type="image/png",
            size_bytes=2048,
            storage_path="image/test/art2.png",
            download_url="https://s3.ai.example.com/art2.png",
            sha256="2" * 64,
        )
        req2 = JobCompletionRequest(artifacts=[art2], inference_duration_ms=9999.0)
        completed2 = await manager.complete_job(job.job_id, req2)
        assert completed2.state == JobState.SUCCEEDED
        assert len(completed2.artifacts) == 1
        assert completed2.artifacts[0].artifact_id == "art-1"
        assert completed2.telemetry.inference_duration_ms == 1000.0

        # Verify from repo directly
        persisted = await repo.get_job(job.job_id)
        assert len(persisted.artifacts) == 1
        assert persisted.artifacts[0].artifact_id == "art-1"
        assert persisted.telemetry.inference_duration_ms == 1000.0

        await repo.close()

