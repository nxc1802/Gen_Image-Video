"""
tests/unit/repository/test_in_memory_repository.py
Unit tests for InMemoryJobStore verifying capacity bounds, thread/coroutine safety,
TTL automatic cleanup, idempotency lookup, and event replay.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import uuid
import pytest

from factory_contracts.jobs.job_contract import (
    AssetRecord,
    FactoryJob,
    JobProgress,
    JobState,
)
from factory.apps.control_api.src.queue.repository.memory import (
    CapacityExceededError,
    InMemoryJobStore,
)


def _make_job(
    job_id: str = None,
    state: JobState = JobState.QUEUED,
    idempotency_key: str = None,
) -> FactoryJob:
    uid = uuid.uuid4().hex
    jid = job_id or f"job-{uid}"
    req_id = f"req-{uid}"
    return FactoryJob(
        job_id=jid,
        request_id=req_id,
        idempotency_key=idempotency_key,
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "A scenic landscape"},
        state=state,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
class TestInMemoryJobStoreCRUD:
    async def test_create_and_get_job(self):
        repo = InMemoryJobStore(max_capacity=50)
        job = _make_job(idempotency_key="idemp-1")
        await repo.create_job(job)

        retrieved = await repo.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id
        assert retrieved.service == "image"
        assert retrieved.state == JobState.QUEUED

        # Idempotency lookup
        by_idemp = await repo.get_by_idempotency_key("idemp-1")
        assert by_idemp is not None
        assert by_idemp.job_id == job.job_id

    async def test_update_job(self):
        repo = InMemoryJobStore(max_capacity=50)
        job = _make_job()
        await repo.create_job(job)

        job.state = JobState.RUNNING
        job.progress = JobProgress(stage="rendering", percent=50.0)
        await repo.update_job(job)

        updated = await repo.get_job(job.job_id)
        assert updated.state == JobState.RUNNING
        assert updated.progress.percent == 50.0

    async def test_list_jobs_and_filtering(self):
        repo = InMemoryJobStore(max_capacity=50)
        j1 = _make_job(state=JobState.QUEUED)
        j2 = _make_job(state=JobState.RUNNING)
        await repo.create_job(j1)
        await repo.create_job(j2)

        queued = await repo.list_jobs(state="QUEUED")
        assert len(queued) == 1
        assert queued[0].job_id == j1.job_id

        running = await repo.list_jobs(state="RUNNING")
        assert len(running) == 1
        assert running[0].job_id == j2.job_id

        all_jobs = await repo.list_jobs()
        assert len(all_jobs) == 2


@pytest.mark.asyncio
class TestCapacityLimitsAndEviction:
    async def test_capacity_exceeded_error_when_all_non_terminal(self):
        repo = InMemoryJobStore(max_capacity=3, default_ttl_seconds=3600)
        await repo.create_job(_make_job(state=JobState.QUEUED))
        await repo.create_job(_make_job(state=JobState.RUNNING))
        await repo.create_job(_make_job(state=JobState.QUEUED))

        # 4th non-terminal job should raise CapacityExceededError
        with pytest.raises(CapacityExceededError):
            await repo.create_job(_make_job(state=JobState.QUEUED))

    async def test_overflow_evicts_oldest_terminal_job(self):
        repo = InMemoryJobStore(max_capacity=3, default_ttl_seconds=3600)
        now = datetime.now(timezone.utc)

        # 1 terminal job, 2 active jobs
        j1 = _make_job(state=JobState.SUCCEEDED, idempotency_key="idemp-completed")
        j1.completed_at = now - timedelta(minutes=10)
        await repo.create_job(j1)
        await repo.create_job(_make_job(state=JobState.RUNNING))
        await repo.create_job(_make_job(state=JobState.RUNNING))

        # Adding a 4th job should evict job-completed to make room
        j4 = _make_job(state=JobState.QUEUED)
        await repo.create_job(j4)

        assert await repo.get_job(j1.job_id) is None
        assert await repo.get_by_idempotency_key("idemp-completed") is None
        assert await repo.get_job(j4.job_id) is not None

    async def test_ttl_automatic_eviction(self):
        # 1-second TTL
        repo = InMemoryJobStore(max_capacity=10, default_ttl_seconds=1)
        old_completed = _make_job(state=JobState.SUCCEEDED)
        old_completed.completed_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await repo.create_job(old_completed)

        # Before eviction cutoff check
        assert await repo.get_job(old_completed.job_id) is not None

        # Next job creation triggers TTL eviction
        await repo.create_job(_make_job(state=JobState.QUEUED))
        assert await repo.get_job(old_completed.job_id) is None


@pytest.mark.asyncio
class TestThreadAndCoroutineSafety:
    async def test_concurrent_job_creation(self):
        repo = InMemoryJobStore(max_capacity=200)

        async def worker(worker_num: int):
            for i in range(10):
                job = _make_job(idempotency_key=f"idemp-{worker_num}-{i}")
                await repo.create_job(job)
                read_job = await repo.get_job(job.job_id)
                assert read_job is not None
                assert read_job.job_id == job.job_id

        # Run 10 concurrent coroutines creating 10 jobs each = 100 jobs
        await asyncio.gather(*(worker(i) for i in range(10)))

        all_jobs = await repo.list_jobs(limit=300)
        assert len(all_jobs) == 100


@pytest.mark.asyncio
class TestEventsAndAssetsAndWorkers:
    async def test_event_recording_and_retrieval(self):
        repo = InMemoryJobStore()
        e1 = await repo.record_event("job-0001", "progress", {"percent": 25.0}, stage="setup")
        e2 = await repo.record_event("job-0001", "progress", {"percent": 75.0}, stage="render")

        events = await repo.get_events_since("job-0001", 0)
        assert len(events) == 2
        assert events[0]["event_id"] == e1
        assert events[1]["event_id"] == e2

        events_since_e1 = await repo.get_events_since("job-0001", e1)
        assert len(events_since_e1) == 1
        assert events_since_e1[0]["event_id"] == e2

    async def test_asset_records(self):
        repo = InMemoryJobStore()
        sha = "a" * 64
        asset = AssetRecord(
            asset_id="ast-1",
            sha256=sha,
            mime_type="image/png",
            size_bytes=1024,
            storage_path="assets/ast-1.png",
            download_url="/v1/assets/ast-1/download",
            s3_url="https://s3.ai.example.com/assets/ast-1.png",
            filename="output.png",
            created_at=datetime.now(timezone.utc),
            verified=True,
        )
        await repo.create_asset(asset)

        by_id = await repo.get_asset("ast-1")
        assert by_id is not None
        assert by_id.size_bytes == 1024

        by_sha = await repo.get_asset_by_sha256(sha)
        assert by_sha is not None
        assert by_sha.asset_id == "ast-1"

    async def test_worker_heartbeat_and_registration(self):
        repo = InMemoryJobStore()
        await repo.upsert_worker_heartbeat(
            worker_id="worker-01",
            subsystem="image",
            framework="comfyui",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
            gpu_device=0,
            vram_used_mb=4096.0,
        )

        worker = await repo.get_worker("worker-01")
        assert worker is not None
        assert worker["worker_id"] == "worker-01"
        assert worker["gpu_device"] == 0
        assert worker["vram_used_mb"] == 4096.0
        assert worker["status"] == "ONLINE"
