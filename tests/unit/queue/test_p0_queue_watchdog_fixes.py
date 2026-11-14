"""
tests/unit/queue/test_p0_queue_watchdog_fixes.py
Targeted regression tests for Phase 0 fixes:
- FIX-01: Reject success if required artifact is missing, empty, or inaccessible -> Produces FAILED.
- FIX-02: Remove fabricated all-zero SHA-256 fallback, compute real hash.
- FIX-03: Retry through same GPU-aware enqueue logic as first submission.
- FIX-04: Consolidate duplicated SUBSYSTEM_GPU_MAP into policy.py.
- FIX-05: Watchdog task/model/variant timeouts and heartbeat-aware deadlines.
"""
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import tempfile
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]

from factory_contracts.errors import ErrorCode, ErrorDetail
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
)
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.watchdog import (
    SERVICE_EXECUTION_TIMEOUTS,
    check_active_jobs_watchdog,
)
from factory.apps.control_api.src.streaming.manager import StreamManager


@pytest.mark.asyncio
async def test_fix_04_explicit_model_bindings_replace_subsystem_gpu_map():
    from model_registry.deployment import DeploymentRegistry
    registry = DeploymentRegistry.from_environment()
    assert registry.assignment("qwen-image-2.1").devices == (0,)
    assert registry.assignment("wan-2.2").devices == (1,)
    for name in ["manager", "leasing", "scheduler", "watchdog", "policy"]:
        source = Path(f"factory/apps/control-api/src/queue/{name}.py").read_text()
        assert "SUBSYSTEM_GPU_MAP" not in source
        assert "get_subsystem_gpu" not in source


@pytest.mark.asyncio
async def test_fix_01_missing_artifact_produces_failed():
    """FIX-01: complete_job with missing or empty artifact_path produces FAILED, not SUCCEEDED."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    job1, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test prompt 1"},
        ),
        request_id="req-fix-01-a",
    )

    # 1. Non-existent artifact path
    non_existent = "/tmp/does_not_exist_image_xyz.png"
    comp_req = JobCompletionRequest(
        artifact_path=non_existent,
        inference_duration_ms=100.0,
    )
    res = await manager.complete_job(job1.job_id, comp_req)
    assert res.state == JobState.FAILED
    assert res.error.code == ErrorCode.ARTIFACT_MISSING

    # 2. Empty (0-byte) artifact file with fresh job
    job2, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test prompt 2"},
        ),
        request_id="req-fix-01-b",
    )
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        comp_req2 = JobCompletionRequest(
            artifact_path=tmp.name,
            inference_duration_ms=100.0,
        )
        res2 = await manager.complete_job(job2.job_id, comp_req2)
        assert res2.state == JobState.FAILED
        assert res2.error.code == ErrorCode.ARTIFACT_MISSING

    await repo.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_fix_02_real_sha256_computation_and_rejection_of_zeros():
    """FIX-02: Fabricated all-zero sha256 is rejected; genuine file bytes compute real sha256."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "mountain landscape"},
        ),
        request_id="req-fix-02",
    )

    # Rejects fabricated 0*64 sha256
    art_fake = ArtifactReference(
        artifact_id="art-fake-01",
        type="image",
        mime_type="image/png",
        size_bytes=100,
        storage_path="some/path.png",
        download_url="https://example.com/art.png",
        sha256="0" * 64,
    )
    req_fake = JobCompletionRequest(artifacts=[art_fake], inference_duration_ms=500.0)
    failed_job = await manager.complete_job(job.job_id, req_fake)
    assert failed_job.state == JobState.FAILED
    assert failed_job.error.code == ErrorCode.ARTIFACT_MISSING

    # Real file with genuine bytes produces matching sha256
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        content = b"GENUINE_FACTORY_IMAGE_PAYLOAD_1234567890"
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    expected_sha = hashlib.sha256(content).hexdigest()
    try:
        job_valid, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "mountain landscape valid"},
            ),
            request_id="req-fix-02-valid",
        )
        req_valid = JobCompletionRequest(artifact_path=tmp_path, inference_duration_ms=600.0)
        succeeded_job = await manager.complete_job(job_valid.job_id, req_valid)
        assert succeeded_job.state == JobState.SUCCEEDED
        assert len(succeeded_job.artifacts) == 1
        assert succeeded_job.artifacts[0].sha256 == expected_sha
        assert succeeded_job.artifacts[0].size_bytes == len(content)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        await repo.close()
        await scheduler.close()


@pytest.mark.asyncio
async def test_fix_03_retry_gpu_aware_enqueue():
    """FIX-03: retry_job must route through enqueue_for_gpu for the correct GPU partition."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    # Video job targets GPU 1
    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="video",
            model="wan-2.2",
            task="text_to_video",
            inputs={"prompt": "ocean waves"},
        ),
        request_id="req-fix-03",
    )
    # Dequeue it from GPU 1
    dequeued_id = await scheduler.dequeue_for_gpu(1)
    assert dequeued_id == job.job_id

    # Fail job with retryable=True
    failed_job = await manager.fail_job(
        job.job_id,
        JobFailureRequest(
            error=ErrorDetail(code=ErrorCode.TIMEOUT, message="Backend took too long", request_id="req-fix-03"),
            retryable=True,
        ),
    )
    assert failed_job.state == JobState.QUEUED
    assert failed_job.retry_count == 1

    # Verify that the retried job was re-enqueued to GPU 1 partition queue
    dequeued_retry_id = await scheduler.dequeue_for_gpu(1)
    assert dequeued_retry_id == job.job_id

    # And GPU 0 queue remains empty
    dequeued_gpu0 = await scheduler.dequeue_for_gpu(0)
    assert dequeued_gpu0 is None

    await repo.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_fix_05_watchdog_configurable_timeouts_and_heartbeat_grace():
    """FIX-05: Watchdog uses configurable task/input timeouts and does not kill progressing jobs."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    now = datetime.now(timezone.utc)
    # Job with custom timeout of 1200s in inputs
    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="video",
            model="wan-2.2",
            task="video_extension",
            inputs={"prompt": "extend clip", "timeout_seconds": 1200},
        ),
        request_id="req-fix-05",
    )

    # Advance to RUNNING with progress
    await manager.update_progress(
        job.job_id,
        ProgressUpdateRequest(stage="sampling", step=15, total=30, percent=50.0),
    )
    async with repo._lock:
        repo._jobs[job.job_id].started_at = now - timedelta(seconds=700)
        repo._jobs[job.job_id].worker = WorkerLease(
            worker_id="worker-vid-01",
            leased_at=now - timedelta(seconds=700),
            gpu_device=1,
        )
    await repo.upsert_worker_heartbeat(
        worker_id="worker-vid-01",
        active_job_id=job.job_id,
        subsystem="video",
        gpu_device=1,
    )
    async with repo._lock:
        repo._workers["worker-vid-01"]["last_heartbeat"] = now.isoformat()

    # Even though 700s has elapsed (exceeding default 300s or 600s),
    # custom timeout (1200s) + active heartbeat + 50% progress keeps the job alive!
    failed_count = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=60.0,
        default_job_timeout_seconds=300.0,
    )
    assert failed_count == 0
    active_job = await repo.get_job(job.job_id)
    assert active_job.state == JobState.RUNNING

    await repo.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_fix_05_watchdog_hierarchy_without_explicit_input_timeout():
    """FIX-05: Watchdog resolves timeouts through hierarchy without explicit inputs.timeout_seconds."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    now = datetime.now(timezone.utc)
    # 1. Job without explicit timeout_seconds in inputs
    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="video",
            model="wan-2.2",
            task="video_extension",
            inputs={"prompt": "extend clip default hierarchy"},
        ),
        request_id="req-fix-05-hierarchy",
    )

    await manager.update_progress(
        job.job_id,
        ProgressUpdateRequest(stage="sampling", step=15, total=30, percent=50.0),
    )
    async with repo._lock:
        repo._jobs[job.job_id].started_at = now - timedelta(seconds=700)
        repo._jobs[job.job_id].worker = WorkerLease(
            worker_id="worker-vid-02",
            leased_at=now - timedelta(seconds=700),
            gpu_device=1,
        )
    await repo.upsert_worker_heartbeat(
        worker_id="worker-vid-02",
        active_job_id=job.job_id,
        subsystem="video",
        gpu_device=1,
    )
    async with repo._lock:
        repo._workers["worker-vid-02"]["last_heartbeat"] = now.isoformat()

    # Run sweep with service_timeouts=None -> TASK_EXECUTION_TIMEOUTS (1200s) resolves and keeps job alive
    failed_count = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=60.0,
        default_job_timeout_seconds=300.0,
    )
    assert failed_count == 0
    active_job = await repo.get_job(job.job_id)
    assert active_job.state == JobState.RUNNING

    # Also verify that passing service_timeouts={"video": 900.0} does not shadow video_extension (1200s)
    failed_count_shadow = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=60.0,
        service_timeouts={"video": 900.0},
    )
    assert failed_count_shadow == 0
    active_job_shadow = await repo.get_job(job.job_id)
    assert active_job_shadow.state == JobState.RUNNING

    await repo.close()
    await scheduler.close()

