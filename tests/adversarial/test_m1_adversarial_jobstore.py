"""
tests/adversarial/test_m1_adversarial_jobstore.py
Adversarial Stress and Empirical Verification Suite for Factory v4 Lite Milestone 1 Gate.

Targeting:
1. Concurrency Stress: 150 concurrent coroutines updating, reading, creating jobs simultaneously.
2. Capacity Overflow: Submitting beyond capacity limit (1,000+ jobs), FIFO eviction of terminal jobs,
   and CapacityExceededError enforcement on active jobs. Memory leak profiling under churn (5,000 jobs).
3. Idempotency Collision: Concurrent submissions sharing identical idempotency_key at store and manager levels.
4. TTL Eviction: Expiration of terminal jobs vs preservation of active/queued jobs, and auditing
   for the claimed `cleanup_expired_jobs` API and timezone comparison safety.
5. InMemoryScheduler: Concurrency, priority ordering, and lock reentrancy deadlock audit on close().
"""
import asyncio
from datetime import datetime, timedelta, timezone
import gc
import os
import time
import tracemalloc
from typing import Dict, List, Optional, Set, Tuple
import uuid
import pytest

from factory_contracts.jobs.job_contract import (
    AssetRecord,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    JobTelemetry,
)
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.memory import (
    CapacityExceededError,
    InMemoryJobStore,
)
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.streaming.manager import StreamManager


def _make_job(
    job_id: Optional[str] = None,
    state: JobState = JobState.QUEUED,
    idempotency_key: Optional[str] = None,
    priority: int = 10,
    service: str = "image",
    created_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> FactoryJob:
    if job_id:
        if job_id.startswith("job-") and all(c in "0123456789abcdef-" for c in job_id[4:]):
            jid = job_id
        else:
            jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, job_id)}"
    else:
        jid = f"job-{uuid.uuid4()}"

    uid = uuid.uuid4().hex
    req_id = f"req-{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:]}"
    now = created_at or datetime.now(timezone.utc)
    return FactoryJob(
        job_id=jid,
        request_id=req_id,
        idempotency_key=idempotency_key,
        service=service,
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "Adversarial benchmark prompt"},
        state=state,
        priority=priority,
        created_at=now,
        completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# 1. CONCURRENCY STRESS (150+ CONCURRENT COROUTINES)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestJobStoreConcurrencyStress:
    async def test_150_concurrent_mixed_operations(self):
        """
        Adversarially stress InMemoryJobStore with 150 concurrent coroutines performing
        mixed operations (create, read, update progress, record events, list, heartbeat).
        Verifies asyncio.Lock prevents race conditions, state corruption, and partial writes.
        Logs throughput, latency percentiles, and memory delta.
        """
        store = InMemoryJobStore(max_capacity=2000, default_ttl_seconds=7200)
        num_workers = 150
        ops_per_worker = 10

        latencies: List[float] = []
        errors: List[Exception] = []

        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()
        start_time = time.perf_counter()

        async def worker_task(worker_id: int):
            for i in range(ops_per_worker):
                t0 = time.perf_counter()
                try:
                    jid = f"job-{uuid.uuid4()}"
                    key = f"key-{worker_id}-{i}"
                    job = _make_job(job_id=jid, idempotency_key=key, priority=(worker_id % 20) + 1)

                    # 1. Create
                    await store.create_job(job)

                    # 2. Update progress
                    job.state = JobState.RUNNING
                    job.progress = JobProgress(stage="rendering", percent=float(i * 10))
                    await store.update_job(job)

                    # 3. Read back
                    retrieved = await store.get_job(jid)
                    assert retrieved is not None
                    assert retrieved.state == JobState.RUNNING
                    assert retrieved.progress.percent == float(i * 10)

                    # 4. Record event
                    eid = await store.record_event(jid, "progress", {"step": i})
                    assert eid > 0

                    # 5. Heartbeat
                    await store.upsert_worker_heartbeat(
                        worker_id=f"worker-{worker_id % 10}",
                        subsystem="image",
                        vram_used_mb=float(1024 + worker_id),
                    )

                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)  # ms
                except Exception as exc:
                    errors.append(exc)

        # Launch 150 concurrent workers
        await asyncio.gather(*(worker_task(w) for w in range(num_workers)))

        total_time = time.perf_counter() - start_time
        mem_after, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_ops = num_workers * ops_per_worker * 5  # 5 operations per loop
        throughput = total_ops / total_time

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        max_lat = latencies[-1]

        # Empirical verifications
        assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent stress: {errors[:5]}"

        # Total jobs created should be exactly 150 * 10 = 1500
        all_jobs = await store.list_jobs(limit=3000)
        assert len(all_jobs) == num_workers * ops_per_worker

        # Active vs queued counts
        active, queued = await store.count_active_and_queued()
        assert active == num_workers * ops_per_worker
        assert queued == 0

        # Print empirical metrics for audit log
        print(f"\n--- Concurrency Stress Metrics ---")
        print(f"Total Coroutines: {num_workers}")
        print(f"Total Operations: {total_ops}")
        print(f"Total Duration: {total_time:.3f} s")
        print(f"Throughput: {throughput:.1f} ops/sec")
        print(f"Latency P50: {p50:.3f} ms | P95: {p95:.3f} ms | P99: {p99:.3f} ms | Max: {max_lat:.3f} ms")
        print(f"Memory Net Delta: {(mem_after - mem_before) / 1024:.1f} KB | Peak: {mem_peak / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# 2. CAPACITY OVERFLOW TESTING & MEMORY LEAK PROFILING
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestCapacityOverflowAndEviction:
    async def test_fifo_eviction_of_terminal_jobs_beyond_1000_capacity(self):
        """
        Verify that submitting jobs beyond max_capacity cleanly triggers FIFO eviction
        of terminal jobs (SUCCEEDED, FAILED, CANCELLED) strictly ordered by completion/creation time,
        while maintaining active jobs without loss.
        """
        capacity = 1000
        store = InMemoryJobStore(max_capacity=capacity, default_ttl_seconds=7200)
        now = datetime.now(timezone.utc)

        # 1. Populate store with 600 terminal jobs and 400 active (QUEUED) jobs
        # Total = 1000 jobs (at full capacity)
        terminal_job_ids: List[str] = []
        for i in range(600):
            jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'term-{i:04d}')}"
            completed_time = now - timedelta(seconds=(600 - i) * 10)
            job = _make_job(
                job_id=jid,
                state=JobState.SUCCEEDED if i % 2 == 0 else JobState.FAILED,
                idempotency_key=f"idem-term-{i}",
                created_at=completed_time - timedelta(seconds=5),
                completed_at=completed_time,
            )
            await store.create_job(job)
            terminal_job_ids.append(jid)

        active_job_ids: List[str] = []
        for i in range(400):
            jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'active-{i:04d}')}"
            job = _make_job(
                job_id=jid,
                state=JobState.QUEUED,
                idempotency_key=f"idem-active-{i}",
            )
            await store.create_job(job)
            active_job_ids.append(jid)

        assert len(await store.list_jobs(limit=1500)) == 1000

        # 2. Now submit 200 NEW jobs.
        # This exceeds capacity. The store must evict the 200 OLDEST terminal jobs.
        new_job_ids: List[str] = []
        for i in range(200):
            jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'new-{i:04d}')}"
            job = _make_job(job_id=jid, state=JobState.QUEUED, idempotency_key=f"idem-new-{i}")
            await store.create_job(job)
            new_job_ids.append(jid)

        # Total jobs in store must remain exactly <= capacity (1000)
        all_jobs = await store.list_jobs(limit=1500)
        assert len(all_jobs) == 1000

        # The first 200 terminal jobs (oldest) must have been evicted
        for jid in terminal_job_ids[:200]:
            assert await store.get_job(jid) is None, f"Oldest terminal job {jid} was not evicted"

        # The subsequent 400 terminal jobs must still exist
        for jid in terminal_job_ids[200:]:
            assert await store.get_job(jid) is not None, f"Terminal job {jid} was evicted prematurely"

        # ALL 400 original active jobs MUST still exist (never evicted)
        for jid in active_job_ids:
            assert await store.get_job(jid) is not None, f"Active job {jid} was wrongfully evicted!"

        # ALL 200 new jobs MUST exist
        for jid in new_job_ids:
            assert await store.get_job(jid) is not None, f"New job {jid} was not stored"

    async def test_capacity_enforcement_raises_when_all_jobs_active(self):
        """
        When all 1000 jobs in the store are non-terminal (QUEUED, RUNNING, ADMITTED),
        submitting job 1001 MUST raise CapacityExceededError with HTTP 429 status code.
        """
        capacity = 100
        store = InMemoryJobStore(max_capacity=capacity, default_ttl_seconds=7200)

        for i in range(capacity):
            job = _make_job(
                job_id=f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'active-full-{i}')}",
                state=JobState.RUNNING if i % 2 == 0 else JobState.QUEUED,
                idempotency_key=f"key-full-{i}",
            )
            await store.create_job(job)

        # Overflow submission
        overflow_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'overflow-job')}"
        overflow_job = _make_job(job_id=overflow_jid, state=JobState.QUEUED)
        with pytest.raises(CapacityExceededError) as exc_info:
            await store.create_job(overflow_job)

        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "QUEUE_FULL"

        # Verify overflow job was not inserted
        assert await store.get_job(overflow_jid) is None
        # Verify exactly capacity retained
        assert len(await store.list_jobs(limit=200)) == capacity

    async def test_memory_leak_under_sustained_churn(self):
        """
        Profile memory usage under sustained churn: 5,000 job cycles through a 200-capacity store.
        Verifies that memory does not continuously grow linearly (leakage).
        """
        capacity = 200
        store = InMemoryJobStore(max_capacity=capacity, default_ttl_seconds=7200)

        gc.collect()
        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()

        total_churn = 5000
        batch_size = 50

        for b in range(total_churn // batch_size):
            # Create 50 jobs
            jobs = []
            for i in range(batch_size):
                jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'churn-{b}-{i}')}"
                j = _make_job(job_id=jid, idempotency_key=f"idem-{b}-{i}")
                await store.create_job(j)
                jobs.append(j)

            # Mark them SUCCEEDED to allow eviction
            now = datetime.now(timezone.utc)
            for j in jobs:
                j.state = JobState.SUCCEEDED
                j.completed_at = now
                await store.update_job(j)

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Job store size must be bounded by capacity
        all_stored = await store.list_jobs(limit=1000)
        assert len(all_stored) <= capacity

        # Memory should be well under 10 MB for 200 stored items
        current_mb = current_mem / (1024 * 1024)
        peak_mb = peak_mem / (1024 * 1024)
        print(f"\n--- Memory Churn Profile (5,000 jobs through 200-cap store) ---")
        print(f"Retained Jobs: {len(all_stored)}")
        print(f"Current Memory: {current_mb:.2f} MB | Peak Memory: {peak_mb:.2f} MB")
        assert current_mb < 15.0, f"Memory usage ({current_mb:.2f} MB) indicates potential memory leak!"


# ---------------------------------------------------------------------------
# 3. IDEMPOTENCY COLLISION TESTING
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestIdempotencyCollisionAdversarial:
    async def test_jobstore_create_job_deduplicates_by_idempotency_key(self):
        """
        Adversarial test on `InMemoryJobStore.create_job`:
        If 50 concurrent callers create a job with different `job_id`s but the SAME `idempotency_key`,
        InMemoryJobStore must deduplicate at storage layer:
        it returns the existing job directly without inserting duplicate jobs or overwriting idempotency map.
        """
        store = InMemoryJobStore(max_capacity=500)
        shared_key = f"collision-store-{uuid.uuid4()}"

        async def submit_one(idx: int):
            job = _make_job(
                job_id=f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, f'collision-{idx}')}",
                idempotency_key=shared_key,
            )
            return await store.create_job(job)

        results = await asyncio.gather(*(submit_one(i) for i in range(50)))
        returned_job_ids = [r.job_id for r in results]

        # Inspect store state
        by_key = await store.get_by_idempotency_key(shared_key)
        all_jobs = await store.list_jobs(limit=100)

        print(f"\n--- InMemoryJobStore.create_job Idempotency Behavior ---")
        print(f"Unique job_ids returned from 50 create_job calls: {len(set(returned_job_ids))}")
        print(f"Total jobs in _jobs dictionary: {len(all_jobs)}")
        print(f"Job pointed by get_by_idempotency_key: {by_key.job_id if by_key else None}")

        # Empirical finding: InMemoryJobStore properly deduplicates at storage layer
        assert len(set(returned_job_ids)) == 1, "Expected all 50 calls to return the same deduplicated job"
        assert len(all_jobs) == 1
        assert by_key is not None
        assert by_key.job_id == returned_job_ids[0]

    async def test_concurrent_idempotency_collisions_via_job_lifecycle_manager(self):
        """
        Adversarially test 50 concurrent submissions sharing the EXACT same idempotency key
        through JobLifecycleManager.submit_job().
        Requirement: concurrent submissions with identical idempotency_key must return
        the identical job_id with 0 duplicates.
        """
        store = InMemoryJobStore(max_capacity=1000)
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=store, broker=broker, stream_manager=stream_mgr)

        shared_key = f"shared-idemp-{uuid.uuid4()}"
        req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "Idempotency collision test"},
            priority=10,
            idempotency_key=shared_key,
        )

        uid = uuid.uuid4().hex
        valid_req_id = f"req-{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:]}"

        # Fire 50 concurrent requests simultaneously
        results: List[Tuple[FactoryJob, bool]] = await asyncio.gather(
            *(manager.submit_job(req, request_id=valid_req_id) for _ in range(50))
        )

        job_ids = [job.job_id for job, _ in results]
        created_flags = [created for _, created in results]

        unique_job_ids = set(job_ids)
        num_created = sum(1 for c in created_flags if c is True)
        num_reused = sum(1 for c in created_flags if c is False)

        print(f"\n--- JobLifecycleManager Concurrent Idempotency Metrics ---")
        print(f"Concurrent Submissions: 50")
        print(f"Unique Job IDs Generated: {len(unique_job_ids)} -> {unique_job_ids}")
        print(f"Jobs Marked Created: {num_created} | Jobs Reused: {num_reused}")

        # Assert zero duplicate job IDs generated
        assert len(unique_job_ids) == 1, (
            f"Race condition in idempotency: {len(unique_job_ids)} distinct jobs created for key '{shared_key}': {unique_job_ids}"
        )
        assert num_created == 1, f"Expected exactly 1 created job, got {num_created}"
        assert num_reused == 49, f"Expected 49 reused jobs, got {num_reused}"


# ---------------------------------------------------------------------------
# 4. TTL EVICTION STRESS & CLEANUP_EXPIRED_JOBS API AUDIT
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestTTLEvictionStress:
    async def test_cleanup_expired_jobs_api_implemented_and_functional(self):
        """
        Verify that `InMemoryJobStore.cleanup_expired_jobs` is implemented, callable,
        and actively sweeps terminal jobs exceeding max_age_seconds/TTL.
        """
        store = InMemoryJobStore(max_capacity=100, default_ttl_seconds=3600)
        assert hasattr(store, "cleanup_expired_jobs"), "cleanup_expired_jobs must be implemented"
        assert callable(getattr(store, "cleanup_expired_jobs")), "cleanup_expired_jobs must be callable"

        now = datetime.now(timezone.utc)
        # Create an expired terminal job
        expired_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'exp-sweep')}"
        expired_job = _make_job(
            job_id=expired_jid,
            state=JobState.SUCCEEDED,
            created_at=now - timedelta(seconds=60),
            completed_at=now - timedelta(seconds=50),
        )
        await store.create_job(expired_job)

        # Create an active job
        active_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'active-keep')}"
        active_job = _make_job(
            job_id=active_jid,
            state=JobState.RUNNING,
            created_at=now - timedelta(seconds=60),
        )
        await store.create_job(active_job)

        # Run cleanup_expired_jobs actively
        deleted_count = await store.cleanup_expired_jobs(max_age_seconds=10)
        assert deleted_count == 1
        assert await store.get_job(expired_jid) is None
        assert await store.get_job(active_jid) is not None

    async def test_ttl_eviction_purges_expired_terminal_and_preserves_active(self):
        """
        Simulate time advancement for expired jobs and verify that expired terminal jobs
        are cleanly purged while preserving active/queued jobs during create_job eviction.
        """
        ttl_seconds = 10
        store = InMemoryJobStore(max_capacity=100, default_ttl_seconds=ttl_seconds)
        now = datetime.now(timezone.utc)

        # 1. Add expired terminal job (completed 60s ago, TTL=10s)
        expired_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'job-expired-terminal')}"
        expired_terminal = _make_job(
            job_id=expired_jid,
            state=JobState.SUCCEEDED,
            idempotency_key="key-exp-term",
            created_at=now - timedelta(seconds=70),
            completed_at=now - timedelta(seconds=60),
        )
        await store.create_job(expired_terminal)

        # 2. Add fresh terminal job (completed 2s ago, TTL=10s)
        fresh_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'job-fresh-terminal')}"
        fresh_terminal = _make_job(
            job_id=fresh_jid,
            state=JobState.FAILED,
            idempotency_key="key-fresh-term",
            created_at=now - timedelta(seconds=5),
            completed_at=now - timedelta(seconds=2),
        )
        await store.create_job(fresh_terminal)

        # 3. Add expired active/queued job (created 60s ago, state=QUEUED)
        # Active jobs must NEVER be purged by TTL eviction!
        old_active_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'job-old-active')}"
        expired_active = _make_job(
            job_id=old_active_jid,
            state=JobState.QUEUED,
            idempotency_key="key-old-active",
            created_at=now - timedelta(seconds=60),
        )
        await store.create_job(expired_active)

        # 4. Trigger eviction via next creation
        trigger_jid = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'job-trigger')}"
        trigger_job = _make_job(job_id=trigger_jid, state=JobState.QUEUED)
        await store.create_job(trigger_job)

        # Empirical verifications:
        # Expired terminal job must be purged
        assert await store.get_job(expired_jid) is None
        assert await store.get_by_idempotency_key("key-exp-term") is None

        # Fresh terminal job must remain
        assert await store.get_job(fresh_jid) is not None

        # Expired active job must NOT be purged
        assert await store.get_job(old_active_jid) is not None, "Active job was purged by TTL eviction!"

        # Trigger job must exist
        assert await store.get_job(trigger_jid) is not None

    async def test_naive_datetime_eviction_handled_gracefully(self):
        """
        Adversarially verify that if a job has timezone-naive datetime timestamps,
        InMemoryJobStore._evict_expired_or_overflow_locked handles it gracefully
        by normalizing naive datetimes to UTC, avoiding any TypeError comparison crash.
        """
        store = InMemoryJobStore(max_capacity=10)
        naive_dt = datetime.now()  # offset-naive
        job1 = FactoryJob(
            job_id=f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'naive-1')}",
            request_id=f"req-{uuid.uuid4()}",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test"},
            state=JobState.SUCCEEDED,
            created_at=naive_dt,
            completed_at=naive_dt,
        )
        await store.create_job(job1)

        # Subsequent create_job triggers eviction comparison: must succeed cleanly without TypeError!
        job2 = _make_job(job_id=f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'trigger')}")
        created2 = await store.create_job(job2)
        assert created2 is not None
        assert await store.get_job(job2.job_id) is not None


# ---------------------------------------------------------------------------
# 5. INMEMORY SCHEDULER DUAL T4 PARTITIONING & DEADLOCK AUDIT
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestInMemorySchedulerStress:
    async def test_gpu_partitioning_and_priority_dequeue(self):
        """
        Verify InMemoryScheduler partitions jobs to Dual T4 GPUs (GPU 0 vs GPU 1),
        maintains strict priority ordering (-priority min-heap), and enforces busy flags.
        """
        scheduler = InMemoryScheduler()

        j0_p5 = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'gpu0-p5')}"
        j0_p20 = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'gpu0-p20')}"
        j0_p10 = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'gpu0-p10')}"

        j1_p1 = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'gpu1-p1')}"
        j1_p50 = f"job-{uuid.uuid5(uuid.NAMESPACE_DNS, 'gpu1-p50')}"

        # Enqueue jobs with different priorities on GPU 0 and GPU 1
        await scheduler.enqueue_for_gpu(j0_p5, service="image", gpu_device=0, priority=5)
        await scheduler.enqueue_for_gpu(j0_p20, service="image", gpu_device=0, priority=20)
        await scheduler.enqueue_for_gpu(j0_p10, service="image", gpu_device=0, priority=10)

        # GPU 1: priorities 1, 50
        await scheduler.enqueue_for_gpu(j1_p1, service="video", gpu_device=1, priority=1)
        await scheduler.enqueue_for_gpu(j1_p50, service="video", gpu_device=1, priority=50)

        # Verify GPU 0 dequeues highest priority first: 20 -> 10 -> 5
        assert await scheduler.dequeue_for_gpu(0) == j0_p20
        assert await scheduler.dequeue_for_gpu(0) == j0_p10
        assert await scheduler.dequeue_for_gpu(0) == j0_p5
        assert await scheduler.dequeue_for_gpu(0) is None

        # Verify GPU 1 dequeues: 50 -> 1
        assert await scheduler.dequeue_for_gpu(1) == j1_p50
        assert await scheduler.dequeue_for_gpu(1) == j1_p1
        assert await scheduler.dequeue_for_gpu(1) is None

        # Concurrency & Busy flags
        assert scheduler.is_gpu_busy(0) is False
        scheduler.set_gpu_busy(0, True)
        assert scheduler.is_gpu_busy(0) is True
        scheduler.set_gpu_busy(0, False)
        assert scheduler.is_gpu_busy(0) is False

    async def test_scheduler_close_succeeds_without_deadlock(self):
        """
        Verify that InMemoryScheduler.close() cleanly shuts down without deadlocking
        on non-reentrant asyncio.Lock.
        """
        scheduler = InMemoryScheduler()
        # Must complete cleanly without timing out
        await asyncio.wait_for(scheduler.close(), timeout=1.0)
        assert scheduler._gpu_queues == {}
        assert scheduler.is_gpu_busy(0) is False
        assert scheduler.is_gpu_busy(1) is False
