"""
tests/adversarial/test_adversarial_m2.py
Milestone 2 Iteration 2 Gate Adversarial Challenge Suite

Adversarial stress-testing of:
1. Reconnection Deduplication Race (StreamManager under high concurrency publish/subscribe)
2. Terminal State Immutability (JobLifecycleManager complete_job, cancel_job, fail_job, update_progress)
3. Queue Broker Eviction & Worker Double-Pickup Prevention (LeaseCoordinator under heavy worker concurrency)
"""
import asyncio
import copy
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Set

import pytest

from factory_contracts.errors import ErrorCode, ErrorDetail, FactoryError, InvalidRequestError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
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
    WorkerLeaseRequest,
    WorkerLeaseResponse,
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


@pytest.mark.asyncio
class TestReconnectionDeduplicationRace:
    """Adversarial stress tests for StreamManager replay and deduplication."""

    async def test_concurrent_publish_and_reconnection_no_duplicates(self):
        """
        Adversarially publish events across multiple concurrent background workers
        while a reconnecting subscriber reads with last_event_id.
        Verify that ZERO duplicate event IDs are delivered to the subscriber,
        and all IDs arrive in strictly monotonic order.
        """
        sm = StreamManager(history_size=300)
        channel = "adversarial-job-stream-1"

        # Pre-populate history with 15 events
        for i in range(1, 16):
            await sm.publish(channel, "progress", {"step": i, "percent": float(i)})

        reconnected_events: List[dict] = []
        stop_publisher = asyncio.Event()

        # Concurrent publisher task continuously publishing events
        async def _aggressive_publisher():
            step = 16
            while not stop_publisher.is_set():
                await sm.publish(
                    channel,
                    "progress",
                    {"step": step, "percent": float(step)},
                )
                step += 1
                await asyncio.sleep(0.001)

        pub_task = asyncio.create_task(_aggressive_publisher())

        # Subscriber reconnects at Last-Event-ID = "10"
        async def _reconnecting_subscriber():
            async for evt in sm.subscribe(channel, last_event_id="10"):
                reconnected_events.append(evt)
                if len(reconnected_events) >= 50:
                    break

        sub_task = asyncio.create_task(_reconnecting_subscriber())

        try:
            await asyncio.wait_for(sub_task, timeout=5.0)
        finally:
            stop_publisher.set()
            await pub_task

        # Assertions
        assert len(reconnected_events) >= 50
        event_ids = [int(e["id"]) for e in reconnected_events]

        # 1. Zero duplicate event IDs
        assert len(event_ids) == len(set(event_ids)), f"Duplicate event IDs found: {event_ids}"

        # 2. All delivered events must have ID > 10 (Last-Event-ID threshold)
        assert all(eid > 10 for eid in event_ids), f"Received event <= 10: {event_ids}"

        # 3. Strictly monotonically increasing
        assert event_ids == sorted(event_ids), f"Out of order events: {event_ids}"

    async def test_multi_subscriber_concurrent_reconnection_fuzzing(self):
        """
        Spawns 10 concurrent subscribers reconnecting at arbitrary historical offsets
        while concurrent publishers inject 100 events into the stream.
        Assert each subscriber experiences zero duplicate IDs and strictly monotonic order.
        """
        sm = StreamManager(history_size=200)
        channel = "adversarial-job-stream-fuzz"

        # Initial seed of 20 events
        for i in range(1, 21):
            await sm.publish(channel, "progress", {"seed_step": i})

        async def _client_worker(client_id: int, start_offset: str) -> List[int]:
            received = []
            async for evt in sm.subscribe(channel, last_event_id=start_offset):
                received.append(int(evt["id"]))
                if evt.get("event") == "completed" or len(received) >= 40:
                    break
            return received

        # Run 10 clients with varying last_event_ids
        client_tasks = [
            asyncio.create_task(_client_worker(c, str(random.randint(0, 15))))
            for c in range(10)
        ]

        # Rapidly publish 60 live events concurrently
        for step in range(21, 81):
            await sm.publish(channel, "progress", {"step": step})
            if step % 10 == 0:
                await asyncio.sleep(0.002)

        # Publish terminal event
        await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        results = await asyncio.gather(*client_tasks)

        for idx, client_ids in enumerate(results):
            assert len(client_ids) > 0, f"Client {idx} received no events"
            # Ensure unique
            assert len(client_ids) == len(set(client_ids)), f"Client {idx} got duplicates: {client_ids}"
            # Ensure strictly ascending
            assert client_ids == sorted(client_ids), f"Client {idx} received unordered IDs: {client_ids}"

    async def test_reconnection_with_adversarial_last_event_ids(self):
        """
        Test malformed or boundary Last-Event-ID headers:
        - Non-numeric strings return empty history and stream live events.
        - Future Last-Event-IDs return empty history and stream live events.
        """
        sm = StreamManager(history_size=50)
        channel = "adversarial-boundary-channel"

        for i in range(1, 6):
            await sm.publish(channel, "progress", {"step": i})

        # Non-numeric Last-Event-ID: get_since returns [] and caller receives subsequent live events
        received_live = []

        async def _live_reader():
            async for evt in sm.subscribe(channel, last_event_id="invalid-id-xyz"):
                received_live.append(evt)
                if evt["event"] == "completed":
                    break

        reader_task = asyncio.create_task(_live_reader())
        await asyncio.sleep(0.01)

        # Now publish live events
        await sm.publish(channel, "progress", {"step": 6})
        await sm.publish(channel, "completed", {"status": "ok"})

        await asyncio.wait_for(reader_task, timeout=2.0)
        assert len(received_live) == 2
        assert received_live[0]["data"]["step"] == 6
        assert received_live[1]["event"] == "completed"

    async def test_reconnection_after_terminal_event_exits_cleanly(self):
        """
        Adversarial test for reconnection exit behavior:
        When a stream has already emitted a terminal event ('completed' or 'failed'),
        a reconnecting subscriber reading with last_event_id must yield historical events
        (including the terminal event) and terminate cleanly without hanging on q.get().
        """
        sm = StreamManager(history_size=50)
        channel = "adversarial-terminal-channel"

        # Publish progress events followed by terminal event
        await sm.publish(channel, "progress", {"percent": 50})
        await sm.publish(channel, "progress", {"percent": 90})
        await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        received = []

        async def _reconnecting_subscriber():
            async for evt in sm.subscribe(channel, last_event_id="0"):
                received.append(evt)

        # Must exit cleanly within 2.0s once terminal event is delivered from history.
        # If it hangs on q.get(), this will raise asyncio.TimeoutError.
        await asyncio.wait_for(_reconnecting_subscriber(), timeout=2.0)

        assert len(received) == 3
        assert received[0]["event"] == "progress"
        assert received[1]["event"] == "progress"
        assert received[2]["event"] == "completed"

    async def test_terminated_channel_reconnect_boundary_last_event_ids(self):
        """
        Adversarial test for boundary last_event_ids on already-terminated channels:
        - last_event_id equal to terminal event id: must return 0 events and exit cleanly.
        - last_event_id greater than terminal event id: must return 0 events and exit cleanly.
        - last_event_id=None: must return 0 events and exit cleanly.
        None of these must hang on q.get().
        """
        sm = StreamManager(history_size=50)
        channel = "adversarial-boundary-terminated-channel"

        await sm.publish(channel, "progress", {"percent": 50})
        await sm.publish(channel, "progress", {"percent": 90})
        terminal_id = await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        # 1. last_event_id == terminal_id
        received_term = []
        async for evt in sm.subscribe(channel, last_event_id=str(terminal_id)):
            received_term.append(evt)
        assert len(received_term) == 0

        # 2. last_event_id > terminal_id
        received_future = []
        async for evt in sm.subscribe(channel, last_event_id="9999"):
            received_future.append(evt)
        assert len(received_future) == 0

        # 3. last_event_id is None
        received_none = []
        async for evt in sm.subscribe(channel, last_event_id=None):
            received_none.append(evt)
        assert len(received_none) == 0

    async def test_reconnection_live_events_queued_during_history_replay_not_dropped(self):
        """
        Adversarial race test for StreamManager historical exit:
        When a subscriber reconnects with last_event_id and historical events are being yielded,
        if subsequent events (including terminal events) are published concurrently to the channel,
        the subscriber must NOT discard events buffered in the subscriber queue when historical
        replay finishes. All events must be delivered and the stream must exit on the terminal event.
        """
        sm = StreamManager(history_size=50)
        channel = "adversarial-dropped-events-channel"

        # Pre-populate 2 events
        await sm.publish(channel, "progress", {"percent": 10})
        await sm.publish(channel, "progress", {"percent": 20})

        received = []

        async def _subscriber():
            async for evt in sm.subscribe(channel, last_event_id="1"):
                received.append(evt)
                # Simulate consumer iteration latency
                await asyncio.sleep(0.02)

        sub_task = asyncio.create_task(_subscriber())
        await asyncio.sleep(0.005)

        # Concurrently publish live progress and terminal event
        await sm.publish(channel, "progress", {"percent": 50})
        await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        await asyncio.wait_for(sub_task, timeout=2.0)

        received_ids = [int(e["id"]) for e in received]
        assert received_ids == [2, 3, 4], f"Events dropped! Received: {received_ids}"
        assert received[-1]["event"] == "completed"

    async def test_reconnect_last_event_id_gte_terminal_event_terminates_cleanly(self):
        """
        Adversarial regression test for subscriber reconnect hang:
        When a subscriber reconnects with last_event_id >= terminal_event_id,
        live events (including the terminal event) arriving in the subscriber queue
        satisfy `eid <= highest_event_id`. The subscriber must NOT skip the terminal
        break and hang forever on q.get(); it must terminate cleanly.
        """
        # Scenario A: last_event_id > terminal_event_id with 'completed' terminal event
        sm_completed = StreamManager()
        channel_a = "reconnect-gte-terminal-completed"

        received_a: List[dict] = []
        sub_gen_a = sm_completed.subscribe(channel_a, last_event_id="50")

        async def _consume_a():
            async for evt in sub_gen_a:
                received_a.append(evt)

        consume_task_a = asyncio.create_task(_consume_a())
        await asyncio.sleep(0.01)

        await sm_completed.publish(channel_a, "progress", {"step": 1})
        await sm_completed.publish(channel_a, "completed", {"result": "ok"})

        await asyncio.wait_for(consume_task_a, timeout=1.0)
        assert len(received_a) == 0, f"Expected 0 events since last_event_id=50 >= all event IDs, got {received_a}"

        # Scenario B: last_event_id == terminal_event_id with 'failed' terminal event
        sm_failed = StreamManager()
        channel_b = "reconnect-gte-terminal-failed"

        # Pre-publish 3 progress events
        for i in range(1, 4):
            await sm_failed.publish(channel_b, "progress", {"step": i})

        received_b: List[dict] = []
        # Reconnect with last_event_id="4" (which will match the upcoming terminal event id 4)
        sub_gen_b = sm_failed.subscribe(channel_b, last_event_id="4")

        async def _consume_b():
            async for evt in sub_gen_b:
                received_b.append(evt)

        consume_task_b = asyncio.create_task(_consume_b())
        await asyncio.sleep(0.01)

        terminal_id = await sm_failed.publish(channel_b, "failed", {"error": "out of memory"})
        assert terminal_id == 4

        await asyncio.wait_for(consume_task_b, timeout=1.0)
        assert len(received_b) == 0, f"Expected 0 events since last_event_id=4 >= terminal_id=4, got {received_b}"

        # Scenario C: last_event_id between progress and terminal event
        sm_c = StreamManager()
        channel_c = "reconnect-between-progress-and-terminal"

        await sm_c.publish(channel_c, "progress", {"step": 1})

        received_c: List[dict] = []
        # Reconnect with last_event_id="1"
        sub_gen_c = sm_c.subscribe(channel_c, last_event_id="1")

        async def _consume_c():
            async for evt in sub_gen_c:
                received_c.append(evt)

        consume_task_c = asyncio.create_task(_consume_c())
        await asyncio.sleep(0.01)

        await sm_c.publish(channel_c, "progress", {"step": 2})
        await sm_c.publish(channel_c, "completed", {"status": "success"})

        await asyncio.wait_for(consume_task_c, timeout=1.0)
        assert len(received_c) == 2
        assert received_c[0]["id"] == "2"
        assert received_c[1]["event"] == "completed"





@pytest.mark.asyncio
class TestTerminalStateImmutability:
    """Adversarial stress tests for complete_job and terminal state immutability."""

    async def test_complete_job_mutated_payload_attack_prevented(self):
        """
        Attempt to mutate an already SUCCEEDED job with malicious artifact URLs,
        altered completion timestamps, and fake telemetry via subsequent complete_job calls.
        Assert that original artifacts, timestamps, and telemetry are preserved verbatim.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        # 1. Submit and move to RUNNING
        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "original prompt"},
            priority=10,
        )
        job, _ = await manager.submit_job(sub_req, request_id=f"req-{uuid.uuid4()}")
        await manager.update_progress(
            job.job_id,
            ProgressUpdateRequest(stage="sampling", step=1, total=10, percent=10.0),
        )

        # 2. Legitimate initial completion
        valid_artifact = ArtifactReference(
            artifact_id="art-legit-001",
            mime_type="image/png",
            size_bytes=512000,
            storage_path="image/valid/output.png",
            download_url="https://s3.ai.example.com/image/valid/output.png",
            sha256="0123456789abcdef" * 4,
        )
        legit_req = JobCompletionRequest(
            artifacts=[valid_artifact],
            inference_duration_ms=2500.0,
            tokens_per_sec=None,
            fps=None,
            peak_vram_mb=7200.0,
        )

        legit_job = await manager.complete_job(job.job_id, legit_req)
        assert legit_job.state == JobState.SUCCEEDED
        original_completed_at = legit_job.completed_at
        assert original_completed_at is not None

        # 3. Adversarial subsequent completion attempt with malicious artifact URL and fake telemetry
        malicious_artifact = ArtifactReference(
            artifact_id="art-malicious-666",
            mime_type="application/x-executable",
            size_bytes=99999999,
            storage_path="image/evil/backdoor.sh",
            download_url="http://malicious-hacker.com/payload.sh",
            sha256="ffffffffffffffff" * 4,
        )
        malicious_req = JobCompletionRequest(
            artifacts=[malicious_artifact],
            inference_duration_ms=0.01,
            tokens_per_sec=99999.0,
            fps=60.0,
            peak_vram_mb=1.0,
        )

        attack_response = await manager.complete_job(job.job_id, malicious_req)

        # 4. Verify in-memory return value has not mutated
        assert attack_response.state == JobState.SUCCEEDED
        assert len(attack_response.artifacts) == 1
        assert attack_response.artifacts[0].artifact_id == "art-legit-001"
        assert attack_response.artifacts[0].download_url == "https://s3.ai.example.com/image/valid/output.png"
        assert attack_response.artifacts[0].storage_path == "image/valid/output.png"
        assert attack_response.telemetry.inference_duration_ms == 2500.0
        assert attack_response.telemetry.peak_vram_mb == 7200.0
        assert attack_response.completed_at == original_completed_at

        # 5. Verify database persistent record has not mutated
        persisted_job = await repo.get_job(job.job_id)
        assert persisted_job.state == JobState.SUCCEEDED
        assert len(persisted_job.artifacts) == 1
        assert persisted_job.artifacts[0].artifact_id == "art-legit-001"
        assert persisted_job.artifacts[0].download_url == "https://s3.ai.example.com/image/valid/output.png"
        assert persisted_job.telemetry.inference_duration_ms == 2500.0
        assert persisted_job.completed_at == original_completed_at

        await repo.close()

    async def test_complete_job_on_failed_or_cancelled_job_strictly_rejected(self):
        """
        Assert that calling complete_job on a FAILED or CANCELLED job raises an
        InvalidStateTransitionError and does NOT mutate the job into SUCCEEDED.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        req_id_1 = f"req-{uuid.uuid4()}"
        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "failing job"},
            priority=10,
        )
        job_fail, _ = await manager.submit_job(sub_req, request_id=req_id_1)
        await manager.fail_job(
            job_fail.job_id,
            JobFailureRequest(
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="Out of memory",
                    request_id=req_id_1,
                ),
                retryable=False,
            ),
        )

        dummy_artifact = ArtifactReference(
            artifact_id="art-test",
            mime_type="image/png",
            size_bytes=100,
            storage_path="path/1.png",
            download_url="https://s3/1.png",
            sha256="0" * 64,
        )
        comp_req = JobCompletionRequest(artifacts=[dummy_artifact], inference_duration_ms=100.0)

        # complete_job on FAILED job must raise InvalidStateTransitionError
        with pytest.raises(InvalidStateTransitionError):
            await manager.complete_job(job_fail.job_id, comp_req)

        persisted_failed = await repo.get_job(job_fail.job_id)
        assert persisted_failed.state == JobState.FAILED
        assert persisted_failed.artifacts == []

        # Setup CANCELLED job
        req_id_2 = f"req-{uuid.uuid4()}"
        job_cancel, _ = await manager.submit_job(sub_req, request_id=req_id_2)
        await manager.cancel_job(job_cancel.job_id)

        # complete_job on CANCELLED job must raise InvalidStateTransitionError
        with pytest.raises(InvalidStateTransitionError):
            await manager.complete_job(job_cancel.job_id, comp_req)

        persisted_cancelled = await repo.get_job(job_cancel.job_id)
        assert persisted_cancelled.state == JobState.CANCELLED
        assert persisted_cancelled.artifacts == []

        await repo.close()

    async def test_update_progress_on_terminal_jobs_is_noop(self):
        """
        Verify that update_progress on SUCCEEDED, FAILED, and CANCELLED jobs
        is a complete no-op and cannot modify progress percentage or stage.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test"},
            priority=10,
        )
        job, _ = await manager.submit_job(sub_req, request_id=f"req-{uuid.uuid4()}")
        await manager.update_progress(
            job.job_id,
            ProgressUpdateRequest(stage="sampling", step=10, total=20, percent=50.0),
        )
        art = ArtifactReference(
            artifact_id="art-done",
            mime_type="image/png",
            size_bytes=100,
            storage_path="p",
            download_url="d",
            sha256="0" * 64,
        )
        await manager.complete_job(job.job_id, JobCompletionRequest(artifacts=[art]))

        # Attempt to reset progress on completed job
        mutated = await manager.update_progress(
            job.job_id,
            ProgressUpdateRequest(stage="fake_stage", step=0, total=100, percent=0.0),
        )
        assert mutated.state == JobState.SUCCEEDED
        assert mutated.progress.percent == 100.0
        assert mutated.progress.stage == "completed"

        await repo.close()


@pytest.mark.asyncio
class TestBrokerEvictionAndWorkerConcurrency:
    """Adversarial stress tests for broker queue eviction and double-pickup prevention."""

    async def test_single_job_lease_eviction_from_broker(self):
        """
        Verify that when a job is leased, it is immediately and atomically removed
        from the active broker queue, preventing any other caller from popping it.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        job_id = f"job-{uuid.uuid4()}"
        req_id = f"req-{uuid.uuid4()}"
        job = FactoryJob(
            job_id=job_id,
            request_id=req_id,
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "eviction test"},
            priority=10,
            state=JobState.QUEUED,
        )
        await repo.create_job(job)
        await broker.enqueue(job.job_id, job.service, priority=job.priority)

        # Initial broker depth must be 1
        assert await broker.get_queue_depth("image") == 1

        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-001",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=120,
            )
        )
        assert lease_resp.job is not None
        assert lease_resp.job.job_id == job.job_id

        # Broker depth must immediately become 0
        assert await broker.get_queue_depth("image") == 0
        # Subsequent direct dequeue from broker must return None
        dequeued = await broker.dequeue("image")
        assert dequeued is None

        await repo.close()
        await broker.close()

    async def test_massive_worker_concurrency_race_zero_duplicate_leases(self):
        """
        Adversarial race condition:
        Enqueue 10 jobs. Spawn 40 concurrent workers simultaneously requesting leases.
        Assert that:
        1. Exactly 10 leases are granted across the 40 workers.
        2. All 10 leased jobs are distinct (ZERO duplicate pickups).
        3. The remaining 30 workers receive job=None.
        4. Broker queue depth is exactly 0 after all leases are granted.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        num_jobs = 10
        num_workers = 40

        for i in range(num_jobs):
            job = FactoryJob(
                job_id=f"job-{uuid.uuid4()}",
                request_id=f"req-{uuid.uuid4()}",
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": f"job {i}"},
                priority=10 + i,
                state=JobState.QUEUED,
            )
            await repo.create_job(job)
            await broker.enqueue(job.job_id, job.service, priority=job.priority)

        # Initial queue depth
        assert await broker.get_queue_depth("image") == num_jobs

        # Worker lease task
        async def _worker_attempt(worker_id: str) -> WorkerLeaseResponse:
            await asyncio.sleep(random.uniform(0.0001, 0.002))
            return await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=worker_id,
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                    lease_duration_seconds=300,
                )
            )

        worker_tasks = [
            asyncio.create_task(_worker_attempt(f"worker-{w:02d}"))
            for w in range(num_workers)
        ]

        responses: List[WorkerLeaseResponse] = await asyncio.gather(*worker_tasks)

        successful_leases = [r for r in responses if r.job is not None]
        unsuccessful_leases = [r for r in responses if r.job is None]

        # 1. Exactly num_jobs leases granted
        assert len(successful_leases) == num_jobs, (
            f"Expected {num_jobs} successful leases, got {len(successful_leases)}"
        )
        # 2. Exactly (num_workers - num_jobs) unsuccessful
        assert len(unsuccessful_leases) == (num_workers - num_jobs)

        # 3. All leased jobs have unique job_ids (zero duplicate pickups)
        leased_job_ids = [r.job.job_id for r in successful_leases]
        assert len(leased_job_ids) == len(set(leased_job_ids)), (
            f"Duplicate job leased: {leased_job_ids}"
        )

        # 4. Broker queue depth is 0
        final_depth = await broker.get_queue_depth("image")
        assert final_depth == 0, f"Broker depth expected 0, got {final_depth}"

        # 5. Direct dequeue returns None
        assert await broker.dequeue("image") is None

        await repo.close()
        await broker.close()

    async def test_cancelled_job_evicted_from_broker(self):
        """
        Verify that calling cancel_job immediately evicts the job from the broker,
        preventing any subsequent worker from leasing it.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        sub_req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "to be cancelled"},
            priority=10,
        )
        job, _ = await manager.submit_job(sub_req, request_id=f"req-{uuid.uuid4()}")
        assert await broker.get_queue_depth("image") == 1

        # Cancel job
        await manager.cancel_job(job.job_id)
        assert await broker.get_queue_depth("image") == 0

        # Worker attempts to lease -> should get None
        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-001",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=300,
            )
        )
        assert lease_resp.job is None

        await repo.close()
        await broker.close()

    async def test_reaper_auto_retry_re_enqueues_and_subsequent_lease_evicts(self):
        """
        Lifecycle cycle:
        1. Job is leased (broker depth -> 0).
        2. Worker dies; reaper sweeps expired lease, increments retry_count,
           re-enqueues to broker (broker depth -> 1).
        3. Second worker leases the retried job (broker depth -> 0).
        Assert exact queue depth transitions at each step.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        coordinator = LeaseCoordinator(repository=repo, broker=broker)
        reaper = DeadJobReaper(repository=repo, broker=broker, stream_manager=stream_mgr)

        job_id = f"job-{uuid.uuid4()}"
        req_id = f"req-{uuid.uuid4()}"
        job = FactoryJob(
            job_id=job_id,
            request_id=req_id,
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "retry cycle"},
            priority=15,
            state=JobState.QUEUED,
            retry_count=0,
            max_retries=2,
        )
        await repo.create_job(job)
        await broker.enqueue(job.job_id, job.service, priority=job.priority)
        assert await broker.get_queue_depth("image") == 1

        # Worker 1 leases job
        w1_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-crashed",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=30,  # Valid lease duration (schema requires >= 30)
            )
        )
        assert w1_resp.job is not None
        assert await broker.get_queue_depth("image") == 0

        # Simulate expiration by modifying lease_expires_at in DB
        expired_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        leased_job = await repo.get_job(job.job_id)
        leased_job.worker.lease_expires_at = expired_time
        await repo.update_job(leased_job)

        # Reaper sweeps
        recovered = await reaper.sweep_once()
        assert recovered == 1

        # Job is back to QUEUED, retry_count=1, and re-enqueued to broker
        reloaded = await repo.get_job(job.job_id)
        assert reloaded.state == JobState.QUEUED
        assert reloaded.retry_count == 1
        assert await broker.get_queue_depth("image") == 1

        # Worker 2 leases the retried job
        w2_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-healthy",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
                lease_duration_seconds=300,
            )
        )
        assert w2_resp.job is not None
        assert w2_resp.job.job_id == job.job_id
        assert w2_resp.job.worker.worker_id == "worker-healthy"

        # Broker depth is 0 again
        assert await broker.get_queue_depth("image") == 0

        await repo.close()
        await broker.close()

    async def test_100_concurrent_job_submissions_accepted_and_enqueued(self):
        """
        Adversarial test for R3 reliability:
        System must accept 100 concurrent job submissions without crashing, dropping requests,
        or creating duplicate job IDs. All 100 jobs must be enqueued into the broker.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        async def _submit(i: int):
            req = JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": f"concurrent prompt {i}"},
                priority=(i % 10) + 1,
            )
            return await manager.submit_job(req, request_id=f"req-{uuid.uuid4()}")

        results = await asyncio.gather(*[_submit(i) for i in range(100)])
        assert len(results) == 100
        assert all(created for _, created in results)
        job_ids = [j.job_id for j, _ in results]
        assert len(set(job_ids)) == 100
        assert await broker.get_queue_depth("image") == 100

        await repo.close()
        await broker.close()

    async def test_burst_worker_concurrency_heterogeneous_subsystems(self):
        """
        Adversarial test for high worker concurrency across multiple subsystems:
        10 image jobs and 10 video jobs enqueued.
        50 image workers and 50 video workers (100 total concurrent workers) request leases simultaneously.
        Assert:
        - Exactly 10 image leases and 10 video leases granted.
        - Exactly 40 image workers and 40 video workers receive None.
        - Broker queue depth is 0 for both image and video.
        - Zero duplicate job pickups.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        coordinator = LeaseCoordinator(repository=repo, broker=broker, candidate_limit=50)

        for i in range(10):
            j1 = FactoryJob(
                job_id=f"job-{uuid.uuid4()}",
                request_id=f"req-{uuid.uuid4()}",
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": f"img {i}"},
                priority=10,
                state=JobState.QUEUED,
            )
            await repo.create_job(j1)
            await broker.enqueue(j1.job_id, j1.service, priority=j1.priority)

        for i in range(10):
            j2 = FactoryJob(
                job_id=f"job-{uuid.uuid4()}",
                request_id=f"req-{uuid.uuid4()}",
                service="video",
                model="wan-2.2",
                task="text_to_video",
                inputs={"prompt": f"vid {i}"},
                priority=10,
                state=JobState.QUEUED,
            )
            await repo.create_job(j2)
            await broker.enqueue(j2.job_id, j2.service, priority=j2.priority)

        async def _worker(wid, sub, model, task):
            return await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=wid,
                    subsystem=sub,
                    supported_models=[model],
                    supported_tasks=[task],
                    gpu_device=0,
                    lease_duration_seconds=300,
                )
            )

        tasks = []
        for i in range(50):
            tasks.append(asyncio.create_task(_worker(f"w-img-{i}", "image", "qwen-image-2.1", "text_to_image")))
        for i in range(50):
            tasks.append(asyncio.create_task(_worker(f"w-vid-{i}", "video", "wan-2.2", "text_to_video")))

        responses = await asyncio.gather(*tasks)

        img_leases = [r for r in responses[:50] if r.job is not None]
        vid_leases = [r for r in responses[50:] if r.job is not None]

        assert len(img_leases) == 10
        assert len(vid_leases) == 10

        # Unique leases
        img_ids = [r.job.job_id for r in img_leases]
        vid_ids = [r.job.job_id for r in vid_leases]
        assert len(img_ids) == len(set(img_ids))
        assert len(vid_ids) == len(set(vid_ids))

        # Broker depths drained
        assert await broker.get_queue_depth("image") == 0
        assert await broker.get_queue_depth("video") == 0

        await repo.close()
        await broker.close()


