"""
tests/adversarial/test_adversarial_m2_gate_probes.py
Adversarial Gate Probes for Milestone 2 Iteration 4 Gate.
Probing StreamManager event replay, queue draining, boundary conditions, and concurrency.
"""
import asyncio
import random
from typing import List, Optional
import pytest

from factory.apps.control_api.src.streaming.manager import (
    StreamManager,
    TERMINAL_STREAM_EVENTS,
)


@pytest.mark.asyncio
class TestStreamManagerBoundaryProbes:
    """Rigorous empirical adversarial boundary probes for StreamManager."""

    async def test_reconnect_terminated_channel_last_event_id_none(self):
        """
        Probe 1: Reconnecting to terminated channel with last_event_id=None.
        Must yield 0 events and terminate immediately without hanging.
        """
        sm = StreamManager(history_size=50)
        channel = "probe-ch-term-none"

        await sm.publish(channel, "progress", {"step": 1})
        await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        received = []
        async def _sub():
            async for evt in sm.subscribe(channel, last_event_id=None):
                received.append(evt)

        # Must finish within 0.5s
        await asyncio.wait_for(_sub(), timeout=0.5)
        assert len(received) == 0, f"Expected 0 events, got {received}"
        # Verify subscriber cleanup
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_reconnect_terminated_channel_last_event_id_zero(self):
        """
        Probe 2: Reconnecting with last_event_id='0' to a terminated channel.
        Must yield all historical events up to and including the terminal event,
        then terminate cleanly.
        """
        sm = StreamManager(history_size=50)
        channel = "probe-ch-term-zero"

        for i in range(1, 11):
            if i == 10:
                await sm.publish(channel, "completed", {"step": i})
            else:
                await sm.publish(channel, "progress", {"step": i})

        received = []
        async def _sub():
            async for evt in sm.subscribe(channel, last_event_id="0"):
                received.append(evt)

        await asyncio.wait_for(_sub(), timeout=1.0)
        assert len(received) == 10, f"Expected 10 events, got {len(received)}"
        event_ids = [int(e["id"]) for e in received]
        assert event_ids == list(range(1, 11))
        assert received[-1]["event"] == "completed"
        # Verify subscriber cleanup
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_reconnect_last_event_id_equal_to_terminal_id(self):
        """
        Probe 3: Reconnecting with last_event_id equal to terminal event ID.
        Must yield 0 events and terminate immediately without hanging.
        """
        sm = StreamManager(history_size=50)
        channel = "probe-ch-term-equal"

        await sm.publish(channel, "progress", {"step": 1})
        term_id = await sm.publish(channel, "failed", {"error": "fatal"})

        received = []
        async def _sub():
            async for evt in sm.subscribe(channel, last_event_id=str(term_id)):
                received.append(evt)

        await asyncio.wait_for(_sub(), timeout=0.5)
        assert len(received) == 0, f"Expected 0 events, got {received}"
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_burst_10_concurrent_subscribers_during_terminal_publication(self):
        """
        Probe 4: Rapid bursts of 10 concurrent subscribers reconnecting at staggered
        offsets while active publishers push progress and terminal events.
        All subscribers must receive strictly monotonic events without duplicates or hangs.
        """
        sm = StreamManager(history_size=200)
        channel = "probe-ch-burst-concurrent"

        # Seed 10 initial events
        for i in range(1, 11):
            await sm.publish(channel, "progress", {"seed": i})

        subscriber_results: List[List[dict]] = [[] for _ in range(10)]

        async def _subscriber(idx: int, last_id: Optional[str]):
            async for evt in sm.subscribe(channel, last_event_id=last_id):
                subscriber_results[idx].append(evt)
                # occasional micro-yield to trigger race conditions
                if random.random() < 0.3:
                    await asyncio.sleep(0.001)

        # Offsets: some with None, some with '0', some mid-stream
        offsets = [None, "0", "3", "5", "8", "10", "12", None, "0", "7"]
        sub_tasks = [
            asyncio.create_task(_subscriber(i, offsets[i]))
            for i in range(10)
        ]

        # Concurrently publish live events
        async def _publisher():
            for step in range(11, 25):
                await sm.publish(channel, "progress", {"step": step})
                await asyncio.sleep(0.002)
            await sm.publish(channel, "completed", {"status": "ALL_DONE"})

        pub_task = asyncio.create_task(_publisher())

        await asyncio.wait_for(asyncio.gather(pub_task, *sub_tasks), timeout=5.0)

        for idx, events in enumerate(subscriber_results):
            eids = [int(e["id"]) for e in events]
            # Must be strictly unique
            assert len(eids) == len(set(eids)), f"Subscriber {idx} received duplicate IDs: {eids}"
            # Must be sorted
            assert eids == sorted(eids), f"Subscriber {idx} received out-of-order IDs: {eids}"

            offset = offsets[idx]
            if offset is not None:
                min_expected = int(offset)
                assert all(eid > min_expected for eid in eids), (
                    f"Subscriber {idx} with offset {offset} received events <= offset: {eids}"
                )

            # If subscriber received events, the last event must be terminal 'completed'
            if len(events) > 0:
                assert events[-1]["event"] == "completed", (
                    f"Subscriber {idx} did not terminate on completed event: {events[-1]}"
                )

        # Cleanup check
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_large_queue_drain_slow_subscriber_race(self):
        """
        Probe 5: Adversarial queue draining with 50 live events queued during slow history replay.
        Verify all 50 events are delivered in order without drops or premature exit.
        """
        sm = StreamManager(history_size=100)
        channel = "probe-ch-large-drain"

        # Pre-seed 3 events
        await sm.publish(channel, "progress", {"step": 1})
        await sm.publish(channel, "progress", {"step": 2})
        await sm.publish(channel, "progress", {"step": 3})

        received = []
        started_event = asyncio.Event()

        async def _slow_subscriber():
            async for evt in sm.subscribe(channel, last_event_id="1"):
                received.append(evt)
                started_event.set()
                # Artificial delay on event 2 to allow live events to pile up in queue
                if int(evt["id"]) == 2:
                    await asyncio.sleep(0.05)

        sub_task = asyncio.create_task(_slow_subscriber())
        await started_event.wait()

        # While subscriber is slow-yielding event 2, publish 20 progress events + completed
        for s in range(4, 24):
            await sm.publish(channel, "progress", {"step": s})
        await sm.publish(channel, "completed", {"status": "DONE"})

        await asyncio.wait_for(sub_task, timeout=3.0)

        received_ids = [int(e["id"]) for e in received]
        expected_ids = list(range(2, 25))
        assert received_ids == expected_ids, (
            f"Queue drain failed! Expected {expected_ids}, got {received_ids}"
        )
        assert received[-1]["event"] == "completed"
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_subscriber_early_cancellation_cleans_up_registry(self):
        """
        Probe 6: Ensure cancelling a subscription mid-stream cleans up subscriber queue from memory.
        """
        from contextlib import aclosing

        sm = StreamManager(history_size=50)
        channel = "probe-ch-cancellation"

        # Case A: Cancelled while suspended waiting on q.get()
        async def _sub_waiting():
            async for _ in sm.subscribe(channel):
                pass

        task_a = asyncio.create_task(_sub_waiting())
        await asyncio.sleep(0.01)
        assert channel in sm._subscribers and len(sm._subscribers[channel]) == 1
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

        # Case B: Cancelled while consumer is processing a yielded event (using aclosing)
        await sm.publish(channel, "progress", {"step": 1})

        async def _sub_yielding():
            async with aclosing(sm.subscribe(channel, last_event_id="0")) as gen:
                async for _ in gen:
                    await asyncio.sleep(10.0)

        task_b = asyncio.create_task(_sub_yielding())
        await asyncio.sleep(0.01)
        assert channel in sm._subscribers and len(sm._subscribers[channel]) == 1
        task_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_b
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_high_throughput_reconnection_churn_stress(self):
        """
        Probe 7: High-throughput reconnection churn stress.
        Continuous publication of 150 events while 15 subscribers randomly connect, disconnect,
        and reconnect at varying historical offsets.
        Every subscriber that runs to completion must observe strictly monotonic IDs with zero duplicates.
        """
        sm = StreamManager(history_size=200)
        channel = "probe-ch-churn-stress"

        # Publisher task
        async def _publisher():
            for i in range(1, 151):
                if i == 150:
                    await sm.publish(channel, "completed", {"done": True})
                else:
                    await sm.publish(channel, "progress", {"step": i})
                if i % 25 == 0:
                    await asyncio.sleep(0.001)

        sub_errors = []

        async def _churn_client(cid: int):
            try:
                # Random starting point
                offset = str(random.randint(0, 50))
                received = []
                async for evt in sm.subscribe(channel, last_event_id=offset):
                    received.append(int(evt["id"]))
                    if evt["event"] == "completed":
                        break
                    if len(received) > 30 and random.random() < 0.2:
                        # Client disconnects early
                        break
                # Validate received events
                if received:
                    assert len(received) == len(set(received)), f"Client {cid} got duplicates: {received}"
                    assert received == sorted(received), f"Client {cid} got out-of-order: {received}"
                    assert all(eid > int(offset) for eid in received), f"Client {cid} got event <= offset: {received}"
            except Exception as e:
                sub_errors.append((cid, e))

        pub_task = asyncio.create_task(_publisher())
        clients = [asyncio.create_task(_churn_client(i)) for i in range(15)]

        await asyncio.wait_for(asyncio.gather(pub_task, *clients), timeout=5.0)
        assert len(sub_errors) == 0, f"Subscriber errors encountered: {sub_errors}"

