"""
tests/adversarial/test_adversarial_m2_reconnection_churn.py
Empirical stress tests for concurrent reconnection churn on StreamManager.subscribe.
Validates clean termination, monotonicity, deduplication, and zero hangs across all last_event_id boundaries.
"""
import asyncio
import random
from typing import Any, Dict, List, Optional
import pytest

from factory.apps.control_api.src.streaming.manager import (
    StreamManager,
    TERMINAL_STREAM_EVENTS,
)


@pytest.mark.asyncio
class TestReconnectionChurnStress:
    """Rigorous stress suite for concurrent reconnection churn and boundary offsets."""

    @pytest.mark.parametrize(
        "offset_desc,offset_val",
        [
            ("negative_large", "-99999"),
            ("negative_one", "-1"),
            ("zero", "0"),
            ("empty_string", ""),
            ("whitespace", "   "),
            ("alphanumeric", "invalid_event_99"),
            ("special_chars", "@#$%^&*()"),
            ("exact_terminal", "TERMINAL"),  # resolved dynamically
            ("future_small", "FUTURE_PLUS_1"),  # resolved dynamically
            ("future_large", "99999999"),
        ],
    )
    async def test_reconnect_boundary_offsets_on_preterminated_channel(
        self, offset_desc: str, offset_val: str
    ):
        """
        Tests reconnecting to an already terminated channel with various boundary offsets.
        All terminal types (completed, failed, cancelled, error) must terminate cleanly without hanging.
        """
        for term_type in TERMINAL_STREAM_EVENTS:
            sm = StreamManager(history_size=100)
            channel = f"preterm-{offset_desc}-{term_type}"

            # Pre-populate 10 events, ending with term_type
            for step in range(1, 10):
                await sm.publish(channel, "progress", {"step": step})
            term_id = await sm.publish(channel, term_type, {"status": "finished"})
            assert term_id == 10

            # Resolve dynamic offsets
            resolved_offset: Optional[str]
            if offset_val == "TERMINAL":
                resolved_offset = str(term_id)
            elif offset_val == "FUTURE_PLUS_1":
                resolved_offset = str(term_id + 1)
            elif offset_val == "":
                resolved_offset = ""
            else:
                resolved_offset = offset_val

            received: List[Dict[str, Any]] = []

            async def _consume():
                async for evt in sm.subscribe(channel, last_event_id=resolved_offset):
                    received.append(evt)

            # Execution must finish in < 0.5s without any hang
            await asyncio.wait_for(_consume(), timeout=0.5)

            # Verification of received events
            if resolved_offset in ("-99999", "-1", "0"):
                # Should receive all 10 events
                assert len(received) == 10, f"Expected 10 events for offset {resolved_offset}, got {len(received)}"
                assert [int(e["id"]) for e in received] == list(range(1, 11))
                assert received[-1]["event"] == term_type
            elif resolved_offset in (str(term_id), str(term_id + 1), "99999999"):
                # Already at or past terminal, should receive 0 events
                assert len(received) == 0, f"Expected 0 events for offset {resolved_offset}, got {len(received)}"
            elif resolved_offset in ("", "   ", "invalid_event_99", "@#$%^&*()"):
                # Invalid offset returns empty historical events, and channel is already terminated -> 0 events
                assert len(received) == 0, f"Expected 0 events for invalid offset {resolved_offset}, got {len(received)}"

            # Confirm cleanup
            assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_reconnect_boundary_offsets_during_live_terminal_publication(self):
        """
        Subscribers connect with various boundary offsets BEFORE or AS the terminal event is published.
        Verify that in 100% of cases, every subscriber terminates cleanly without hanging.
        """
        sm = StreamManager(history_size=100)
        channel = "live-term-boundaries"

        # Pre-populate 5 progress events
        for step in range(1, 6):
            await sm.publish(channel, "progress", {"step": step})

        offsets = [
            "-500",
            "-1",
            "0",
            "3",
            "5",
            "6",   # equal to upcoming terminal event id (6)
            "7",   # future +1
            "100", # future large
            "xyz", # invalid
        ]

        received_by_offset: Dict[str, List[Dict[str, Any]]] = {off: [] for off in offsets}

        async def _subscriber(off: str):
            async for evt in sm.subscribe(channel, last_event_id=off):
                received_by_offset[off].append(evt)

        tasks = [asyncio.create_task(_subscriber(off)) for off in offsets]
        await asyncio.sleep(0.01)

        # Publish terminal event (id 6)
        term_id = await sm.publish(channel, "completed", {"status": "SUCCESS"})
        assert term_id == 6

        # Wait for all tasks to complete cleanly
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        # Verify each offset behavior
        for off, evts in received_by_offset.items():
            eids = [int(e["id"]) for e in evts]
            # No duplicates
            assert len(eids) == len(set(eids)), f"Offset {off} had duplicates: {eids}"
            # Strictly sorted
            assert eids == sorted(eids), f"Offset {off} had unsorted IDs: {eids}"

            if off in ("-500", "-1", "0"):
                # All 6 events
                assert eids == list(range(1, 7)), f"Offset {off} expected 1..6, got {eids}"
                assert evts[-1]["event"] == "completed"
            elif off == "3":
                # Events 4, 5, 6
                assert eids == [4, 5, 6], f"Offset 3 expected [4,5,6], got {eids}"
                assert evts[-1]["event"] == "completed"
            elif off == "5":
                # Only terminal event 6
                assert eids == [6], f"Offset 5 expected [6], got {eids}"
                assert evts[-1]["event"] == "completed"
            elif off in ("6", "7", "100"):
                # At or past terminal event id: must receive 0 events and terminate cleanly
                assert eids == [], f"Offset {off} expected 0 events, got {eids}"
            elif off == "xyz":
                # Invalid offset -> no historical, but live terminal event arrives (id 6)
                assert eids == [6], f"Offset xyz expected live terminal [6], got {eids}"
                assert evts[-1]["event"] == "completed"

        # Confirm cleanup
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_high_concurrency_reconnection_churn_with_random_unsubscribes(self):
        """
        Stress test: 40 concurrent subscribers continuously connecting, reading,
        and reconnecting with random offsets (negative, 0, mid-stream, future, terminal)
        while 2 publishers emit 100 total events concurrently.
        """
        sm = StreamManager(history_size=200)
        channel = "churn-mass-concurrency"

        active_clients = 40
        events_to_publish = 100
        client_stats = {"clean_exits": 0, "premature_cancels": 0, "errors": []}

        async def _client_churn_worker(cid: int):
            try:
                # Pick a random offset
                rand_choice = random.choice([
                    "-10", "0", "15", "50", "80", "99", "100", "500", "invalid"
                ])
                received = []
                # Randomly either run to completion or disconnect early
                early_disconnect_target = (
                    random.randint(5, 20) if random.random() < 0.25 else None
                )

                async for evt in sm.subscribe(channel, last_event_id=rand_choice):
                    received.append(evt)
                    if early_disconnect_target and len(received) >= early_disconnect_target:
                        client_stats["premature_cancels"] += 1
                        break
                    if evt.get("event") in TERMINAL_STREAM_EVENTS:
                        break

                # Invariant checks for received events
                eids = [int(e["id"]) for e in received if "id" in e and e["id"].isdigit()]
                assert len(eids) == len(set(eids)), f"Client {cid} got duplicates: {eids}"
                assert eids == sorted(eids), f"Client {cid} got out-of-order IDs: {eids}"

                try:
                    num_offset = int(rand_choice)
                    for eid in eids:
                        assert eid > num_offset, f"Client {cid} received event {eid} <= offset {num_offset}"
                except ValueError:
                    pass

                client_stats["clean_exits"] += 1
            except Exception as exc:
                client_stats["errors"].append((cid, repr(exc)))

        # Launch publishers
        async def _publisher(start_idx: int, count: int, is_final: bool):
            for i in range(start_idx, start_idx + count):
                await sm.publish(channel, "progress", {"step": i})
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            if is_final:
                await sm.publish(channel, "completed", {"status": "ALL_DONE"})

        pub1 = asyncio.create_task(_publisher(1, 50, False))
        pub2 = asyncio.create_task(_publisher(51, 49, True))  # publishes terminal at step 100

        clients = [
            asyncio.create_task(_client_churn_worker(i))
            for i in range(active_clients)
        ]

        # Everything must finish within 6.0 seconds
        await asyncio.wait_for(asyncio.gather(pub1, pub2, *clients), timeout=6.0)

        assert len(client_stats["errors"]) == 0, f"Client errors found: {client_stats['errors']}"
        assert client_stats["clean_exits"] == active_clients, (
            f"Expected {active_clients} clean exits, got {client_stats['clean_exits']}"
        )
        # Ensure no lingering subscribers
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_reconnection_lock_contention_and_queue_saturation(self):
        """
        Stress test: 50 subscribers all hitting subscribe() at the exact same millisecond
        while publish() emits events in rapid succession.
        Validates lock atomicity, absence of race conditions, and clean shutdown.
        """
        sm = StreamManager(history_size=50)
        channel = "churn-lock-contention"

        start_barrier = asyncio.Event()
        subscribers_done = []

        async def _racing_subscriber(idx: int):
            await start_barrier.wait()
            # Random offset
            offset = str(idx % 10)
            count = 0
            async for evt in sm.subscribe(channel, last_event_id=offset):
                count += 1
                if evt.get("event") in TERMINAL_STREAM_EVENTS:
                    break
            subscribers_done.append((idx, count))

        sub_tasks = [asyncio.create_task(_racing_subscriber(i)) for i in range(50)]

        # Trigger barrier
        start_barrier.set()

        # Simultaneously publish
        for s in range(1, 20):
            await sm.publish(channel, "progress", {"s": s})
            await asyncio.sleep(0.0005)

        await sm.publish(channel, "failed", {"error": "contention_test_done"})

        # Wait for all 50 racing subscribers to complete
        await asyncio.wait_for(asyncio.gather(*sub_tasks), timeout=4.0)

        assert len(subscribers_done) == 50
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    @pytest.mark.parametrize(
        "hostile_id",
        [
            "+10",             # leading plus sign
            "  10  ",          # whitespace padded
            "\t10\n",          # tab/newline padded
            "10.5",            # float string
            "0x10",            # hex string
            "1e5",             # exponential notation
            "99999999999999999999999999999999", # huge integer (arbitrary precision in Python)
            "-99999999999999999999999999999999",
            "NaN",
            "Infinity",
            "\x00\x01\x02",   # null bytes
        ]
    )
    async def test_hostile_last_event_id_formats(self, hostile_id: str):
        """
        Validates that bizarre, hostile, or non-standard last_event_id strings
        never crash StreamManager or cause unhandled exceptions or hangs.
        """
        sm = StreamManager(history_size=20)
        channel = f"hostile-id-{hash(hostile_id) & 0xffff}"

        await sm.publish(channel, "progress", {"step": 1})
        await sm.publish(channel, "completed", {"status": "ok"})

        received = []
        async def _sub():
            async for evt in sm.subscribe(channel, last_event_id=hostile_id):
                received.append(evt)

        await asyncio.wait_for(_sub(), timeout=0.5)
        # Verify clean termination
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

    async def test_slow_consumer_backpressure_with_terminal_break(self):
        """
        Subscriber takes time processing each event while publisher publishes fast.
        When terminal event arrives, subscriber must eventually reach it and terminate cleanly.
        """
        sm = StreamManager(history_size=50)
        channel = "slow-consumer-test"

        for s in range(1, 10):
            await sm.publish(channel, "progress", {"step": s})
        await sm.publish(channel, "cancelled", {"reason": "user_abort"})

        received = []
        async def _slow_sub():
            async for evt in sm.subscribe(channel, last_event_id="0"):
                received.append(evt)
                await asyncio.sleep(0.005)  # 5ms per event

        await asyncio.wait_for(_slow_sub(), timeout=2.0)
        assert len(received) == 10
        assert received[-1]["event"] == "cancelled"
        assert channel not in sm._subscribers or len(sm._subscribers[channel]) == 0

