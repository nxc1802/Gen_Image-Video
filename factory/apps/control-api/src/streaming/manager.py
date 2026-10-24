"""
factory/apps/control-api/src/streaming/manager.py
Unified pub/sub StreamManager with in-memory asyncio.Queue and optional Redis fallback.
"""
import asyncio
import json
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, Optional, Set

from factory.apps.control_api.src.streaming.history import EventHistoryBuffer, StreamEvent

TERMINAL_STREAM_EVENTS = ("completed", "failed", "cancelled", "error")


class StreamManager:
    """
    Multiplexes job progress, token generation, and audio chunks across workers and clients.
    Supports Redis pub/sub channels and in-memory asyncio.Queue fallback.
    """

    def __init__(self, redis_client=None, history_size: int = 100):
        self._redis = redis_client
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._history = EventHistoryBuffer(max_events_per_channel=history_size)
        self._terminated_channels: Set[str] = set()
        self._lock = asyncio.Lock()

    async def publish(self, channel: str, event_type: str, data: Dict[str, Any]) -> int:
        now = time.time()
        event_id = self._history.record(channel, event_type, data, now)

        if event_type in TERMINAL_STREAM_EVENTS:
            self._terminated_channels.add(channel)

        payload = {
            "id": str(event_id),
            "event": event_type,
            "data": data,
            "timestamp": now,
        }

        # In-memory distribution
        if channel in self._subscribers:
            for q in list(self._subscribers[channel]):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

        # Redis Pub/Sub distribution (if configured)
        if self._redis:
            try:
                await self._redis.publish(channel, json.dumps(payload))
            except Exception:
                pass

        return event_id

    async def subscribe(
        self,
        channel: str,
        last_event_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)

        # Track the highest event ID yielded to prevent duplicate delivery
        highest_event_id = -1
        if last_event_id:
            try:
                highest_event_id = int(str(last_event_id).strip())
            except (ValueError, TypeError):
                highest_event_id = -1

        # Atomically fetch history and register subscriber under lock
        async with self._lock:
            if last_event_id:
                historical = self._history.get_since(channel, last_event_id)
            else:
                historical = []
            self._subscribers[channel].add(q)

        try:
            # Replay historical events
            for h in historical:
                if h.event_id > highest_event_id:
                    highest_event_id = max(highest_event_id, h.event_id)
                    yield {
                        "id": str(h.event_id),
                        "event": h.event_type,
                        "data": h.data,
                        "timestamp": h.timestamp,
                    }
                    if h.event_type in TERMINAL_STREAM_EVENTS:
                        return

            # If channel was already terminated and queue is drained, do not enter live event loop
            if channel in self._terminated_channels and q.empty():
                return

            # Stream live events, de-duplicating any event already yielded in history
            while True:
                item = await q.get()
                raw_id = item.get("id")
                is_terminal = item.get("event") in TERMINAL_STREAM_EVENTS

                if raw_id is not None:
                    try:
                        eid = int(raw_id)
                        if eid <= highest_event_id:
                            if is_terminal:
                                break
                            continue  # Skip event already delivered during replay
                        highest_event_id = max(highest_event_id, eid)
                    except (ValueError, TypeError):
                        pass

                yield item
                # Terminal event check
                if is_terminal:
                    break
        finally:
            async with self._lock:
                self._subscribers[channel].discard(q)
                if not self._subscribers[channel]:
                    self._subscribers.pop(channel, None)

    def get_history(self, channel: str, last_event_id: Optional[str] = None):
        if last_event_id:
            return self._history.get_since(channel, last_event_id)
        return []
