"""
Durable job-event streaming for Factory v4.

Job lifecycle events are persisted through BaseJobRepository and distributed
through Redis when available. In-memory queues/history are development-only
fallbacks for non-durable local execution and non-job token/audio streams.
"""
import asyncio
import json
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, Optional, Set

from factory.apps.control_api.src.streaming.history import EventHistoryBuffer

TERMINAL_STREAM_EVENTS = ("completed", "failed", "cancelled", "error")
JOB_EVENT_PREFIX = "factory:events:job:"


class StreamManager:
    def __init__(
        self,
        redis_client=None,
        repository=None,
        history_size: int = 100,
    ):
        self._redis = redis_client
        self._repository = repository
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._history = EventHistoryBuffer(max_events_per_channel=history_size)
        self._terminated_channels: Set[str] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def _job_id_from_channel(channel: str) -> Optional[str]:
        if channel.startswith(JOB_EVENT_PREFIX):
            value = channel[len(JOB_EVENT_PREFIX):].strip()
            return value or None
        return None

    async def publish(self, channel: str, event_type: str, data: Dict[str, Any]) -> int:
        now = time.time()
        job_id = self._job_id_from_channel(channel)

        if job_id and self._repository is not None:
            event_id = await self._repository.record_event(
                job_id=job_id,
                event_type=event_type,
                data=data,
                stage=data.get("stage") if isinstance(data, dict) else None,
                percent=data.get("percent") if isinstance(data, dict) else None,
            )
        else:
            event_id = self._history.record(channel, event_type, data, now)

        if event_type in TERMINAL_STREAM_EVENTS:
            self._terminated_channels.add(channel)

        payload = {
            "id": str(event_id),
            "event": event_type,
            "data": data,
            "timestamp": now,
        }

        if self._redis is not None:
            await self._redis.publish(channel, json.dumps(payload))
        else:
            for q in list(self._subscribers.get(channel, ())):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

        return event_id

    async def _replay_persisted_job_events(
        self,
        channel: str,
        last_event_id: Optional[str],
    ) -> list[Dict[str, Any]]:
        job_id = self._job_id_from_channel(channel)
        if not job_id or self._repository is None:
            return []

        try:
            since = int(str(last_event_id).strip()) if last_event_id is not None else 0
        except (TypeError, ValueError):
            since = 0

        records = await self._repository.get_events_since(job_id, since)
        return [
            {
                "id": str(record["event_id"]),
                "event": record["event_type"],
                "data": record.get("data") or {},
                "timestamp": record.get("created_at"),
            }
            for record in records
        ]

    async def subscribe(
        self,
        channel: str,
        last_event_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            highest_event_id = int(str(last_event_id).strip()) if last_event_id else 0
        except (ValueError, TypeError):
            highest_event_id = 0

        pubsub = None
        if self._redis is not None:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)

        q: Optional[asyncio.Queue] = None
        if self._redis is None:
            q = asyncio.Queue(maxsize=512)
            async with self._lock:
                self._subscribers[channel].add(q)

        try:
            persisted = await self._replay_persisted_job_events(channel, last_event_id)
            if persisted:
                for item in persisted:
                    try:
                        eid = int(item["id"])
                    except (TypeError, ValueError):
                        eid = highest_event_id
                    if eid <= highest_event_id:
                        continue
                    highest_event_id = eid
                    yield item
                    if item.get("event") in TERMINAL_STREAM_EVENTS:
                        return
            elif self._repository is None and last_event_id:
                for h in self._history.get_since(channel, last_event_id):
                    if h.event_id <= highest_event_id:
                        continue
                    highest_event_id = h.event_id
                    item = {
                        "id": str(h.event_id),
                        "event": h.event_type,
                        "data": h.data,
                        "timestamp": h.timestamp,
                    }
                    yield item
                    if h.event_type in TERMINAL_STREAM_EVENTS:
                        return

            if self._redis is not None and pubsub is not None:
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    try:
                        item = json.loads(msg["data"])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue

                    raw_id = item.get("id")
                    if raw_id is not None:
                        try:
                            eid = int(raw_id)
                            if eid <= highest_event_id:
                                continue
                            highest_event_id = eid
                        except (TypeError, ValueError):
                            pass

                    yield item
                    if item.get("event") in TERMINAL_STREAM_EVENTS:
                        return
            else:
                if channel in self._terminated_channels and q is not None and q.empty():
                    return
                while q is not None:
                    item = await q.get()
                    raw_id = item.get("id")
                    if raw_id is not None:
                        try:
                            eid = int(raw_id)
                            if eid <= highest_event_id:
                                if item.get("event") in TERMINAL_STREAM_EVENTS:
                                    return
                                continue
                            highest_event_id = eid
                        except (TypeError, ValueError):
                            pass
                    yield item
                    if item.get("event") in TERMINAL_STREAM_EVENTS:
                        return
        finally:
            if pubsub is not None:
                try:
                    await pubsub.unsubscribe(channel)
                finally:
                    await pubsub.close()
            if q is not None:
                async with self._lock:
                    self._subscribers[channel].discard(q)
                    if not self._subscribers[channel]:
                        self._subscribers.pop(channel, None)

    def get_history(self, channel: str, last_event_id: Optional[str] = None):
        if last_event_id:
            return self._history.get_since(channel, last_event_id)
        return []
