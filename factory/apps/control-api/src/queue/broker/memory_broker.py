"""
factory/apps/control-api/src/queue/broker/memory_broker.py
In-memory priority queue broker using asyncio and heapq.
"""
import asyncio
import heapq
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker


class MemoryQueueBroker(BaseQueueBroker):
    """In-memory priority queue and pub/sub broker for zero-dependency local dev and testing."""

    def __init__(self):
        # Service -> List[(-priority, timestamp, job_id)]
        self._queues: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
        self._active_jobs: Set[str] = set()
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        async with self._lock:
            self._queues.clear()
            self._active_jobs.clear()
            self._subscribers.clear()

    async def enqueue(self, job_id: str, service: str, priority: int = 10) -> None:
        async with self._lock:
            if job_id in self._active_jobs:
                return
            self._active_jobs.add(job_id)
            # Min-heap ordered by -priority (so higher priority comes first), then timestamp (FIFO)
            heapq.heappush(self._queues[service], (-priority, time.time(), job_id))

    async def dequeue(self, service: str) -> Optional[str]:
        async with self._lock:
            q = self._queues.get(service)
            while q:
                _, _, job_id = heapq.heappop(q)
                if job_id in self._active_jobs:
                    self._active_jobs.remove(job_id)
                    return job_id
            return None

    async def remove(self, job_id: str, service: str) -> bool:
        async with self._lock:
            if job_id in self._active_jobs:
                self._active_jobs.remove(job_id)
                # Filter out of heap
                if service in self._queues:
                    self._queues[service] = [item for item in self._queues[service] if item[2] != job_id]
                    heapq.heapify(self._queues[service])
                return True
            return False

    async def get_queue_depth(self, service: Optional[str] = None) -> int:
        async with self._lock:
            if service:
                return sum(1 for item in self._queues.get(service, []) if item[2] in self._active_jobs)
            return len(self._active_jobs)

    async def publish_event(self, channel: str, event_type: str, data: Dict[str, Any]) -> None:
        payload = {"event": event_type, "data": data, "timestamp": time.time()}
        if channel in self._subscribers:
            for q in list(self._subscribers[channel]):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def subscribe_events(self, channel: str) -> AsyncGenerator[Dict[str, Any], None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers[channel].add(q)
        try:
            while True:
                item = await q.get()
                yield item
        finally:
            async with self._lock:
                self._subscribers[channel].discard(q)
                if not self._subscribers[channel]:
                    self._subscribers.pop(channel, None)
