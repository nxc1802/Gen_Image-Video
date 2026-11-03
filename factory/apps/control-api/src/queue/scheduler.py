"""
factory/apps/control-api/src/queue/scheduler.py
Single Control Plane in-memory priority scheduler and event dispatcher for Factory v4 Lite.
"""
import asyncio
import heapq
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker


class InMemoryScheduler(MemoryQueueBroker):
    """
    In-memory queue scheduler and pub/sub event dispatcher.
    - Manages priority queues per subsystem service.
    - Supports discrete Dual T4 GPU partition queues (GPU 0 vs GPU 1).
    - Thread-safe and coroutine-safe with asyncio.Lock.
    - Zero Redis dependencies.
    """

    def __init__(self):
        super().__init__()
        # GPU Partition Queues: gpu_device (0 or 1) -> List[(-priority, timestamp, job_id)]
        self._gpu_queues: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
        self._gpu_busy: Dict[int, bool] = {0: False, 1: False}
        self._active_gpu_jobs: Dict[int, Optional[str]] = {0: None, 1: None}

    async def close(self) -> None:
        await super().close()
        async with self._lock:
            self._gpu_queues.clear()
            self._active_gpu_jobs = {0: None, 1: None}
            self._gpu_busy = {0: False, 1: False}

    async def enqueue_for_gpu(
        self,
        job_id: str,
        service: str,
        gpu_device: int,
        priority: int = 10,
    ) -> None:
        """Enqueues job to both service queue and GPU partition queue."""
        await self.enqueue(job_id=job_id, service=service, priority=priority)
        async with self._lock:
            heapq.heappush(self._gpu_queues[gpu_device], (-priority, time.time(), job_id))

    async def dequeue_for_gpu(self, gpu_device: int) -> Optional[str]:
        """Pulls next highest-priority job targeted for specific GPU partition if not busy."""
        if self.is_gpu_busy(gpu_device):
            return None
        async with self._lock:
            if self.is_gpu_busy(gpu_device):
                return None
            q = self._gpu_queues.get(gpu_device)
            while q:
                _, _, job_id = heapq.heappop(q)
                if job_id in self._active_jobs:
                    self._active_jobs.remove(job_id)
                    return job_id
            return None

    def acquire_gpu(self, gpu_device: int, job_id: str) -> bool:
        """
        Acquires exclusive execution lease on the specified GPU device.
        Enforces strict max_concurrent_services: 1 bound.
        """
        if self._active_gpu_jobs.get(gpu_device) is not None:
            return False
        self._active_gpu_jobs[gpu_device] = job_id
        self._gpu_busy[gpu_device] = True
        return True

    def release_gpu(self, gpu_device: Optional[int], job_id: str) -> bool:
        """
        Releases GPU partition held by job_id.
        """
        if gpu_device is not None and gpu_device in self._active_gpu_jobs:
            if self._active_gpu_jobs[gpu_device] == job_id:
                self._active_gpu_jobs[gpu_device] = None
                self._gpu_busy[gpu_device] = False
                return True
        for dev, active_id in list(self._active_gpu_jobs.items()):
            if active_id == job_id:
                self._active_gpu_jobs[dev] = None
                self._gpu_busy[dev] = False
                return True
        return False

    def set_gpu_busy(self, gpu_device: int, busy: bool) -> None:
        self._gpu_busy[gpu_device] = busy
        if not busy and self._active_gpu_jobs.get(gpu_device) is not None:
            self._active_gpu_jobs[gpu_device] = None

    def is_gpu_busy(self, gpu_device: int) -> bool:
        return self._gpu_busy.get(gpu_device, False) or (self._active_gpu_jobs.get(gpu_device) is not None)


__all__ = ["InMemoryScheduler"]
