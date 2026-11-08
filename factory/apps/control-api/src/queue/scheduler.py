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
from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP, get_subsystem_gpu


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

    def acquire_gpus(self, gpu_devices: List[int], job_id: str) -> bool:
        """
        Atomically acquires exclusive execution leases on multiple GPU devices.
        If ANY requested device is busy, acquires NONE and returns False.
        Never leaves a dual-GPU job half-leased.
        """
        if not gpu_devices:
            return False
        # Pre-check all devices atomically
        for dev in gpu_devices:
            if self.is_gpu_busy(dev):
                return False
        # Acquire all requested devices
        for dev in gpu_devices:
            self._active_gpu_jobs[dev] = job_id
            self._gpu_busy[dev] = True
        return True

    def release_gpus(self, gpu_devices: Optional[List[int]], job_id: str) -> bool:
        """
        Releases multiple GPU devices held by job_id.
        """
        released_any = False
        target_devs = gpu_devices if gpu_devices is not None else list(self._active_gpu_jobs.keys())
        for dev in target_devs:
            if dev in self._active_gpu_jobs and self._active_gpu_jobs[dev] == job_id:
                self._active_gpu_jobs[dev] = None
                self._gpu_busy[dev] = False
                released_any = True
        # Safety sweep: clear any device tagged with job_id
        for dev, active_id in list(self._active_gpu_jobs.items()):
            if active_id == job_id:
                self._active_gpu_jobs[dev] = None
                self._gpu_busy[dev] = False
                released_any = True
        return released_any

    def acquire_gpu(self, gpu_device: int, job_id: str) -> bool:
        """
        Acquires exclusive execution lease on the specified GPU device.
        Enforces strict max_concurrent_services: 1 bound.
        """
        return self.acquire_gpus([gpu_device], job_id)

    def release_gpu(self, gpu_device: Optional[int], job_id: str) -> bool:
        """
        Releases GPU partition held by job_id.
        """
        if gpu_device is None:
            return self.release_gpus(None, job_id)
        return self.release_gpus([gpu_device], job_id)

    def set_gpu_busy(self, gpu_device: int, busy: bool) -> None:
        self._gpu_busy[gpu_device] = busy
        if not busy and self._active_gpu_jobs.get(gpu_device) is not None:
            self._active_gpu_jobs[gpu_device] = None

    def is_gpu_busy(self, gpu_device: int) -> bool:
        return self._gpu_busy.get(gpu_device, False) or (self._active_gpu_jobs.get(gpu_device) is not None)


__all__ = ["InMemoryScheduler"]
