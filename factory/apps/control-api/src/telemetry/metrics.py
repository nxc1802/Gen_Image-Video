"""
factory/apps/control-api/src/telemetry/metrics.py
OpenTelemetry and in-memory metrics collectors for queue depth, active jobs, and VRAM.
"""
from typing import Any, Dict


class TelemetryMetrics:
    """In-memory telemetry metric container and OpenTelemetry gauge interface."""

    def __init__(self):
        self._active_jobs: int = 0
        self._queue_depth: int = 0
        self._gpu_0_vram_used_gb: float = 11.2
        self._gpu_1_vram_used_gb: float = 13.8
        self._requests_total: int = 0
        self._failures_total: int = 0

    def set_queue_depth(self, count: int) -> None:
        self._queue_depth = max(0, count)

    def set_active_jobs(self, count: int) -> None:
        self._active_jobs = max(0, count)

    def set_gpu_vram(self, gpu_0_gb: float, gpu_1_gb: float) -> None:
        self._gpu_0_vram_used_gb = gpu_0_gb
        self._gpu_1_vram_used_gb = gpu_1_gb

    def increment_requests(self) -> None:
        self._requests_total += 1

    def increment_failures(self) -> None:
        self._failures_total += 1

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "gpu_0_vram_used_gb": round(self._gpu_0_vram_used_gb, 1),
            "gpu_1_vram_used_gb": round(self._gpu_1_vram_used_gb, 1),
            "active_jobs": self._active_jobs,
            "queue_depth": self._queue_depth,
            "requests_total": self._requests_total,
            "failures_total": self._failures_total,
        }


metrics = TelemetryMetrics()
