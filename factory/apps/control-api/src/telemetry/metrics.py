"""
factory/apps/control-api/src/telemetry/metrics.py
OpenTelemetry and in-memory metrics collectors for queue depth, active jobs, and VRAM.
"""
from typing import Any, Dict, Optional


class TelemetryMetrics:
    """In-memory telemetry metric container and OpenTelemetry gauge interface."""

    def __init__(self):
        self._active_jobs: int = 0
        self._queue_depth: int = 0
        self._gpu_0_vram_used_gb: Optional[float] = None
        self._gpu_1_vram_used_gb: Optional[float] = None
        self._gpu_0_vram_total_gb: Optional[float] = None
        self._gpu_1_vram_total_gb: Optional[float] = None
        self._gpu_0_process_vram_gb: Optional[float] = None
        self._gpu_1_process_vram_gb: Optional[float] = None
        self._gpu_0_name: Optional[str] = None
        self._gpu_1_name: Optional[str] = None
        self._telemetry_status: str = "UNAVAILABLE"
        self._requests_total: int = 0
        self._failures_total: int = 0

    def set_queue_depth(self, count: int) -> None:
        self._queue_depth = max(0, count)

    def set_active_jobs(self, count: int) -> None:
        self._active_jobs = max(0, count)

    def set_gpu_vram(self, gpu_0_gb: Optional[float] = None, gpu_1_gb: Optional[float] = None) -> None:
        if gpu_0_gb is not None:
            self._gpu_0_vram_used_gb = gpu_0_gb
        if gpu_1_gb is not None:
            self._gpu_1_vram_used_gb = gpu_1_gb

    def set_gpu_totals(self, gpu_0_total_gb: Optional[float] = None, gpu_1_total_gb: Optional[float] = None) -> None:
        if gpu_0_total_gb is not None:
            self._gpu_0_vram_total_gb = gpu_0_total_gb
        if gpu_1_total_gb is not None:
            self._gpu_1_vram_total_gb = gpu_1_total_gb

    def set_gpu_process_vram(self, gpu_0_proc_gb: Optional[float] = None, gpu_1_proc_gb: Optional[float] = None) -> None:
        if gpu_0_proc_gb is not None:
            self._gpu_0_process_vram_gb = gpu_0_proc_gb
        if gpu_1_proc_gb is not None:
            self._gpu_1_process_vram_gb = gpu_1_proc_gb

    def set_gpu_names(self, gpu_0_name: Optional[str] = None, gpu_1_name: Optional[str] = None) -> None:
        if gpu_0_name is not None:
            self._gpu_0_name = gpu_0_name
        if gpu_1_name is not None:
            self._gpu_1_name = gpu_1_name

    def set_telemetry_status(self, status: str) -> None:
        self._telemetry_status = status

    def reset(self) -> None:
        self._active_jobs = 0
        self._queue_depth = 0
        self._gpu_0_vram_used_gb = None
        self._gpu_1_vram_used_gb = None
        self._gpu_0_vram_total_gb = None
        self._gpu_1_vram_total_gb = None
        self._gpu_0_process_vram_gb = None
        self._gpu_1_process_vram_gb = None
        self._gpu_0_name = None
        self._gpu_1_name = None
        self._telemetry_status = "UNAVAILABLE"
        self._requests_total = 0
        self._failures_total = 0

    def increment_requests(self) -> None:
        self._requests_total += 1

    def increment_failures(self) -> None:
        self._failures_total += 1

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "gpu_0_vram_used_gb": round(self._gpu_0_vram_used_gb, 1) if self._gpu_0_vram_used_gb is not None else None,
            "gpu_1_vram_used_gb": round(self._gpu_1_vram_used_gb, 1) if self._gpu_1_vram_used_gb is not None else None,
            "gpu_0_vram_total_gb": round(self._gpu_0_vram_total_gb, 1) if self._gpu_0_vram_total_gb is not None else None,
            "gpu_1_vram_total_gb": round(self._gpu_1_vram_total_gb, 1) if self._gpu_1_vram_total_gb is not None else None,
            "gpu_0_process_vram_gb": round(self._gpu_0_process_vram_gb, 2) if self._gpu_0_process_vram_gb is not None else None,
            "gpu_1_process_vram_gb": round(self._gpu_1_process_vram_gb, 2) if self._gpu_1_process_vram_gb is not None else None,
            "gpu_0_name": self._gpu_0_name,
            "gpu_1_name": self._gpu_1_name,
            "telemetry_status": self._telemetry_status,
            "active_jobs": self._active_jobs,
            "queue_depth": self._queue_depth,
            "requests_total": self._requests_total,
            "failures_total": self._failures_total,
        }


metrics = TelemetryMetrics()
