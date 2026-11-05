"""
factory_contracts/telemetry.py
Genuine hardware and GPU telemetry sampling via NVML with explicit UNAVAILABLE reporting.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NVMLTelemetryCollector:
    """
    Samples genuine GPU hardware metrics (VRAM, utilization, temperature, running processes) via pynvml.

    If NVML or GPU is unavailable (CPU/Mac/non-NVIDIA environment), reports explicit
    telemetry_status="UNAVAILABLE" with None values. Never fabricates mock GPU telemetry.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._available = False
        self._handle = None
        self._device_count = 0
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            if 0 <= self.device_index < self._device_count:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._available = True
            else:
                self._handle = None
                self._available = False
        except Exception:
            self._available = False
            self._handle = None
            self._device_count = 0

    @property
    def is_available(self) -> bool:
        return self._available and self._handle is not None

    def sample_device(self, index: int, target_pid: Optional[int] = None) -> Dict[str, Any]:
        """Samples telemetry for a specific GPU device index."""
        if not self._available:
            self._init_nvml()
        if not self._available or index < 0 or index >= self._device_count:
            return {
                "telemetry_status": "UNAVAILABLE",
                "index": index,
                "name": None,
                "total_mb": None,
                "total_gb": None,
                "free_mb": None,
                "free_gb": None,
                "used_mb": None,
                "system_vram_used_mb": None,
                "process_vram_used_mb": None,
                "running_processes": [],
                "gpu_utilization_percent": None,
                "temperature_c": None,
            }
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            raw_procs = []
            try:
                raw_procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            except Exception:
                pass
            try:
                raw_procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            except Exception:
                pass

            running_procs = []
            seen_pids = set()
            proc_bytes_total = 0
            target_pid_bytes = 0
            for p in raw_procs:
                pid = getattr(p, "pid", None)
                if pid is None or pid in seen_pids:
                    continue
                seen_pids.add(pid)
                used_b = getattr(p, "usedGpuMemory", 0) or 0
                used_mb = round(used_b / (1024 * 1024), 2)
                running_procs.append({"pid": pid, "used_mb": used_mb})
                proc_bytes_total += used_b
                if target_pid is not None and pid == target_pid:
                    target_pid_bytes += used_b

            proc_mb = round((target_pid_bytes if target_pid is not None else proc_bytes_total) / (1024 * 1024), 2)
            sys_used_mb = round(mem.used / (1024 * 1024), 2)
            total_mb = round(mem.total / (1024 * 1024), 2)
            free_mb = round(mem.free / (1024 * 1024), 2)

            return {
                "telemetry_status": "AVAILABLE",
                "index": index,
                "name": name,
                "total_mb": total_mb,
                "total_gb": round(total_mb / 1024.0, 2),
                "free_mb": free_mb,
                "free_gb": round(free_mb / 1024.0, 2),
                "used_mb": sys_used_mb,
                "system_vram_used_mb": sys_used_mb,
                "process_vram_used_mb": proc_mb,
                "running_processes": running_procs,
                "gpu_utilization_percent": float(util.gpu),
                "temperature_c": float(temp),
            }
        except Exception as e:
            logger.debug(f"Failed to query NVML device {index}: {e}")
            return {
                "telemetry_status": "UNAVAILABLE",
                "index": index,
                "name": None,
                "total_mb": None,
                "total_gb": None,
                "free_mb": None,
                "free_gb": None,
                "used_mb": None,
                "system_vram_used_mb": None,
                "process_vram_used_mb": None,
                "running_processes": [],
                "gpu_utilization_percent": None,
                "temperature_c": None,
            }

    def sample_all_devices(self, target_pid: Optional[int] = None) -> List[Dict[str, Any]]:
        """Samples telemetry across all detected NVML devices."""
        if not self._available:
            self._init_nvml()
        if not self._available or self._device_count == 0:
            return []
        return [self.sample_device(i, target_pid=target_pid) for i in range(self._device_count)]

    def sample(self) -> Dict[str, Any]:
        """
        Samples hardware telemetry for default device. Returns dictionary matching WorkerHeartbeatRequest fields:
        - telemetry_status: 'AVAILABLE' | 'UNAVAILABLE'
        - vram_used_mb: Optional[float]
        - system_vram_used_mb: Optional[float]
        - process_vram_used_mb: Optional[float]
        - vram_total_mb: Optional[float]
        - gpu_utilization_percent: Optional[float]
        - temperature_c: Optional[float]
        """
        dev = self.sample_device(self.device_index)
        if dev["telemetry_status"] == "UNAVAILABLE":
            return {
                "telemetry_status": "UNAVAILABLE",
                "vram_used_mb": None,
                "system_vram_used_mb": None,
                "process_vram_used_mb": None,
                "vram_total_mb": None,
                "gpu_utilization_percent": None,
                "temperature_c": None,
            }
        return {
            "telemetry_status": "AVAILABLE",
            "vram_used_mb": dev["used_mb"],
            "system_vram_used_mb": dev["system_vram_used_mb"],
            "process_vram_used_mb": dev["process_vram_used_mb"],
            "vram_total_mb": dev["total_mb"],
            "gpu_utilization_percent": dev["gpu_utilization_percent"],
            "temperature_c": dev["temperature_c"],
        }
