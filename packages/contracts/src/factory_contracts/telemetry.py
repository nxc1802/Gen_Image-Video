"""
factory_contracts/telemetry.py
Genuine hardware and GPU telemetry sampling via NVML with explicit UNAVAILABLE reporting.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NVMLTelemetryCollector:
    """
    Samples genuine GPU hardware metrics (VRAM, utilization, temperature) via pynvml.

    If NVML or GPU is unavailable (CPU/Mac/non-NVIDIA environment), reports explicit
    telemetry_status="UNAVAILABLE" with None values. Never fabricates mock GPU telemetry.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._available = False
        self._handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if 0 <= self.device_index < device_count:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._available = True
            else:
                self._available = False
        except Exception:
            self._available = False
            self._handle = None

    def sample(self) -> Dict[str, Any]:
        """
        Samples hardware telemetry. Returns dictionary matching WorkerHeartbeatRequest fields:
        - telemetry_status: 'AVAILABLE' | 'UNAVAILABLE'
        - vram_used_mb: Optional[float]
        - gpu_utilization_percent: Optional[float]
        - temperature_c: Optional[float]
        """
        if not self._available or self._handle is None:
            self._init_nvml()
            if not self._available or self._handle is None:
                return {
                    "telemetry_status": "UNAVAILABLE",
                    "vram_used_mb": None,
                    "gpu_utilization_percent": None,
                    "temperature_c": None,
                }

        try:
            import pynvml

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            return {
                "telemetry_status": "AVAILABLE",
                "vram_used_mb": round(mem.used / (1024 * 1024), 2),
                "gpu_utilization_percent": float(util.gpu),
                "temperature_c": float(temp),
            }
        except Exception as e:
            logger.debug(f"Failed to query NVML device {self.device_index}: {e}")
            return {
                "telemetry_status": "UNAVAILABLE",
                "vram_used_mb": None,
                "gpu_utilization_percent": None,
                "temperature_c": None,
            }
