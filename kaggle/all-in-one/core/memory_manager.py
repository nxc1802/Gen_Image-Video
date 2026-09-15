"""
🧠 Core Memory Lifecycle Manager
Điều phối tài nguyên VRAM trên 2x Tesla T4 và 30GB CPU RAM.
Thực hiện PCIe Fast Swapping an toàn giữa VLM và FLUX/Wan2.1 mà không gây OOM.
"""

import gc
import logging
import threading
import time
from typing import Optional, Dict, Any
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryManager")


class MemoryManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MemoryManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.operation_lock = threading.Lock()
        
        # Registry mô hình đã nạp vào System RAM hoặc VRAM
        self.cached_models: Dict[str, Any] = {}
        
        # Model đang chiếm dụng GPU 1 (Visual/Dynamic Slot)
        self.active_gpu1_slot: Optional[str] = None
        
        logger.info("⚡ MemoryManager khởi tạo thành công.")

    @staticmethod
    def clean_gpu():
        """Giải phóng hoàn toàn bộ nhớ cache của tất cả GPU."""
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    @staticmethod
    def report_vram() -> Dict[str, Any]:
        """Báo cáo dung lượng VRAM thực tế trên từng GPU."""
        stats = {}
        if not torch.cuda.is_available():
            return {"device": "cpu", "status": "No CUDA detected"}
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            stats[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
            }
        return stats

    def switch_gpu1_slot(self, target_slot: str, loader_fn):
        """
        Hoán đổi mô hình trên GPU 1:
        Nếu target_slot chưa nạp vào GPU 1:
          1. Di chuyển / unload model cũ khỏi GPU 1.
          2. Dọn sạch cache VRAM.
          3. Nạp model mới (hoặc gọi loader_fn).
        """
        with self.operation_lock:
            if self.active_gpu1_slot == target_slot and target_slot in self.cached_models:
                logger.info(f"✨ Slot '{target_slot}' đã sẵn sàng trên GPU 1 (Zero latency)")
                return self.cached_models[target_slot]

            logger.info(f"🔄 Hoán đổi GPU 1: '{self.active_gpu1_slot}' ➔ '{target_slot}'...")
            t0 = time.time()

            # Bước 1: Giải phóng slot hiện tại nếu khác
            if self.active_gpu1_slot and self.active_gpu1_slot in self.cached_models:
                old_model = self.cached_models.pop(self.active_gpu1_slot, None)
                del old_model
                self.clean_gpu()

            # Bước 2: Nạp target slot
            self.clean_gpu()
            new_model = loader_fn()
            self.cached_models[target_slot] = new_model
            self.active_gpu1_slot = target_slot

            elapsed = time.time() - t0
            logger.info(f"✅ Hoàn tất kích hoạt '{target_slot}' trên GPU 1 trong {elapsed:.2f}s!")
            return new_model


def get_memory_manager() -> MemoryManager:
    return MemoryManager()
