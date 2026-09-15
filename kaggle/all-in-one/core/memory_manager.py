"""
🧠 Core Memory Lifecycle Manager: Unified RAM ↔ VRAM PCIe Orchestrator
Điều phối tài nguyên VRAM và bộ nhớ hệ thống theo chính sách vòng đời (Lifecycle):
1. always_active: Mô hình thường trực 100% trong VRAM (Whisper STT, Kokoro TTS, VLM 7B...).
   - Tuyệt đối không bị giải phóng hay hoán đổi khi các mô hình khác hoạt động.
2. dynamic_switch: Mô hình hoán đổi qua bus PCIe tốc độ cao (12-14 GB/s) giữa CPU RAM và GPU VRAM (FLUX.1, Wan2.1...).
   - Giữ nguyên instance trong CPU RAM (Zero Disk I/O, không bao giờ đọc lại từ SSD).
3. preload & init_target:
   - preload: true | false (Nạp ngay khi server khởi động hay đợi request).
   - init_target: "gpu" | "cpu" (Nạp thẳng vào GPU VRAM hay đỗ sẵn trong CPU RAM).
"""

import gc
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set
import torch

from config import MODELS_CONFIG

logger = logging.getLogger("MemoryOrchestrator")


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

        # 📌 Nhóm 1: Mô hình Thường Trực (Always Active) - Luôn nằm cố định trong GPU VRAM
        self.always_active_models: Dict[str, Any] = {}

        # 🔄 Nhóm 2: Mô hình Hoán Đổi Động (Dynamic Switch Pool) - Đỗ trong CPU RAM, kích hoạt lên GPU khi cần
        self.dynamic_models: Dict[str, Any] = {}

        # Tên mô hình động đang chiếm dụng slot GPU tính toán
        self.active_dynamic_slot: Optional[str] = None

        logger.info("⚡ Unified Memory Lifecycle Orchestrator đã sẵn sàng (RAM ↔ VRAM PCIe Bus).")

    @staticmethod
    def clean_gpu():
        """Giải phóng bộ nhớ đệm cache và phân mảnh CUDA trên tất cả GPU (không chạm vào weights đang dùng)."""
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    @staticmethod
    def get_system_ram() -> Dict[str, Any]:
        """Đọc dung lượng CPU System RAM theo thời gian thực."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "used_gb": round(mem.used / (1024 ** 3), 2),
                "percent": mem.percent,
            }
        except Exception:
            try:
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                info = {line.split(":")[0].strip(): int(line.split(":")[1].split()[0]) for line in lines if ":" in line}
                tot = info.get("MemTotal", 0)
                avail = info.get("MemAvailable", info.get("MemFree", 0))
                return {
                    "total_gb": round(tot / (1024 ** 2), 2),
                    "available_gb": round(avail / (1024 ** 2), 2),
                    "used_gb": round((tot - avail) / (1024 ** 2), 2),
                    "percent": round(((tot - avail) / max(tot, 1)) * 100, 1),
                }
            except Exception:
                return {"status": "RAM info unavailable"}

    def report_vram(self) -> Dict[str, Any]:
        """Báo cáo dung lượng VRAM thực tế và CPU RAM cùng trạng thái các slot hoán đổi."""
        stats = {}
        if not torch.cuda.is_available():
            stats["gpus"] = {"device": "cpu", "status": "No CUDA detected"}
        else:
            gpus = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                gpus[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "free_gb": round(total - reserved, 2),
                }
            stats["gpus"] = gpus
            # Backward compatibility for legacy flat keys
            for k, v in gpus.items():
                stats[k] = v

        stats["ram"] = self.get_system_ram()
        stats["active_dynamic_slot"] = self.active_dynamic_slot
        stats["always_active_slots"] = list(self.always_active_models.keys())
        stats["cached_dynamic_slots"] = list(self.dynamic_models.keys())
        return stats

    def register_always_active(self, slot_name: str, model_instance: Any):
        """Đăng ký mô hình thường trực bất khả xâm phạm."""
        with self.operation_lock:
            self.always_active_models[slot_name] = model_instance
            logger.info(f"📌 [Pinned VRAM] Mô hình '{slot_name}' đã được ghim thường trực (always_active).")

    def switch_dynamic_slot(self, target_slot: str, loader_fn: Callable[[], Any], engine_obj: Optional[Any] = None) -> Any:
        """
        Điều phối hoán đổi mô hình dynamic_switch giữa CPU RAM và GPU VRAM qua bus PCIe:
        1. Nếu target_slot đang tích cực trên GPU: Trả về tức thì (Zero latency).
        2. Nếu slot khác đang trên GPU: Chuyển slot cũ về CPU RAM, dọn cache VRAM.
        3. Đưa target_slot lên GPU từ RAM (hoặc nạp lần đầu nếu chưa có).
        """
        with self.operation_lock:
            # Trường hợp 1: Đang sẵn sàng trên GPU
            if self.active_dynamic_slot == target_slot and target_slot in self.dynamic_models:
                logger.debug(f"✨ [PCIe Fast-Swap] Slot '{target_slot}' đã sẵn sàng trên GPU (Zero latency).")
                return self.dynamic_models[target_slot]

            t0 = time.time()

            # Trường hợp 2: Có một mô hình dynamic khác đang chiếm GPU -> Chuyển về CPU RAM
            if self.active_dynamic_slot and self.active_dynamic_slot != target_slot:
                prev_slot = self.active_dynamic_slot
                logger.info(f"🔄 [GPU ➔ RAM] Nhường VRAM: Chuyển '{prev_slot}' về CPU RAM...")
                prev_model = self.dynamic_models.get(prev_slot)
                if prev_model is not None:
                    if hasattr(prev_model, "to"):
                        try:
                            prev_model.to("cpu")
                        except Exception as e:
                            logger.debug(f"Không thể chuyển {prev_slot} về CPU: {e}")
                self.clean_gpu()
                self.active_dynamic_slot = None

            # Trường hợp 3: Target đã có sẵn trong CPU RAM -> Đẩy lên GPU
            if target_slot in self.dynamic_models:
                logger.info(f"⚡ [RAM ➔ GPU] Kích hoạt '{target_slot}' từ CPU RAM qua bus PCIe (Zero Disk I/O)...")
                self.clean_gpu()
                model = self.dynamic_models[target_slot]
                if engine_obj and hasattr(engine_obj, "reload_to_gpu"):
                    engine_obj.reload_to_gpu()
                elif hasattr(model, "to") and torch.cuda.is_available():
                    try:
                        # Mặc định GPU 1 nếu có 2 GPU, ngược lại GPU 0
                        dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
                        model.to(dev)
                    except Exception as e:
                        logger.debug(f"Model to CUDA note: {e}")
                self.active_dynamic_slot = target_slot
                elapsed = time.time() - t0
                logger.info(f"✨ [PCIe Fast-Swap] '{target_slot}' sẵn sàng trên GPU sau {elapsed:.2f}s!")
                return model

            # Trường hợp 4: Nạp lần đầu (Cold Start)
            logger.info(f"🚀 [Cold Start] Khởi tạo '{target_slot}' lần đầu vào bộ nhớ...")
            self.clean_gpu()
            model = loader_fn()
            self.dynamic_models[target_slot] = model
            self.active_dynamic_slot = target_slot

            elapsed = time.time() - t0
            logger.info(f"✅ [Ready] '{target_slot}' đã nạp thành công vào hệ thống sau {elapsed:.2f}s!")
            return model

    # Tương thích ngược với code cũ
    def switch_heavyweight_slot(self, target_slot: str, loader_fn: Callable[[], Any]) -> Any:
        return self.switch_dynamic_slot(target_slot, loader_fn)

    def switch_gpu1_slot(self, target_slot: str, loader_fn: Callable[[], Any]) -> Any:
        return self.switch_dynamic_slot(target_slot, loader_fn)

    def warmup_models(self, registry):
        """
        Thực thi quy trình Warmup khi khởi động server dựa trên models.yaml:
        - preload: true && init_target: 'gpu'  -> Nạp thẳng vào VRAM.
        - preload: true && init_target: 'cpu'  -> Nạp sẵn vào RAM (sẵn sàng PCIe Fast-Swap).
        - preload: false                       -> Đợi request (Lazy Load).
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 [Startup Warmup] Bắt đầu khởi tạo các mô hình theo cấu hình models.yaml...")
        logger.info("=" * 70)

        for task, cfg in MODELS_CONFIG.items():
            is_preload = cfg.get("preload", False)
            init_target = cfg.get("init_target", "gpu").lower()
            lifecycle = cfg.get("lifecycle", "dynamic_switch").lower()
            model_id = cfg.get("id", "")

            if not is_preload:
                logger.info(f"⏳ [{task.upper()}] '{model_id}' [preload=False, lifecycle={lifecycle}] -> Chờ Lazy Load khi có request.")
                continue

            logger.info(f"🔥 [{task.upper()}] Preloading '{model_id}' (Target: {init_target.upper()} | Policy: {lifecycle})...")
            try:
                engine = registry.get_engine(task)
                if engine is None:
                    continue

                if init_target == "gpu":
                    loaded_obj = engine.load()
                    if lifecycle == "always_active":
                        self.register_always_active(task, loaded_obj)
                    else:
                        self.dynamic_models[task] = loaded_obj
                        self.active_dynamic_slot = task
                elif init_target == "cpu":
                    # Nạp vào RAM và đỗ lại ở CPU
                    loaded_obj = engine.load()
                    if hasattr(engine, "offload_to_cpu"):
                        engine.offload_to_cpu()
                    self.dynamic_models[task] = loaded_obj
                    logger.info(f"🚗 [{task.upper()}] Đỗ sẵn trong CPU RAM (Zero VRAM taken). Sẵn sàng Fast-Swap qua PCIe!")

            except Exception as e:
                logger.warning(f"⚠️ Khởi tạo trước [{task.upper()}] gặp cảnh báo: {e}. Sẽ thử lại khi có request.")

        self.clean_gpu()
        logger.info("=" * 70)
        logger.info(f"📊 [Warmup Hoàn Tất] Trạng thái VRAM sau khởi động:\n{self.report_vram()}")
        logger.info("=" * 70 + "\n")


# Singleton memory manager instance
_global_memory_manager = None


def get_memory_manager() -> MemoryManager:
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager
