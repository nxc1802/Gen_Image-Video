"""
🧠 Adaptive Dynamic Allocator & Device Topology Resolver
Trí thông minh tự động phân bổ và cân bằng tải phần cứng (Adaptive Multi-GPU Balancing):
1. Đọc dung lượng VRAM thực tế theo thời gian thực (torch.cuda.mem_get_info) trên từng GPU.
2. Bỏ hoàn toàn việc chia cứng 50-50:
   - Nếu GPU 0 đang bị chiếm (ví dụ Audio STT/TTS chiếm ~2GB), GPU 1 trống 14.5GB:
     -> Phân bổ layers theo tỷ lệ Watermark Budget (max_memory), GPU 1 gánh nhiều hơn, GPU 0 gánh ít hơn.
   - Nếu model đủ nhỏ để nằm trọn trong 1 GPU (ví dụ Qwen 7B 4-bit ~5.5GB):
     -> Tự động đặt 100% vào GPU có VRAM trống lớn nhất (loại bỏ 100% độ trễ truyền dữ liệu bus PCIe đa GPU).
3. Hỗ trợ đầy đủ các chính sách lifecycle (always_active, dynamic_switch) và init_target (gpu, cpu).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import torch

logger = logging.getLogger("DeviceResolver")


class DeviceTopologyResolver:
    """Bộ giải quyết tài nguyên và điều phối đa GPU tự thích ứng (Adaptive Dynamic Allocator)."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0

        # Bộ giả lập VRAM cho môi trường kiểm thử / test harness
        self._mock_gpu_memory: Optional[List[Dict[str, float]]] = None

    def set_mock_gpu_memory(self, mock_info: Optional[List[Dict[str, float]]]):
        """Hỗ trợ kiểm thử: Thiết lập thông số VRAM giả lập (ví dụ GPU 0: 4GB, GPU 1: 14GB)."""
        self._mock_gpu_memory = mock_info
        if mock_info:
            self.cuda_available = True
            self.gpu_count = len(mock_info)

    @staticmethod
    def extract_param_size_billions(model_id: str) -> Optional[float]:
        """
        Trích xuất số lượng tham số (Billion parameters) từ tên model.
        Ví dụ:
        - "Qwen/Qwen2.5-VL-7B-Instruct" -> 7.0
        - "Qwen/Qwen2.5-VL-26B-Instruct" -> 26.0
        - "Qwen2-VL-2B" -> 2.0
        - "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" -> 1.3
        - "Wan-AI/Wan2.1-T2V-14B-Diffusers" -> 14.0
        """
        if not model_id:
            return None
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?:[^\w]|$)", model_id)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def estimate_vram_requirement_gb(
        self,
        task: str,
        model_id: str,
        quantization: str = "4bit",
        precision: str = "fp16",
    ) -> float:
        """
        Ước lượng dung lượng VRAM cần thiết (GB) cho mô hình (trọng số + buffer tính toán / KV cache).
        """
        task_lower = task.lower()
        if task_lower in ("stt", "whisper"):
            return 1.8  # Whisper large-v3-turbo FP16
        if task_lower in ("tts", "kokoro"):
            return 0.35  # Kokoro 82M FP16

        detected_b = self.extract_param_size_billions(model_id)

        # Image Gen (FLUX.2 Klein: 4B DiT + 4B Qwen3 = ~8B total params in Q4_K_M -> ~5.2 GB total)
        if task_lower in ("image", "flux", "visual_image"):
            if "9b" in model_id.lower():
                return 9.5 if quantization in ("4bit", "q4_k_m") else 18.0
            if quantization in ("4bit", "q4_k_m"):
                return 5.2  # 4-bit GGUF FLUX.2 Klein 4B + Qwen3 4B Text Encoder + VAE
            return 12.0

        # Video Gen (Wan 14B / Wan 1.3B)
        if task_lower in ("video", "wan", "visual_video"):
            if detected_b is not None and detected_b < 5.0:
                return 4.0  # Wan 1.3B
            if quantization == "4bit":
                return 12.0  # Wan 14B 4-bit
            return 28.0

        # VLM / LLM
        if detected_b is not None:
            if quantization == "4bit":
                # ~0.65 GB mỗi Billion params + 1.2 GB activation/KV buffer
                return round(detected_b * 0.65 + 1.2, 2)
            else:
                return round(detected_b * 2.0 + 2.0, 2)

        # Fallback mặc định theo task
        return 6.0

    def get_realtime_gpu_memory(self, headroom_gb: float = 1.5) -> List[Dict[str, Any]]:
        """
        Đọc dung lượng VRAM thực tế theo thời gian thực (torch.cuda.mem_get_info).
        Tính toán Budget khả dụng sau khi trừ đi headroom an toàn (tránh OOM).
        """
        if self._mock_gpu_memory is not None:
            result = []
            for i, item in enumerate(self._mock_gpu_memory):
                free_gb = item["free_gb"]
                total_gb = item.get("total_gb", 16.0)
                budget_gb = max(0.5, free_gb - headroom_gb)
                result.append({
                    "gpu_id": i,
                    "device_name": f"Mock GPU {i}",
                    "total_gb": total_gb,
                    "free_gb": free_gb,
                    "budget_gb": round(budget_gb, 2),
                })
            return result

        if not self.cuda_available or self.gpu_count == 0:
            return []

        result = []
        for i in range(self.gpu_count):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                budget_gb = max(0.5, free_gb - headroom_gb)
                result.append({
                    "gpu_id": i,
                    "device_name": torch.cuda.get_device_name(i),
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "budget_gb": round(budget_gb, 2),
                })
            except Exception as e:
                logger.warning(f"Không thể đọc VRAM GPU {i}: {e}")
                result.append({
                    "gpu_id": i,
                    "device_name": f"GPU {i}",
                    "total_gb": 16.0,
                    "free_gb": 14.0,
                    "budget_gb": 12.5,
                })
        return result

    def resolve(
        self,
        task: str,
        model_id: str,
        requested_strategy: str = "auto",
        quantization: str = "4bit",
        precision: str = "fp16",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Quyết định sơ đồ phân bổ tối ưu theo thời gian thực (Adaptive Dynamic Allocation):
        1. Nếu chỉ định cứng 'gpu_0' hoặc 'gpu_1': Tuân thủ cấu hình.
        2. Nếu 'auto' hoặc 'adaptive':
           - Ước lượng VRAM cần thiết cho mô hình.
           - Kiểm tra VRAM trống thực tế trên từng GPU.
           - NẾU VỪA 1 GPU: Nạp 100% vào GPU trống nhất (Single-GPU, 0% PCIe overhead).
           - NẾU VƯỢT QUÁ 1 GPU: Chia tỷ lệ động theo dung lượng thực tế qua max_memory (BỎ CHIA CỨNG 50-50).
        """
        cfg = config or {}
        strategy = (cfg.get("device_strategy") or requested_strategy or cfg.get("allocation_policy") or "auto").lower()
        preferred_device = cfg.get("preferred_device", "")
        detected_size = self.extract_param_size_billions(model_id)
        needed_gb = self.estimate_vram_requirement_gb(task, model_id, quantization, precision)

        # 1. Trường hợp không có GPU (CPU Mode)
        if not self.cuda_available or self.gpu_count == 0:
            return {
                "device": "cpu",
                "device_map": None,
                "max_memory": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": "Không phát hiện card đồ họa NVIDIA (CUDA), chạy chế độ CPU."
            }

        # 2. ƯU TIÊN TUYỆT ĐỐI CẤU HÌNH NGƯỜI DÙNG CHỈ ĐỊNH (Config-Driven, No Auto-Predict)
        gpu_count_cfg = cfg.get("gpu_count")
        if gpu_count_cfg == 2 or strategy in ("dual_gpu", "2_gpu", "split"):
            return {
                "device": "cuda:0",
                "device_map": "auto",
                "max_memory": {0: "14GiB", 1: "14GiB"} if self.gpu_count > 1 else {0: "14GiB"},
                "is_dual_gpu": (self.gpu_count > 1),
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": "Người dùng chỉ định rõ ràng chế độ 2 GPU trong config (dual_gpu / gpu_count: 2)."
            }
        elif gpu_count_cfg == 1 or strategy in ("single_gpu", "1_gpu"):
            target_idx = 1 if (preferred_device == "gpu_1" and self.gpu_count > 1) else 0
            return {
                "device": f"cuda:{target_idx}",
                "device_map": None,
                "max_memory": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": f"Người dùng chỉ định rõ ràng chế độ 1 GPU trong config (cuda:{target_idx})."
            }

        if strategy == "gpu_0":
            return {
                "device": "cuda:0",
                "device_map": "auto",
                "max_memory": {0: "13GiB"},
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": "Chỉ định cứng GPU 0 (cuda:0)."
            }
        elif strategy == "gpu_1":
            target = "cuda:1" if self.gpu_count > 1 else "cuda:0"
            target_idx = 1 if self.gpu_count > 1 else 0
            return {
                "device": target,
                "device_map": "auto",
                "max_memory": {target_idx: "14GiB"},
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": f"Chỉ định cứng GPU 1 ({target})."
            }

        # 3. Task Âm thanh (nhẹ ~0.35 - 1.8GB): Tự động chọn GPU trống nhất
        if task in ("stt", "tts", "audio"):
            target_idx = 0
            if self.gpu_count > 1:
                gpu_info = self.get_realtime_gpu_memory(headroom_gb=1.0)
                if gpu_info:
                    target_idx = max(gpu_info, key=lambda g: g["free_gb"])["gpu_id"]
            target = f"cuda:{target_idx}"
            return {
                "device": target,
                "device_map": None,
                "max_memory": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": f"Mô hình âm thanh nhẹ (~{needed_gb}GB), phân bổ động vào {target}."
            }

        # 4. CHẾ ĐỘ TỰ THÍCH ỨNG (ADAPTIVE DYNAMIC ALLOCATION)
        gpu_info = self.get_realtime_gpu_memory(headroom_gb=1.5)

        # Nếu chỉ có 1 GPU duy nhất
        if self.gpu_count <= 1:
            return {
                "device": "cuda:0",
                "device_map": None,
                "max_memory": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": "Hệ thống chỉ có 1 GPU (cuda:0), phân bổ toàn bộ vào GPU 0."
            }

        # --- Kiểm tra Khả năng Chạy 1 GPU Đơn (Single-GPU Optimization) ---
        # Quy tắc: Nếu có GPU nào chứa vừa model + 1.5GB headroom -> Ưu tiên chạy 1 GPU đơn!
        # Vì chạy 1 GPU đơn loại bỏ 100% overhead truyền tensor qua bus PCIe.
        eligible_gpus = [g for g in gpu_info if g["free_gb"] >= (needed_gb + 1.2)]

        if eligible_gpus:
            # Nếu preferred_device nằm trong danh sách đủ điều kiện
            selected_gpu = None
            if preferred_device == "gpu_0":
                gpu0_candidate = next((g for g in eligible_gpus if g["gpu_id"] == 0), None)
                if gpu0_candidate:
                    selected_gpu = gpu0_candidate
            elif preferred_device == "gpu_1":
                gpu1_candidate = next((g for g in eligible_gpus if g["gpu_id"] == 1), None)
                if gpu1_candidate:
                    selected_gpu = gpu1_candidate

            # Nếu không chỉ định hoặc không thỏa mãn preferred, chọn GPU có VRAM trống nhiều nhất
            if selected_gpu is None:
                # Đối với visual task (image, video), ưu tiên GPU 1 để tránh GPU 0 (nơi Audio thường trú)
                if task in ("image", "video", "flux", "wan"):
                    gpu1_candidate = next((g for g in eligible_gpus if g["gpu_id"] == 1), None)
                    selected_gpu = gpu1_candidate if gpu1_candidate else max(eligible_gpus, key=lambda g: g["free_gb"])
                else:
                    selected_gpu = max(eligible_gpus, key=lambda g: g["free_gb"])

            target_dev = f"cuda:{selected_gpu['gpu_id']}"
            return {
                "device": target_dev,
                "device_map": None,
                "max_memory": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "estimated_vram_gb": needed_gb,
                "reason": (
                    f"🎯 [Adaptive Single-GPU] Model (~{needed_gb:.1f}GB) nằm trọn trong GPU {selected_gpu['gpu_id']} "
                    f"({selected_gpu['free_gb']:.1f}GB trống). Chạy 1 GPU đơn để đạt hiệu năng tối đa (Zero PCIe Latency)."
                )
            }

        # --- PHÂN BỔ ĐA GPU TỰ THÍCH ỨNG (DYNAMIC WATERMARK SPLITTING) ---
        # Nếu model không vừa bất kỳ 1 GPU đơn lẻ nào (ví dụ Qwen 26B 4-bit ~18.5GB):
        # BỎ HOÀN TOÀN CHIA CỨNG 50-50!
        # Tính toán budget thực tế theo tỷ lệ VRAM còn trống trên GPU 0 và GPU 1.
        budget_0 = max(0.5, gpu_info[0]["budget_gb"]) if len(gpu_info) > 0 else 6.0
        budget_1 = max(0.5, gpu_info[1]["budget_gb"]) if len(gpu_info) > 1 else 12.0

        # max_memory dict cho Hugging Face accelerate / diffusers
        max_memory = {
            0: f"{int(budget_0 * 1024)}MiB",
            1: f"{int(budget_1 * 1024)}MiB",
            "cpu": "24GiB",
        }

        total_gpu_budget = budget_0 + budget_1
        ratio_0 = (budget_0 / total_gpu_budget) * 100
        ratio_1 = (budget_1 / total_gpu_budget) * 100

        logger.info(
            f"⚖️ [Dynamic Watermark Allocation] Model {task} ({needed_gb:.1f}GB) vượt 1 GPU. "
            f"Chia tải động: GPU 0 ({ratio_0:.0f}% ~ {budget_0:.1f}GB) | GPU 1 ({ratio_1:.0f}% ~ {budget_1:.1f}GB)."
        )

        return {
            "device": "cuda:0",
            "device_map": "auto",
            "max_memory": max_memory,
            "is_dual_gpu": True,
            "detected_size_b": detected_size,
            "estimated_vram_gb": needed_gb,
            "budget_breakdown": {
                "gpu_0_gb": budget_0,
                "gpu_1_gb": budget_1,
                "ratio_gpu0_pct": round(ratio_0, 1),
                "ratio_gpu1_pct": round(ratio_1, 1),
            },
            "reason": (
                f"⚖️ [Adaptive Dynamic Balancing] Model (~{needed_gb:.1f}GB) vượt quá 1 GPU. "
                f"Chia tải tự động theo VRAM thực tế: GPU 0 ({budget_0:.1f}GB) + GPU 1 ({budget_1:.1f}GB) "
                f"+ CPU Offload (Bỏ chia cứng 50-50)."
            ),
        }


# Singleton resolver instance
_global_resolver = None


def get_device_resolver() -> DeviceTopologyResolver:
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = DeviceTopologyResolver()
    return _global_resolver
