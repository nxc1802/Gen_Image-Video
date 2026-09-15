"""
🧠 Device Topology Resolver: Trí thông minh tự động phân bổ phần cứng
Tự động tính toán sơ đồ bố trí GPU (GPU 0, GPU 1, hay Dual-GPU) dựa trên:
1. Số lượng GPU thực tế có sẵn trên máy (1x T4, 2x T4, A100, CPU...).
2. Kích thước tham số của mô hình (tự động phân tích từ tên model: ví dụ 2B, 7B, 14B, 26B, 70B...).
3. Mục tiêu: Nếu VLM nhỏ (<= 8B), tự động dồn vào GPU 0 để giải phóng 100% GPU 1 cho Image/Video!
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple
import torch

logger = logging.getLogger("DeviceResolver")


class DeviceTopologyResolver:
    """Bộ giải quyết tài nguyên và sơ đồ thiết bị tự động."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0

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

    def resolve(
        self,
        task: str,
        model_id: str,
        requested_strategy: str = "auto",
    ) -> Dict[str, Any]:
        """
        Trả về cấu hình phân bổ thiết bị chuẩn:
        {
            "device": str ("cuda:0", "cuda:1", "cpu"),
            "device_map": Optional[str] ("auto", "cuda:0", None),
            "is_dual_gpu": bool,
            "detected_size_b": Optional[float],
            "reason": str
        }
        """
        strategy = (requested_strategy or "auto").lower()

        # Nếu không có CUDA: Ép buộc chạy CPU
        if not self.cuda_available or self.gpu_count == 0:
            return {
                "device": "cpu",
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": self.extract_param_size_billions(model_id),
                "reason": "Không phát hiện card đồ họa NVIDIA (CUDA), chạy chế độ CPU."
            }

        detected_size = self.extract_param_size_billions(model_id)

        # 1. Nếu người dùng chỉ định rõ ràng thiết bị:
        if strategy == "gpu_0":
            return {
                "device": "cuda:0",
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "reason": "Người dùng cấu hình chỉ định GPU 0 (cuda:0)."
            }
        elif strategy == "gpu_1":
            target = "cuda:1" if self.gpu_count > 1 else "cuda:0"
            return {
                "device": target,
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "reason": f"Người dùng cấu hình chỉ định GPU 1 ({target})."
            }
        elif strategy == "dual_gpu":
            device_map = "auto" if self.gpu_count > 1 else None
            return {
                "device": "cuda:0",
                "device_map": device_map,
                "is_dual_gpu": (self.gpu_count > 1),
                "detected_size_b": detected_size,
                "reason": "Cấu hình kích hoạt phân tải song song qua cả 2 GPU (device_map='auto')."
            }

        # 2. Chế độ Tự Động Thông Minh ("auto"):
        # --- Task ÂM THANH (STT / TTS) ---
        if task in ("stt", "tts", "audio"):
            return {
                "device": "cuda:0",
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "reason": "Mô hình âm thanh nhẹ (~2GB VRAM), luôn ghim cố định trên GPU 0."
            }

        # --- Task HÌNH ẢNH (Image Generation) ---
        if task in ("image", "flux", "visual_image"):
            target = "cuda:1" if self.gpu_count > 1 else "cuda:0"
            return {
                "device": target,
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "reason": f"Mô hình tạo ảnh phân bổ vào {target} để tránh xung đột với Audio trên GPU 0."
            }

        # --- Task VIDEO (Video Generation) ---
        if task in ("video", "wan", "visual_video"):
            # Nếu là mô hình video lớn (>= 10B) và có 2 GPU
            if detected_size and detected_size >= 10.0 and self.gpu_count > 1:
                return {
                    "device": "cuda:1",
                    "device_map": "auto",
                    "is_dual_gpu": True,
                    "detected_size_b": detected_size,
                    "reason": f"Mô hình Video lớn ({detected_size}B >= 10B) được điều phối đa GPU qua CPU Offload."
                }
            target = "cuda:1" if self.gpu_count > 1 else "cuda:0"
            return {
                "device": target,
                "device_map": None,
                "is_dual_gpu": False,
                "detected_size_b": detected_size,
                "reason": f"Mô hình Video phân bổ vào {target}."
            }

        # --- Task VLM (Vision-Language Model) ---
        if task in ("vlm", "chat", "llm"):
            # Nếu model nhỏ (<= 8B tham số, ví dụ 2B, 7B)
            if detected_size is not None and detected_size <= 8.0:
                logger.info(
                    f"🎯 [Auto Topology] Phát hiện VLM cỡ nhỏ: '{model_id}' ({detected_size}B <= 8B). "
                    f"Tự động gán vào GPU 0 (cuda:0). GPU 1 (16GB) được giải phóng 100% cho Visual models!"
                )
                return {
                    "device": "cuda:0",
                    "device_map": None,
                    "is_dual_gpu": False,
                    "detected_size_b": detected_size,
                    "reason": f"Model {detected_size}B <= 8B đủ nhỏ để nằm trọn trên GPU 0, giải phóng GPU 1 hoàn toàn."
                }
            else:
                # Model lớn (>= 14B như 26B, 32B) hoặc không rõ kích cỡ: Chia tải song song qua cả 2 GPU
                device_map = "auto" if self.gpu_count > 1 else None
                logger.info(
                    f"🎯 [Auto Topology] Phát hiện VLM cỡ lớn: '{model_id}' ({detected_size}B > 8B). "
                    f"Tự động chia tải song song qua cả 2 GPU (device_map='auto')."
                )
                return {
                    "device": "cuda:0",
                    "device_map": device_map,
                    "is_dual_gpu": (self.gpu_count > 1),
                    "detected_size_b": detected_size,
                    "reason": f"Model {detected_size or 'Unknown'}B > 8B cần chia tải song song qua cả 2 GPU."
                }

        # Mặc định fallback:
        return {
            "device": "cuda:0",
            "device_map": None,
            "is_dual_gpu": False,
            "detected_size_b": detected_size,
            "reason": "Phân bổ mặc định vào GPU 0."
        }


# Singleton resolver instance
_global_resolver = None


def get_device_resolver() -> DeviceTopologyResolver:
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = DeviceTopologyResolver()
    return _global_resolver
