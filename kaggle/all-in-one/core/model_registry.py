"""
🏭 Model Registry & Factory
Quản lý vòng đời và điểm truy cập tập trung cho tất cả các Model Engine.
Tự động ánh xạ từ cấu hình models.yaml sang đúng adapter engine tương ứng.
"""

import logging
import threading
from typing import Any, Dict, Optional

from config import MODELS_CONFIG
from core.base_engine import (
    BaseModelEngine,
    BaseVLMEngine,
    BaseImageEngine,
    BaseVideoEngine,
    BaseSTTEngine,
    BaseTTSEngine,
)

logger = logging.getLogger("ModelRegistry")


class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._engines = {}
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("🏭 ModelRegistry khởi tạo thành công.")

    def get_vlm(self) -> BaseVLMEngine:
        """Lấy VLM Engine (Vision-Language)."""
        if "vlm" not in self._engines:
            from vlm.qwen import QwenVLMEngine
            vlm_cfg = MODELS_CONFIG.get("vlm", {})
            self._engines["vlm"] = QwenVLMEngine(vlm_cfg)
        return self._engines["vlm"]

    def get_image(self) -> BaseImageEngine:
        """Lấy Image Engine (FLUX / Diffusion)."""
        if "image" not in self._engines:
            from visual.flux_image import FluxImageEngine
            img_cfg = MODELS_CONFIG.get("image", {})
            self._engines["image"] = FluxImageEngine(img_cfg)
        return self._engines["image"]

    def get_video(self) -> BaseVideoEngine:
        """Lấy Video Engine (Wan2.1 / Video Diffusion)."""
        if "video" not in self._engines:
            from visual.wan_video import WanVideoEngine
            vid_cfg = MODELS_CONFIG.get("video", {})
            self._engines["video"] = WanVideoEngine(vid_cfg)
        return self._engines["video"]

    def get_stt(self) -> BaseSTTEngine:
        """Lấy STT Engine (Whisper)."""
        if "stt" not in self._engines:
            from audio.stt import STTEngine
            stt_cfg = MODELS_CONFIG.get("stt", {})
            self._engines["stt"] = STTEngine(stt_cfg)
        return self._engines["stt"]

    def get_tts(self) -> BaseTTSEngine:
        """Lấy TTS Engine (Kokoro)."""
        if "tts" not in self._engines:
            from audio.tts import TTSEngine
            tts_cfg = MODELS_CONFIG.get("tts", {})
            self._engines["tts"] = TTSEngine(tts_cfg)
        return self._engines["tts"]

    def get_engine(self, task: str) -> Optional[BaseModelEngine]:
        """Truy xuất engine theo tên task linh hoạt."""
        task_lower = task.lower()
        if task_lower in ("vlm", "chat", "llm"):
            return self.get_vlm()
        elif task_lower in ("image", "flux", "visual_image"):
            return self.get_image()
        elif task_lower in ("video", "wan", "visual_video"):
            return self.get_video()
        elif task_lower in ("stt", "transcription"):
            return self.get_stt()
        elif task_lower in ("tts", "speech"):
            return self.get_tts()
        return None

    def get_catalog(self) -> Dict[str, Any]:
        """Báo cáo danh mục các mô hình hiện hành và trạng thái nạp."""
        return {
            task: {
                "model_id": cfg.get("id"),
                "device_strategy": cfg.get("device_strategy"),
                "quantization": cfg.get("quantization") or cfg.get("precision"),
            }
            for task, cfg in MODELS_CONFIG.items()
        }


# Singleton accessor
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
