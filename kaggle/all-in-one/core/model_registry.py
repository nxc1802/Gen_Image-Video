"""
🏭 Model Registry & Factory
Quản lý vòng đời và điểm truy cập tập trung cho tất cả các Model Engine.
Tự động ánh xạ từ cấu hình models.yaml sang đúng adapter engine tương ứng.
"""

import logging
import threading
from typing import Any, Dict, Optional

import config
from core.base_engine import (
    BaseModelEngine,
    BaseVLMEngine,
    BaseImageEngine,
    BaseVideoEngine,
    BaseSTTEngine,
    BaseTTSEngine,
    BaseEmbeddingEngine,
    BaseRerankEngine,
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
            vlm_cfg = config.MODELS_CONFIG.get("vlm", {})
            self._engines["vlm"] = QwenVLMEngine(vlm_cfg)
        return self._engines["vlm"]

    def get_image(self) -> BaseImageEngine:
        """Lấy Image Engine (Qwen-Image-2.1 ComfyUI)."""
        if "image" not in self._engines:
            from visual import get_qwen_image_engine
            img_cfg = config.MODELS_CONFIG.get("image", {})
            self._engines["image"] = get_qwen_image_engine(img_cfg)
        return self._engines["image"]

    def get_video(self) -> BaseVideoEngine:
        """Lấy Video Engine (Wan2.1 / Wan2.2 Video Diffusion)."""
        if "video" not in self._engines:
            from visual import get_wan_engine
            vid_cfg = config.MODELS_CONFIG.get("video", {})
            self._engines["video"] = get_wan_engine(vid_cfg)
        return self._engines["video"]

    def get_minimax(self, gpu_device: Optional[int] = None) -> BaseVideoEngine:
        """Lấy MiniMax H3 Video Engine theo GPU cụ thể (hỗ trợ Dual-GPU song song)."""
        if not hasattr(self, "_minimax_engines"):
            self._minimax_engines = {}

        target_gpu = gpu_device
        if target_gpu is None:
            from config import resolve_model_gpu_device
            _, default_gpu = resolve_model_gpu_device("minimax")
            target_gpu = default_gpu

        target_gpu = int(target_gpu)
        if target_gpu not in self._minimax_engines:
            from visual import get_minimax_engine
            mm_cfg = dict(config.MODELS_CONFIG.get("minimax", config.MODELS_CONFIG.get("video", {})))
            mm_cfg["gpu_device"] = target_gpu
            mm_cfg["device"] = f"cuda:{target_gpu}"
            self._minimax_engines[target_gpu] = get_minimax_engine(mm_cfg, gpu_device=target_gpu)
            if "minimax" not in self._engines:
                self._engines["minimax"] = self._minimax_engines[target_gpu]
        return self._minimax_engines[target_gpu]

    def get_stt(self) -> BaseSTTEngine:
        """Lấy STT Engine (Qwen3-ASR 1.7B / Whisper)."""
        if "stt" not in self._engines:
            from audio.stt import STTEngine
            stt_cfg = config.MODELS_CONFIG.get("stt", {})
            self._engines["stt"] = STTEngine(stt_cfg)
        return self._engines["stt"]

    def get_tts(self) -> BaseTTSEngine:
        """Lấy TTS Engine (Qwen3-TTS 1.7B & XTTS-v2 Vietnamese)."""
        if "tts" not in self._engines:
            from audio.tts import TTSEngine
            tts_cfg = config.MODELS_CONFIG.get("tts", {})
            self._engines["tts"] = TTSEngine(tts_cfg)
        return self._engines["tts"]

    def get_embedding(self) -> BaseEmbeddingEngine:
        """Lấy Embedding Engine (Qwen3-Embedding)."""
        if "embedding" not in self._engines:
            from embedding.embedding_service import get_embedding_service_manager
            self._engines["embedding"] = get_embedding_service_manager().get_embedding_engine()
        return self._engines["embedding"]

    def get_rerank(self) -> BaseRerankEngine:
        """Lấy Rerank Engine (Qwen3-Reranker)."""
        if "rerank" not in self._engines:
            from embedding.embedding_service import get_embedding_service_manager
            self._engines["rerank"] = get_embedding_service_manager().get_rerank_engine()
        return self._engines["rerank"]

    def get_engine(self, task: str) -> Optional[BaseModelEngine]:
        """Truy xuất engine theo tên task linh hoạt."""
        task_lower = task.lower()
        if task_lower in ("vlm", "chat", "llm"):
            return self.get_vlm()
        elif task_lower in ("image", "flux", "visual_image"):
            return self.get_image()
        elif task_lower in ("video", "wan", "visual_video"):
            return self.get_video()
        elif task_lower in ("minimax", "h3", "minimax_h3"):
            return self.get_minimax()
        elif task_lower in ("stt", "transcription", "asr", "qwen_asr"):
            return self.get_stt()
        elif task_lower in ("tts", "speech", "qwen_tts", "xtts", "xtts_v2"):
            return self.get_tts()
        elif task_lower in ("embedding", "embed"):
            return self.get_embedding()
        elif task_lower in ("rerank", "reranker"):
            return self.get_rerank()
        return None

    def get_catalog(self) -> Dict[str, Any]:
        """Báo cáo danh mục các mô hình hiện hành và chính sách vòng đời đầy đủ."""
        return {
            task: {
                "model_id": cfg.get("id"),
                "device": cfg.get("device"),
                "gpu_device": cfg.get("gpu_device"),
                "device_strategy": cfg.get("device_strategy"),
                "quantization": cfg.get("quantization") or cfg.get("precision"),
                "lifecycle": cfg.get("lifecycle"),
                "preload": cfg.get("preload"),
                "init_target": cfg.get("init_target"),
                "allocation_policy": cfg.get("allocation_policy"),
            }
            for task, cfg in config.MODELS_CONFIG.items()
        }


# Singleton accessor
_global_registry = None


def get_model_registry() -> ModelRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
