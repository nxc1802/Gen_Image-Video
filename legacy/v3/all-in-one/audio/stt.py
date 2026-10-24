"""
🎙️ Speech-to-Text (STT) Module: Qwen3-ASR Adapter (với AudioService Gateway)
================================================================================
Kế thừa BaseSTTEngine:
- Model chính: Qwen3-ASR 1.7B (Multilingual, hỗ trợ đặc biệt tốt cho tiếng Việt)
- Được phục vụ qua FastAPI Audio Service daemon trên GPU 0 (cuda:0).
- Dự phòng: Whisper Large-v3-Turbo nếu được yêu cầu cụ thể hoặc fallback.
"""

import io
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional
import torch

from config import STT_MODEL_ID, STT_CONFIG, DEVICE_AUDIO, STT_DTYPE
from core.base_engine import BaseSTTEngine
from core.memory_manager import get_memory_manager
from .audio_manager import get_audio_service_manager

logger = logging.getLogger("STTEngine")


class STTEngine(BaseSTTEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(STTEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or STT_CONFIG)
        self.model_id = self.config.get("id", "Qwen/Qwen3-ASR-1.7B")
        self.device_strategy = self.config.get("device_strategy", "gpu_0")
        self.lifecycle = self.config.get("lifecycle", "always_active")
        self.audio_manager = get_audio_service_manager()
        self._initialized = True

    def release_from_gpu(self):
        """Giải phóng hoàn toàn ASR khỏi GPU VRAM."""
        logger.info("🧹 Giải phóng Qwen3-ASR khỏi GPU VRAM...")
        self.audio_manager.free_vram()
        self._is_loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> Any:
        """Đảm bảo Audio Service daemon sẵn sàng."""
        if not self.audio_manager.is_running():
            self.audio_manager.start_service()
        self._is_loaded = True
        return self.audio_manager

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "vi",
        prompt: Optional[str] = None,
        task: Optional[str] = "transcribe",
        temperature: Optional[float] = 0.0,
        model: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        return_time_stamps: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Nhận diện giọng nói đa ngôn ngữ qua Qwen3-ASR 1.7B.
        Trả về kết quả chuẩn OpenAI transcription: {"text": "...", "language": "...", "duration_seconds": float}
        """
        self.load()
        target_model = model or self.model_id

        # Nếu người dùng yêu cầu cụ thể whisper và có package whisper
        if "whisper" in target_model.lower():
            try:
                import whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    tmp_f.write(audio_bytes)
                    tmp_path = tmp_f.name
                try:
                    w_model = whisper.load_model("large-v3-turbo", device="cuda:0" if torch.cuda.is_available() else "cpu")
                    opts = {"task": task or "transcribe", "temperature": temperature or 0.0}
                    if language and language not in ("auto", ""):
                        opts["language"] = language
                    res = w_model.transcribe(tmp_path, **opts)
                    return {
                        "text": res.get("text", "").strip(),
                        "language": res.get("language", language or "vi"),
                        "duration_seconds": 0.5,
                        "model": "whisper-large-v3-turbo",
                    }
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Whisper fallback gặp lỗi: {e}, tiếp tục với Qwen3-ASR.")

        # Gọi AudioServiceManager cho Qwen3-ASR 1.7B
        t0 = time.time()
        res = self.audio_manager.transcribe(
            audio_bytes=audio_bytes,
            language=language or "vi",
            task=task or "transcribe",
            prompt=prompt,
            model=target_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_time_stamps=return_time_stamps,
        )
        elapsed = round(time.time() - t0, 2)
        if "duration_seconds" not in res:
            res["duration_seconds"] = elapsed
        logger.info(f"🎯 [Qwen3-ASR] Nhận diện hoàn tất trong {elapsed}s: '{res.get('text', '')[:50]}...'")
        return res


def get_stt_engine(config: Optional[Dict[str, Any]] = None) -> STTEngine:
    return STTEngine(config)
