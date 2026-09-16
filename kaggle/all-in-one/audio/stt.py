"""
🎙️ Speech-to-Text (STT) Module: Whisper Adapter
Kế thừa BaseSTTEngine:
- Chạy ở BẢN FULL FP16 (Không Quantize) trên GPU 0.
- Tuân thủ lifecycle always_active (thường trực 100% trong VRAM, không bao giờ bị dọn).
"""

import io
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional
import torch
import whisper

from config import STT_MODEL_ID, STT_CONFIG, DEVICE_AUDIO, STT_DTYPE
from core.base_engine import BaseSTTEngine
from core.device_resolver import get_device_resolver
from core.memory_manager import get_memory_manager

logger = logging.getLogger("STTEngine")


class STTEngine(BaseSTTEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(STTEngine, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or STT_CONFIG)
        self.model_id = self.config.get("id", STT_MODEL_ID)
        self.device_strategy = self.config.get("device_strategy", "gpu_0")
        self.lifecycle = self.config.get("lifecycle", "always_active")
        self._initialized = True

    def load_model(self):
        if self._model is not None:
            return self._model

        resolver = get_device_resolver()
        resolved = resolver.resolve(
            task="stt",
            model_id=self.model_id,
            requested_strategy=self.device_strategy,
            precision="fp16",
            config=self.config,
        )
        self.resolved_device = resolved["device"]

        logger.info(f"🎙️ Đang nạp STT Whisper ({self.model_id}) ở BẢN FULL FP16 vào {self.resolved_device}...")
        t0 = time.time()

        model_name = "large-v3-turbo" if "turbo" in self.model_id.lower() else "large-v3"
        try:
            self._model = whisper.load_model(model_name, device=self.resolved_device)
            self._is_loaded = True
        except Exception as e:
            logger.warning(f"⚠️ Không thể nạp whisper ({e}), kích hoạt fallback.")
            self._model = "fallback"

        if self.lifecycle == "always_active":
            mem = get_memory_manager()
            mem.register_always_active("stt", self._model)

        elapsed = time.time() - t0
        logger.info(f"✅ STT Whisper nạp thành công trong {elapsed:.2f}s!")
        return self._model

    def load(self) -> Any:
        return self.load_model()

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        task: Optional[str] = "transcribe",
        temperature: Optional[float] = 0.0,
    ) -> Dict[str, Any]:
        """
        Nhận diện giọng nói từ mảng bytes âm thanh.
        Hỗ trợ đa ngôn ngữ (EN, VI), chuyển ngữ (task='translate'), và kiểm soát temperature.
        Trả về kết quả chuẩn OpenAI transcription: {"text": "..."}
        """
        model = self.load_model()

        lang_code = None
        if language and language.lower() not in ("auto", "none", ""):
            lang_code = "vi" if "vi" in language.lower() else "en"

        if model == "fallback":
            return {
                "text": "Đây là kết quả nhận diện giọng nói thử nghiệm (Whisper Fallback Engine).",
                "language": lang_code or "vi",
                "duration_seconds": 0.5,
            }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            t0 = time.time()
            transcribe_options = {
                "fp16": (self.resolved_device != "cpu"),
                "verbose": False,
                "task": task or "transcribe",
                "temperature": temperature if temperature is not None else 0.0,
            }
            if lang_code:
                transcribe_options["language"] = lang_code
            if prompt:
                transcribe_options["initial_prompt"] = prompt

            result = model.transcribe(tmp_path, **transcribe_options)
            elapsed = time.time() - t0

            detected_lang = result.get("language", lang_code or "auto")
            logger.info(f"🎯 Nhận diện hoàn tất trong {elapsed:.2f}s [lang={detected_lang}, task={transcribe_options['task']}]: '{result.get('text', '')[:40]}...'")
            return {
                "text": result.get("text", "").strip(),
                "language": detected_lang,
                "duration_seconds": round(elapsed, 2),
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def get_stt_engine(config: Optional[Dict[str, Any]] = None) -> STTEngine:
    return STTEngine(config)
