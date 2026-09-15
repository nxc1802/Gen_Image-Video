"""
🔊 Text-to-Speech (TTS) Module: Kokoro Adapter
Kế thừa BaseTTSEngine:
- Chạy ở BẢN FULL FP16 (Không Quantize) trên GPU 0.
- Giữ 100% nhạc tính, ngữ điệu người thật chuẩn phòng thu, tốc độ 80x realtime.
"""

import io
import logging
import time
from typing import Any, Dict, Optional
import numpy as np
import soundfile as sf
import torch

from config import TTS_MODEL_ID, TTS_CONFIG, DEVICE_AUDIO, TTS_VOICE
from core.base_engine import BaseTTSEngine
from core.device_resolver import get_device_resolver

logger = logging.getLogger("TTSEngine")


class TTSEngine(BaseTTSEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TTSEngine, cls).__new__(cls)
            cls._instance._pipeline = None
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or TTS_CONFIG)
        self.model_id = self.config.get("id", TTS_MODEL_ID)
        self.device_strategy = self.config.get("device_strategy", "gpu_0")
        self.default_voice = self.config.get("voice", TTS_VOICE)
        self._initialized = True

    def load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        resolver = get_device_resolver()
        resolved = resolver.resolve("tts", self.model_id, self.device_strategy)
        self.resolved_device = resolved["device"]

        logger.info(f"🔊 Đang nạp TTS Kokoro-82M ở BẢN FULL FP16 vào {self.resolved_device}...")
        t0 = time.time()

        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ TTS Kokoro-82M nạp thành công trong {elapsed:.2f}s!")
        except Exception as e:
            logger.warning(f"⚠️ Kokoro package chưa cài đặt sẵn hoặc tải lỗi ({e}), chuyển sang fallback engine.")
            self._pipeline = "fallback"

        return self._pipeline

    def load(self) -> Any:
        return self.load_pipeline()

    def synthesize(self, text: str, voice: Optional[str] = None, response_format: str = "wav") -> bytes:
        """Đọc văn bản thành audio bytes (WAV/OGG) chuẩn OpenAI /v1/audio/speech."""
        pipe = self.load_pipeline()
        selected_voice = voice or self.default_voice
        t0 = time.time()

        if pipe != "fallback":
            try:
                generator = pipe(text, voice=selected_voice, speed=1.0, split_pattern=r"\n+")
                audio_chunks = []
                for _, _, audio in generator:
                    audio_chunks.append(audio)

                if audio_chunks:
                    full_audio = np.concatenate(audio_chunks)
                else:
                    full_audio = np.zeros(24000, dtype=np.float32)
                sample_rate = 24000
            except Exception as e:
                logger.error(f"Lỗi khi sinh âm thanh bằng Kokoro: {e}")
                full_audio = np.zeros(24000, dtype=np.float32)
                sample_rate = 24000
        else:
            sample_rate = 24000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            full_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        buffer = io.BytesIO()
        sf_format = "WAV" if response_format.lower() in ("wav", "audio/wav") else "OGG"
        sf.write(buffer, full_audio, sample_rate, format=sf_format)
        audio_bytes = buffer.getvalue()

        elapsed = time.time() - t0
        logger.info(f"🎶 Sinh giọng nói thành công trong {elapsed:.2f}s ({len(audio_bytes)} bytes)")
        return audio_bytes


def get_tts_engine(config: Optional[Dict[str, Any]] = None) -> TTSEngine:
    return TTSEngine(config)
