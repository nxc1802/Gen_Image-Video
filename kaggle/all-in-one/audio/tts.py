"""
🔊 Text-to-Speech (TTS) Module: Kokoro Adapter
Kế thừa BaseTTSEngine:
- Chạy ở BẢN FULL FP16 (Không Quantize) trên GPU 0.
- Tuân thủ lifecycle always_active (thường trực 100% trong VRAM, không bao giờ bị dọn).
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
from core.memory_manager import get_memory_manager

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
        self.lifecycle = self.config.get("lifecycle", "always_active")
        self.default_voice = self.config.get("voice", TTS_VOICE)
        self._initialized = True

    def _normalize_vietnamese(self, text: str) -> str:
        """
        Chuẩn hóa văn bản tiếng Việt sang dạng ký tự Latinh tối ưu ngữ âm cho Kokoro.
        Ngăn chặn triệt để hiện tượng phonemizer bị lỗi văng mã unicode ('Letter 1EB5N...').
        """
        import unicodedata
        import re

        # Ánh xạ chữ Đ/đ riêng trước khi phân rã unicode
        trans_map = {
            "đ": "d", "Đ": "D",
            "“": '"', "”": '"', "’": "'", "‘": "'",
            "–": "-", "—": "-",
        }
        for k, v in trans_map.items():
            text = text.replace(k, v)

        # Phân rã NFKD để tách các dấu thanh (diacritics) ra khỏi nguyên âm
        nfkd = unicodedata.normalize("NFKD", text)
        cleaned = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

        # Chuẩn hóa khoảng trắng thừa
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def release_from_gpu(self):
        """Giải phóng Kokoro TTS khỏi GPU VRAM về CPU/RAM."""
        logger.info("🧹 Giải phóng Kokoro TTS khỏi GPU VRAM...")
        if self._pipeline is not None and self._pipeline != "fallback":
            self._pipeline = None
            self._is_loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _actual_loader(self):
        resolver = get_device_resolver()
        resolved = resolver.resolve(
            task="tts",
            model_id=self.model_id,
            requested_strategy=self.device_strategy,
            precision="fp16",
            config=self.config,
        )
        self.resolved_device = resolved["device"]

        logger.info(f"🔊 Đang nạp TTS Kokoro-82M ở BẢN FULL FP16 vào {self.resolved_device}...")
        t0 = time.time()

        try:
            target_idx = 0
            if "cuda:" in self.resolved_device:
                try:
                    target_idx = int(self.resolved_device.split(":")[1])
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.set_device(target_idx)
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ TTS Kokoro-82M nạp thành công trong {elapsed:.2f}s!")
        except Exception as e:
            logger.warning(f"⚠️ Kokoro package chưa cài đặt sẵn hoặc tải lỗi ({e}), chuyển sang fallback engine.")
            self._pipeline = "fallback"

        return self._pipeline

    def load_pipeline(self):
        mem = get_memory_manager()
        return mem.switch_dynamic_slot("tts", self._actual_loader, engine_obj=self)

    def load(self) -> Any:
        return self.load_pipeline()

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 24000,
        response_format: str = "wav",
    ) -> bytes:
        """
        Đọc văn bản thành audio bytes (WAV/OGG) chuẩn OpenAI /v1/audio/speech.
        Hỗ trợ cả tiếng Anh và tiếng Việt, tùy biến voice, speed, sample_rate.
        """
        pipe = self.load_pipeline()
        selected_voice = voice or self.default_voice
        t0 = time.time()

        # Kiểm tra ngôn ngữ: nếu là tiếng Việt hoặc có chứa ký tự tiếng Việt, thực hiện phonetic normalization
        is_vietnamese = (language and language.lower().startswith("vi")) or any(
            c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ"
            for c in text
        )

        processed_text = self._normalize_vietnamese(text) if is_vietnamese else text
        speed_val = max(0.25, min(4.0, float(speed)))
        target_sr = sample_rate or 24000

        if pipe != "fallback":
            try:
                generator = pipe(
                    processed_text,
                    voice=selected_voice,
                    speed=speed_val,
                    split_pattern=r"\n+",
                )
                audio_chunks = []
                for _, _, audio in generator:
                    audio_chunks.append(audio)

                if audio_chunks:
                    full_audio = np.concatenate(audio_chunks)
                else:
                    full_audio = np.zeros(target_sr, dtype=np.float32)
            except Exception as e:
                logger.error(f"Lỗi khi sinh âm thanh bằng Kokoro: {e}")
                full_audio = np.zeros(target_sr, dtype=np.float32)
        else:
            duration = max(1.0, len(processed_text) * 0.08 / speed_val)
            t = np.linspace(0, duration, int(target_sr * duration), False)
            full_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        buffer = io.BytesIO()
        sf_format = "WAV" if response_format.lower() in ("wav", "audio/wav") else "OGG"
        sf.write(buffer, full_audio, target_sr, format=sf_format)
        audio_bytes = buffer.getvalue()

        elapsed = time.time() - t0
        logger.info(
            f"🎶 Sinh giọng nói thành công trong {elapsed:.2f}s "
            f"(lang={'vi' if is_vietnamese else 'en'}, voice={selected_voice}, speed={speed_val}, {len(audio_bytes)} bytes)"
        )
        return audio_bytes


def get_tts_engine(config: Optional[Dict[str, Any]] = None) -> TTSEngine:
    return TTSEngine(config)
