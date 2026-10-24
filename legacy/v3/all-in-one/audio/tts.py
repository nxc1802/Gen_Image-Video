"""
🔊 Text-to-Speech (TTS) Module: Qwen3-TTS & XTTS-v2 Vietnamese (với AudioService Gateway)
================================================================================
Kế thừa BaseTTSEngine:
- Engine Multilingual & VoiceDesign chính: Qwen3-TTS 1.7B
- Engine chuyên biệt tiếng Việt + Voice Cloning: XTTS-v2 Vietnamese
- Phục vụ qua FastAPI Audio Service daemon trên GPU 0 (cuda:0)
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
from core.memory_manager import get_memory_manager
from .audio_manager import get_audio_service_manager

logger = logging.getLogger("TTSEngine")


class TTSEngine(BaseTTSEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TTSEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or TTS_CONFIG)
        self.model_id = self.config.get("id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.xtts_model_id = self.config.get("xtts_id", "thivux/XTTS-v2-vietnamse")
        self.device_strategy = self.config.get("device_strategy", "gpu_0")
        self.lifecycle = self.config.get("lifecycle", "always_active")
        self.default_voice = self.config.get("voice", "default")
        self.audio_manager = get_audio_service_manager()
        self._initialized = True

    def release_from_gpu(self):
        """Giải phóng TTS khỏi GPU VRAM."""
        logger.info("🧹 Giải phóng TTS khỏi GPU VRAM...")
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

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = "vi",
        speed: float = 1.0,
        sample_rate: int = 24000,
        response_format: str = "wav",
        speaker_wav: Optional[str] = None,
        voice_description: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: Optional[bool] = None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        gpt_cond_len: Optional[int] = None,
        max_ref_len: Optional[int] = None,
        sound_norm_refs: Optional[bool] = None,
        enable_text_splitting: Optional[bool] = None,
    ) -> bytes:
        """
        Đọc văn bản thành audio bytes (WAV/OGG) chuẩn OpenAI /v1/audio/speech.
        Hỗ trợ:
        1. Qwen3-TTS 1.7B: Đa ngôn ngữ, VoiceDesign, CustomVoice, Zero-shot voice cloning.
        2. XTTS-v2 Vietnamese: Chuyên biệt giọng đọc tiếng Việt + Cross-language voice cloning.
        3. Kokoro-82M: Engine dự phòng.
        """
        self.load()
        t0 = time.time()
        target_model = model or engine or ""

        # Kiểm tra nếu chỉ định Kokoro cụ thể
        if "kokoro" in target_model.lower():
            try:
                from kokoro import KPipeline
                pipe = KPipeline(lang_code="a")
                generator = pipe(text, voice=voice or "af_heart", speed=speed)
                chunks = [audio for _, _, audio in generator]
                if chunks:
                    full_audio = np.concatenate(chunks)
                    buf = io.BytesIO()
                    sf.write(buf, full_audio, sample_rate, format="WAV")
                    return buf.getvalue()
            except Exception as e:
                logger.warning(f"Kokoro fallback gặp lỗi: {e}, tiếp tục chuyển sang Qwen3-TTS/XTTS.")

        # Định tuyến thông minh giữa XTTS-v2 và Qwen3-TTS
        target = target_model.lower()
        if not target and voice:
            if "xtts" in voice.lower():
                target = "xtts"
            elif "qwen" in voice.lower():
                target = "qwen"
            elif "kokoro" in voice.lower():
                target = "kokoro"

        is_xtts = (
            "xtts" in target
            or (not target and bool(speaker_wav) and (not language or language.lower() in ("vi", "vietnamese")))
        )

        if is_xtts:
            audio_bytes = self.audio_manager.synthesize_xtts(
                text=text,
                language=language or "vi",
                speaker_wav=speaker_wav or prompt_audio,
                speed=speed,
                sample_rate=sample_rate,
                response_format=response_format,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                gpt_cond_len=gpt_cond_len,
                max_ref_len=max_ref_len,
                sound_norm_refs=sound_norm_refs,
                enable_text_splitting=enable_text_splitting,
            )
            eng_name = "XTTS-v2 Vietnamese"
        else:
            audio_bytes = self.audio_manager.synthesize_qwen(
                text=text,
                language=language or "vi",
                voice=voice or self.default_voice,
                voice_description=voice_description,
                prompt_audio=prompt_audio,
                speaker_wav=speaker_wav,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                speed=speed,
                sample_rate=sample_rate,
                response_format=response_format,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            eng_name = "Qwen3-TTS 1.7B"

        elapsed = time.time() - t0
        logger.info(
            f"🎶 [{eng_name}] Sinh giọng nói thành công trong {elapsed:.2f}s "
            f"(lang={language}, {len(audio_bytes)} bytes)"
        )
        return audio_bytes


def get_tts_engine(config: Optional[Dict[str, Any]] = None) -> TTSEngine:
    return TTSEngine(config)
