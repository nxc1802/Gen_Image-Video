"""
🎙️ Speech-to-Text (STT) Module: Whisper-large-v3-turbo
Chạy ở BẢN FULL FP16 (Không Quantize) trên GPU 0.
Giữ 100% độ chính xác nhận diện tiếng Việt và lọc tạp âm.
"""

import io
import logging
import os
import tempfile
import time
from typing import Optional
import torch
import whisper

from config import STT_MODEL_ID, DEVICE_AUDIO, STT_DTYPE

logger = logging.getLogger("STTEngine")


class STTEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(STTEngine, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def load_model(self):
        if self._model is not None:
            return self._model

        logger.info(f"🎙️ Đang nạp STT Whisper ({STT_MODEL_ID}) ở BẢN FULL FP16 vào {DEVICE_AUDIO}...")
        t0 = time.time()
        
        # Nạp mô hình Whisper ở FP16 gốc
        model_name = "large-v3-turbo" if "turbo" in STT_MODEL_ID.lower() else "large-v3"
        self._model = whisper.load_model(model_name, device=DEVICE_AUDIO)
        
        elapsed = time.time() - t0
        logger.info(f"✅ STT Whisper nạp thành công trong {elapsed:.2f}s!")
        return self._model

    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None, prompt: Optional[str] = None) -> dict:
        """
        Nhận diện giọng nói từ mảng bytes âm thanh.
        Trả về kết quả chuẩn OpenAI transcription: {"text": "..."}
        """
        model = self.load_model()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            t0 = time.time()
            transcribe_options = {
                "fp16": (DEVICE_AUDIO != "cpu"),
                "verbose": False,
            }
            if language:
                transcribe_options["language"] = language
            if prompt:
                transcribe_options["initial_prompt"] = prompt

            result = model.transcribe(tmp_path, **transcribe_options)
            elapsed = time.time() - t0
            
            logger.info(f"🎯 Nhận diện hoàn tất trong {elapsed:.2f}s: '{result.get('text', '')[:40]}...'")
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "auto"),
                "duration_seconds": round(elapsed, 2),
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def get_stt_engine() -> STTEngine:
    return STTEngine()
