"""
🧩 Base Model Engine Template (Abstract Base Classes)
Chuẩn hóa giao diện cho toàn bộ mô hình AI trong hệ thống:
- VLM (Vision-Language)
- Image Generation
- Video Generation
- Speech-to-Text (STT)
- Text-to-Speech (TTS)
Tất cả các mô hình mới chỉ cần kế thừa các lớp này để hoạt động tức thì trong hệ thống.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("BaseModelEngine")


class BaseModelEngine(ABC):
    """Lớp trừu tượng cơ sở cho mọi Model Engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_id = self.config.get("id", "default")
        self.device_strategy = self.config.get("device_strategy", "auto")
        self.quantization = self.config.get("quantization", "none")
        self.resolved_device = "cpu"
        self._model = None
        self._is_loaded = False

    @abstractmethod
    def load(self) -> Any:
        """Khởi tạo mô hình vào bộ nhớ."""
        pass

    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """Thực thi suy luận chuẩn."""
        pass

    def offload_to_cpu(self):
        """Chuyển trọng số sang CPU RAM qua bus PCIe khi rảnh."""
        if self._model is not None and hasattr(self._model, "to"):
            try:
                self._model.to("cpu")
                logger.info(f"💾 [{self.model_id}] Đã chuyển sang CPU RAM.")
            except Exception as e:
                logger.debug(f"Offload to CPU skipped for {self.model_id}: {e}")

    def reload_to_gpu(self, target_device: Optional[str] = None):
        """Kéo trọng số từ CPU RAM trở lại GPU VRAM."""
        device = target_device or self.resolved_device
        if self._model is not None and hasattr(self._model, "to") and device != "cpu":
            try:
                self._model.to(device)
                logger.info(f"⚡ [{self.model_id}] Đã kích hoạt trở lại {device}.")
            except Exception as e:
                logger.debug(f"Reload to GPU skipped for {self.model_id}: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# ==============================================================================
# 1. TEMPLATE CHO VLM / CHAT
# ==============================================================================
class BaseVLMEngine(BaseModelEngine):
    """Template chuẩn cho Vision-Language Models (Qwen, MiniCPM, Llama-Vision...)."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Nhận danh sách messages chuẩn OpenAI (có thể kèm text và ảnh base64/URL).
        Trả về dict: {"text": "...", "tokens": int, "elapsed": float}
        """
        pass

    def infer(self, *args, **kwargs) -> Any:
        return self.chat(*args, **kwargs)


# ==============================================================================
# 2. TEMPLATE CHO TẠO ẢNH (IMAGE GENERATION)
# ==============================================================================
class BaseImageEngine(BaseModelEngine):
    """Template chuẩn cho Image Generation Models (FLUX.1, SD3.5, PixArt...)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Sinh ảnh từ văn bản.
        Trả về tuple: (base64_png_string, elapsed_seconds)
        """
        pass

    def infer(self, *args, **kwargs) -> Any:
        return self.generate(*args, **kwargs)


# ==============================================================================
# 3. TEMPLATE CHO TẠO VIDEO (VIDEO GENERATION)
# ==============================================================================
class BaseVideoEngine(BaseModelEngine):
    """Template chuẩn cho Video Generation Models (Wan2.1, LTX-Video, CogVideoX...)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        num_frames: int = 25,
        width: int = 768,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, float]:
        """
        Sinh video từ văn bản (Text-to-Video).
        Trả về tuple: (video_mp4_bytes, elapsed_seconds)
        """
        pass

    def infer(self, *args, **kwargs) -> Any:
        return self.generate(*args, **kwargs)


# ==============================================================================
# 4. TEMPLATE CHO NHẬN DIỆN ÂM THANH (STT)
# ==============================================================================
class BaseSTTEngine(BaseModelEngine):
    """Template chuẩn cho Speech-to-Text (Whisper, Conformer...)."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Nhận diện giọng nói từ mảng audio bytes.
        Trả về dict chuẩn OpenAI: {"text": "...", "language": "...", "duration_seconds": float}
        """
        pass

    def infer(self, *args, **kwargs) -> Any:
        return self.transcribe(*args, **kwargs)


# ==============================================================================
# 5. TEMPLATE CHO TỔNG HỢP ÂM THANH (TTS)
# ==============================================================================
class BaseTTSEngine(BaseModelEngine):
    """Template chuẩn cho Text-to-Speech (Kokoro, MeloTTS, Bark...)."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        response_format: str = "wav",
    ) -> bytes:
        """
        Đọc văn bản thành audio bytes (WAV/OGG).
        Trả về bytes âm thanh chuẩn.
        """
        pass

    def infer(self, *args, **kwargs) -> Any:
        return self.synthesize(*args, **kwargs)
