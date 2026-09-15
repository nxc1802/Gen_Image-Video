"""
👁️ Vision-Language Model (VLM) Module: Qwen Adapter (Hỗ trợ 2B, 7B, 14B, 26B, 32B...)
Kế thừa BaseVLMEngine và sử dụng DeviceTopologyResolver:
- Nếu là model nhỏ (<= 8B như Qwen 7B/2B): Tự động nạp vào GPU 0 (cuda:0), giải phóng 100% GPU 1 cho Visual models!
- Nếu là model lớn (>= 14B như Qwen 26B): Tự động chia tải song song qua cả 2 GPU (device_map='auto').
- Hỗ trợ đầy đủ định dạng OpenAI /v1/chat/completions (Văn bản + Hình ảnh).
"""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as AutoVLMModel
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoVLMModel

from config import VLM_MODEL_ID, VLM_CONFIG
from core.base_engine import BaseVLMEngine
from core.device_resolver import get_device_resolver
from core.memory_manager import get_memory_manager

logger = logging.getLogger("VLMEngine")


class QwenVLMEngine(BaseVLMEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenVLMEngine, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._processor = None
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or VLM_CONFIG)
        self.model_id = self.config.get("id", VLM_MODEL_ID)
        self.device_strategy = self.config.get("device_strategy", "auto")
        self.quantization = self.config.get("quantization", "4bit")
        self._initialized = True

    def load_model(self):
        if self._model is not None:
            return self._model, self._processor

        resolver = get_device_resolver()
        resolved = resolver.resolve("vlm", self.model_id, self.device_strategy)
        self.resolved_device = resolved["device"]
        device_map = resolved["device_map"] or self.resolved_device

        logger.info(
            f"👁️ Đang nạp VLM ({self.model_id}) "
            f"[Thiết bị: {device_map} | Lý do: {resolved['reason']}]..."
        )
        t0 = time.time()

        # Cấu hình 4-bit NF4 nếu được chỉ định
        bnb_config = None
        if self.quantization == "4bit" and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model = AutoVLMModel.from_pretrained(
                self.model_id,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ VLM ({self.model_id}) nạp thành công trong {elapsed:.2f}s!")
        except Exception as e:
            logger.error(f"❌ Không thể nạp {self.model_id} ({e}), sử dụng fallback.")
            self._model = "fallback"
            self._processor = "fallback"

        return self._model, self._processor

    def load(self) -> Any:
        return self.load_model()

    @staticmethod
    def _parse_image(url_or_b64: str) -> Image.Image:
        """Chuyển đổi URL hoặc chuỗi base64 thành PIL Image."""
        if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
            resp = requests.get(url_or_b64, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        elif url_or_b64.startswith("data:image"):
            header, encoded = url_or_b64.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img_bytes = base64.b64decode(url_or_b64)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Xử lý chat đa phương thức chuẩn OpenAI."""
        model, processor = self.load_model()
        t0 = time.time()

        if model == "fallback":
            user_text = messages[-1].get("content", "") if messages else ""
            if isinstance(user_text, list):
                user_text = " ".join([c.get("text", "") for c in user_text if isinstance(c, dict) and "text" in c])
            mock_reply = f"[Mock VLM Response] Tôi đã nhận được yêu cầu: '{user_text[:80]}'."
            return {"text": mock_reply, "tokens": 20, "elapsed": round(time.time() - t0, 2)}

        # Chuyển đổi định dạng hội thoại cho Hugging Face
        qwen_messages = []
        raw_images = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                qwen_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    p_type = part.get("type", "")
                    if p_type == "text":
                        text_parts.append(part.get("text", ""))
                    elif p_type == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url:
                            try:
                                pil_img = self._parse_image(img_url)
                                raw_images.append(pil_img)
                                text_parts.append("<|image_pad|>")
                            except Exception as e:
                                logger.error(f"Lỗi khi đọc ảnh đầu vào: {e}")

                qwen_messages.append({"role": role, "content": " ".join(text_parts)})

        try:
            text_prompt = processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )

            if raw_images:
                inputs = processor(
                    text=[text_prompt],
                    images=raw_images,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt",
                )

            # Đưa input vào đúng device
            target_device = self.resolved_device if self.resolved_device != "cpu" else "cpu"
            if torch.cuda.is_available() and target_device != "cpu":
                inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    do_sample=(temperature > 0),
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            elapsed = time.time() - t0
            token_count = len(generated_ids[0])
            logger.info(f"✨ VLM sinh {token_count} tokens trong {elapsed:.2f}s ({token_count / max(elapsed, 0.01):.1f} tps)")

            return {
                "text": output_text.strip(),
                "tokens": token_count,
                "elapsed": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"Lỗi suy luận VLM: {e}", exc_info=True)
            return {
                "text": f"Lỗi trong quá trình suy luận VLM: {str(e)}",
                "tokens": 0,
                "elapsed": round(time.time() - t0, 2),
            }

    # Alias tương thích OpenAI route
    chat_completion = chat


# Alias tương thích ngược
VLMEngine = QwenVLMEngine


def get_vlm_engine(config: Optional[Dict[str, Any]] = None) -> QwenVLMEngine:
    return QwenVLMEngine(config)
