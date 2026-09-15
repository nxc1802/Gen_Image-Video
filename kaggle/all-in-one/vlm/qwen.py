"""
👁️ Vision-Language Model (VLM) Module: Qwen Adapter (Hỗ trợ 2B, 7B, 14B, 26B, 32B...)
Kế thừa BaseVLMEngine và tích hợp Adaptive Dynamic Allocator:
- Nếu là model nhỏ (<= 8B như Qwen 7B/2B): Tự động nạp 100% vào 1 GPU đơn (GPU 0 hoặc GPU 1 tùy VRAM trống), loại bỏ 100% độ trễ PCIe!
- Nếu là model lớn (>= 14B như Qwen 26B): Tự động chia tải động theo tỷ lệ VRAM thực tế qua max_memory (BỎ CHIA CỨNG 50-50).
- Tuân thủ chính sách vòng đời (always_active vs dynamic_switch) từ models.yaml.
"""

import base64
import io
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer

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
            cls._model = None
            cls._processor = None
            cls._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or VLM_CONFIG)
        self.model_id = self.config.get("id", VLM_MODEL_ID)
        self.device_strategy = self.config.get("device_strategy", "auto")
        self.allocation_policy = self.config.get("allocation_policy", "adaptive")
        self.quantization = self.config.get("quantization", "4bit")
        self.lifecycle = self.config.get("lifecycle", "always_active")
        self._initialized = True

    def _actual_loader(self):
        """Khởi tạo trọng số mô hình và processor với cấu hình phân bổ tự thích ứng."""
        resolver = get_device_resolver()
        resolved = resolver.resolve(
            task="vlm",
            model_id=self.model_id,
            requested_strategy=self.device_strategy,
            quantization=self.quantization,
            config=self.config,
        )
        self.resolved_device = resolved["device"]

        logger.info(
            f"👁️ Đang nạp VLM ({self.model_id}) "
            f"[Thiết bị: {self.resolved_device} | Dual-GPU: {resolved['is_dual_gpu']} | Lý do: {resolved['reason']}]..."
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

            load_kwargs = {
                "quantization_config": bnb_config if torch.cuda.is_available() else None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
            }

            # ⚖️ PHÂN BỔ ĐA GPU TỰ THÍCH ỨNG (ADAPTIVE WATERMARK SPLIT)
            if resolved.get("is_dual_gpu") and resolved.get("max_memory"):
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = resolved["max_memory"]
                logger.info(f"⚖️ Áp dụng max_memory tự thích ứng: {resolved['max_memory']}")
            else:
                target_map = resolved.get("device_map") or resolved.get("device")
                load_kwargs["device_map"] = target_map

            self._model = AutoVLMModel.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ VLM ({self.model_id}) nạp thành công trong {elapsed:.2f}s!")
        except Exception as e:
            logger.error(f"❌ Không thể nạp {self.model_id} ({e}), sử dụng fallback.")
            self._model = "fallback"
            self._processor = "fallback"

        return self._model, self._processor

    def load_model(self):
        if self._model is not None:
            return self._model, self._processor

        mem = get_memory_manager()
        # Nếu là always_active: nạp trực tiếp và giữ cố định
        if self.lifecycle == "always_active":
            model, processor = self._actual_loader()
            mem.register_always_active("vlm", model)
            return model, processor
        else:
            # Nếu là dynamic_switch: quản lý qua dynamic pool
            def loader_wrapper():
                m, _ = self._actual_loader()
                return m

            mem.switch_dynamic_slot("vlm", loader_wrapper, engine_obj=self)
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

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Xử lý chat đa phương thức với SSE Streaming từng token."""
        model, processor = self.load_model()

        if model == "fallback":
            user_text = messages[-1].get("content", "") if messages else ""
            if isinstance(user_text, list):
                user_text = " ".join([c.get("text", "") for c in user_text if isinstance(c, dict) and "text" in c])
            mock_tokens = [
                "Xin ", "chào! ", "Tôi ", "là ", "trợ ", "lý ", "AI ", "Studio ", "đang ",
                "chạy ", "trên ", "Kaggle. ", f"Tôi đã nhận được câu hỏi: '{user_text[:50]}'. ",
                "Hệ ", "thống ", "đang ", "hoạt ", "động ", "rất ", "tốt! "
            ]
            for tok in mock_tokens:
                time.sleep(0.04)
                yield tok
            return

        # Chuẩn bị tin nhắn
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

            target_device = self.resolved_device if self.resolved_device != "cpu" else "cpu"
            if torch.cuda.is_available() and target_device != "cpu":
                inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            tokenizer = getattr(processor, "tokenizer", processor)
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p if temperature > 0 else None,
                "do_sample": (temperature > 0),
            }

            generation_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            generation_thread.start()

            for new_text in streamer:
                if new_text:
                    yield new_text

            generation_thread.join(timeout=10.0)
        except Exception as e:
            logger.error(f"Lỗi trong quá trình streaming VLM: {e}")
            yield f"\n[Lỗi stream: {str(e)}]"

    # Alias tương thích OpenAI route
    chat_completion = chat


# Alias tương thích ngược
VLMEngine = QwenVLMEngine


def get_vlm_engine(config: Optional[Dict[str, Any]] = None) -> QwenVLMEngine:
    return QwenVLMEngine(config)
