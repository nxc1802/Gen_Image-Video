"""
🖼️ Image Generation Module: FLUX.1 Adapter (4-bit NF4)
Kế thừa BaseImageEngine:
- Nạp trực tiếp Transformer & T5 text encoder ở chế độ 4-bit NF4 (low_cpu_mem_usage=True).
- Điều phối VRAM qua Unified Memory Orchestrator để không gây OOM.
"""

import base64
import io
import logging
import time
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel

from config import FLUX_MODEL_ID, FLUX_CONFIG, FLUX_NUM_STEPS, FLUX_GUIDANCE, DEVICE_VISUAL
from core.base_engine import BaseImageEngine
from core.device_resolver import get_device_resolver
from core.memory_manager import get_memory_manager

logger = logging.getLogger("FluxImageEngine")


class FluxImageEngine(BaseImageEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FluxImageEngine, cls).__new__(cls)
            cls._pipe = None
            cls._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or FLUX_CONFIG)
        self.model_id = self.config.get("id", FLUX_MODEL_ID)
        self.device_strategy = self.config.get("device_strategy", "gpu_1")
        self.steps = int(self.config.get("steps", FLUX_NUM_STEPS))
        self.guidance = float(self.config.get("guidance", FLUX_GUIDANCE))
        self._initialized = True

    def _loader(self):
        resolver = get_device_resolver()
        resolved = resolver.resolve("image", self.model_id, self.device_strategy)
        self.resolved_device = resolved["device"]

        logger.info(f"🖼️ Đang nạp FLUX.1 ({self.model_id}) 4-bit NF4 vào {self.resolved_device}...")
        t0 = time.time()

        try:
            bnb_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            logger.info("🖼️ Đang nạp Transformer 4-bit NF4...")
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=bnb_4bit,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            logger.info("🖼️ Đang nạp T5 text_encoder_2 4-bit...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                quantization_config=bnb_4bit,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            logger.info("🖼️ Khởi tạo FluxPipeline kết hợp...")
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.float16,
            )

            if self.resolved_device.startswith("cuda"):
                pipe.enable_model_cpu_offload(device=torch.device(self.resolved_device))
            else:
                pipe.to("cpu")

            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ FLUX.1 nạp thành công vào VRAM trong {elapsed:.2f}s!")
            return pipe
        except Exception as e:
            logger.error(f"❌ Không thể nạp FLUX.1 ({e}), kích hoạt mock generator.")
            return "fallback"

    def get_pipeline(self):
        mem = get_memory_manager()
        return mem.switch_heavyweight_slot("flux", self._loader)

    def load(self) -> Any:
        return self.get_pipeline()

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Sinh ảnh từ prompt văn bản. Trả về (base64_png, elapsed_seconds)."""
        pipe = self.get_pipeline()
        t0 = time.time()

        try:
            w, h = map(int, size.lower().split("x"))
        except Exception:
            w, h = 1024, 1024

        step_count = steps or self.steps
        guide_val = guidance if guidance is not None else self.guidance

        if pipe != "fallback":
            generator = None
            if seed is not None:
                dev = self.resolved_device if self.resolved_device.startswith("cuda") else "cpu"
                generator = torch.Generator(device=dev).manual_seed(seed)

            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=step_count,
                    guidance_scale=guide_val,
                    generator=generator,
                ).images[0]
        else:
            image = Image.new("RGB", (w, h), color=(30, 30, 40))

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        elapsed = time.time() - t0
        logger.info(f"🎨 Sinh ảnh FLUX.1 hoàn tất trong {elapsed:.2f}s ({w}x{h}, {step_count} steps)")
        return b64_str, round(elapsed, 2)


def get_flux_engine(config: Optional[Dict[str, Any]] = None) -> FluxImageEngine:
    return FluxImageEngine(config)
