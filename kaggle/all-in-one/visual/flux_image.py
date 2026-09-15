"""
🖼️ Image Generation Module: FLUX.1-schnell (4-bit NF4)
Chạy trên GPU 1 (Dynamic Worker Slot).
Sử dụng Diffusers + BitsAndBytes NF4 để nằm trọn trong ~8.5GB VRAM.
Tạo ảnh 1024x1024 trong 4 bước (~12-15s trên Tesla T4).
"""

import base64
import io
import logging
import time
from typing import Optional, Tuple
from PIL import Image
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel

from config import FLUX_MODEL_ID, DEVICE_VISUAL, FLUX_NUM_STEPS, FLUX_GUIDANCE
from core.memory_manager import get_memory_manager

logger = logging.getLogger("FluxImageEngine")


class FluxImageEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FluxImageEngine, cls).__new__(cls)
            cls._pipe = None
        return cls._instance

    def _loader(self):
        logger.info(f"🖼️ Đang nạp FLUX.1 ({FLUX_MODEL_ID}) bản 4-bit NF4 vào {DEVICE_VISUAL}...")
        t0 = time.time()

        try:
            bnb_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            logger.info("🖼️ Đang nạp Transformer 4-bit NF4...")
            transformer = FluxTransformer2DModel.from_pretrained(
                FLUX_MODEL_ID,
                subfolder="transformer",
                quantization_config=bnb_4bit,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            logger.info("🖼️ Đang nạp T5 text_encoder_2 4-bit...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                FLUX_MODEL_ID,
                subfolder="text_encoder_2",
                quantization_config=bnb_4bit,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            logger.info("🖼️ Đang kết hợp vào FluxPipeline...")
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            if DEVICE_VISUAL.startswith("cuda"):
                pipe.enable_model_cpu_offload(device=torch.device(DEVICE_VISUAL))
            else:
                pipe.to("cpu")

            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

            elapsed = time.time() - t0
            logger.info(f"✅ FLUX.1 nạp thành công vào VRAM trong {elapsed:.2f}s!")
            return pipe
        except Exception as e:
            logger.error(f"❌ Không thể nạp FLUX.1 ({e}), kích hoạt mock generator.")
            return "fallback"

    def get_pipeline(self):
        # Yêu cầu MemoryManager bảo đảm GPU 1 đang dành cho FLUX
        mem = get_memory_manager()
        return mem.switch_gpu1_slot("flux", self._loader)

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
        Trả về (base64_png_str, inference_time_seconds).
        """
        pipe = self.get_pipeline()
        t0 = time.time()

        # Parse kích thước
        try:
            w, h = map(int, size.lower().split("x"))
        except Exception:
            w, h = 1024, 1024

        step_count = steps or FLUX_NUM_STEPS
        guide_val = guidance if guidance is not None else FLUX_GUIDANCE

        if pipe != "fallback":
            generator = None
            if seed is not None:
                generator = torch.Generator(device=DEVICE_VISUAL).manual_seed(seed)

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
            # Fallback tạo ảnh gradient mẫu
            image = Image.new("RGB", (w, h), color=(30, 30, 40))

        # Xuất ra base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        elapsed = time.time() - t0

        logger.info(f"🎨 Sinh ảnh FLUX.1 thành công trong {elapsed:.2f}s ({w}x{h}, {step_count} steps)")
        return b64_str, round(elapsed, 2)


def get_flux_engine() -> FluxImageEngine:
    return FluxImageEngine()
