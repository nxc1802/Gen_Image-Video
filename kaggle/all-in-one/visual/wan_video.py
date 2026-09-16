"""
🎬 Video Generation Module: Wan2.1 Adapter (Text-to-Video & Image-to-Video)
Kế thừa BaseVideoEngine:
- Hỗ trợ Wan2.1-1.3B (nhanh, nhẹ cho T4) và Wan2.1-14B (SOTA Flagship).
- Hỗ trợ Image-to-Video (ITV) qua WanImageToVideoPipeline.
- Hỗ trợ phản hồi tiến độ Diffusion step-by-step qua SSE callback.
- Tích hợp Adaptive Dynamic Allocator và Memory Lifecycle Orchestrator (dynamic_switch).
"""

import base64
import io
import logging
import os
import queue
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import requests
import torch
from transformers import BitsAndBytesConfig

# 🛡️ Monkey-patch Hugging Face and Diffusers caching_allocator_warmup
# caching_allocator_warmup attempts to preallocate dummy tensors (5.54GB to 7.47GB)
# using torch.empty(), causing instantaneous CUDA Out of Memory on 16GB Tesla T4!
try:
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
except Exception:
    pass
try:
    import diffusers.models.model_loading_utils
    diffusers.models.model_loading_utils._caching_allocator_warmup = lambda *args, **kwargs: None
except Exception:
    pass

from config import (
    VIDEO_MODEL_ID,
    VIDEO_FALLBACK_ID,
    VIDEO_LOAD_IN_4BIT,
    VIDEO_CONFIG,
    DEVICE_VISUAL,
    resolve_video_model_id,
)
from core.base_engine import BaseVideoEngine
from core.device_resolver import get_device_resolver
from core.memory_manager import get_memory_manager

logger = logging.getLogger("WanVideoEngine")


class WanVideoEngine(BaseVideoEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WanVideoEngine, cls).__new__(cls)
            cls._pipe = None
            cls._i2v_pipe = None
            cls._current_loaded_id = None
            cls._initialized = False
            cls._lock = threading.Lock()
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or VIDEO_CONFIG)
        self.model_id = self.config.get("id", VIDEO_MODEL_ID)
        self.fallback_id = self.config.get("fallback_id", VIDEO_FALLBACK_ID)
        self.device_strategy = self.config.get("device_strategy", "auto")
        self.allocation_policy = self.config.get("allocation_policy", "adaptive")
        self.lifecycle = self.config.get("lifecycle", "dynamic_switch")
        self.num_frames = int(self.config.get("num_frames", 17))
        self.width = int(self.config.get("width", 832))
        self.height = int(self.config.get("height", 480))
        self.steps = int(self.config.get("steps", 25))
        self.guidance = float(self.config.get("guidance", 5.0))
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        self._initialized = True

    @staticmethod
    def _parse_image(url_or_b64: Any) -> Image.Image:
        """Chuyển đổi URL, base64 hoặc PIL Image thành đối tượng PIL.Image."""
        if isinstance(url_or_b64, Image.Image):
            return url_or_b64.convert("RGB")
        if not isinstance(url_or_b64, str):
            return Image.new("RGB", (768, 512), color=(100, 100, 100))

        url_or_b64 = url_or_b64.strip()
        if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
            resp = requests.get(url_or_b64, timeout=20)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        elif url_or_b64.startswith("data:image"):
            _, encoded = url_or_b64.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img_bytes = base64.b64decode(url_or_b64)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _loader(self, target_model_id: Optional[str] = None, is_i2v: bool = False):
        target_id = target_model_id or (
            resolve_video_model_id(None, is_i2v=True) if is_i2v else self.model_id
        )
        resolver = get_device_resolver()
        resolved = resolver.resolve(
            task="video",
            model_id=target_id,
            requested_strategy=self.device_strategy,
            quantization="4bit",
            config=self.config,
        )
        self.resolved_device = resolved["device"]

        logger.info(
            f"🎬 Đang nạp Video Engine ({target_id}) [i2v={is_i2v}] "
            f"[Thiết bị: {self.resolved_device} | Lý do: {resolved['reason']}]..."
        )
        t0 = time.time()

        target_device = 1 if torch.cuda.device_count() > 1 else 0
        dev_str = f"cuda:{target_device}" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            torch.cuda.set_device(target_device)
            torch.cuda.empty_cache()

        if is_i2v:
            # 1. Thử nạp WanImageToVideoPipeline nếu model Wan2.1 I2V khả dụng
            try:
                from diffusers import WanImageToVideoPipeline
                logger.info(f"🎬 Thử nạp WanImageToVideoPipeline từ '{target_id}'...")
                pipe = WanImageToVideoPipeline.from_pretrained(
                    target_id,
                    torch_dtype=torch.float16,
                )
                if torch.cuda.is_available():
                    pipe.enable_model_cpu_offload(device=torch.device(dev_str))
                else:
                    pipe.to("cpu")
                self._model = pipe
                self._i2v_pipe = pipe
                self._current_loaded_id = target_id
                self._is_loaded = True
                logger.info(f"✅ Wan2.1 I2V ({target_id}) nạp thành công trong {time.time() - t0:.2f}s!")
                return pipe
            except Exception as e:
                logger.info(f"WanImageToVideoPipeline chưa tải được ({e}). Thử nạp Stable Video Diffusion Pipeline...")

            # 2. Nạp SVD (Stable Video Diffusion) - Model Image-to-Video chuyên dụng cực nhẹ (~3.5GB FP16)
            try:
                from diffusers import StableVideoDiffusionPipeline
                logger.info("🎬 Đang nạp StableVideoDiffusionPipeline (stabilityai/stable-video-diffusion-img2vid-xt)...")
                svd_id = "stabilityai/stable-video-diffusion-img2vid-xt"
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    svd_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                if torch.cuda.is_available():
                    pipe.enable_model_cpu_offload(device=torch.device(dev_str))
                else:
                    pipe.to("cpu")
                self._model = pipe
                self._i2v_pipe = pipe
                self._current_loaded_id = svd_id
                self._is_loaded = True
                logger.info(f"✅ Stable Video Diffusion (I2V) nạp thành công trong {time.time() - t0:.2f}s!")
                return pipe
            except Exception as se:
                logger.warning(f"⚠️ Không nạp được SVD ({se}), fallback sang Wan T2V pipeline.")

        # Text-to-Video Pipeline (Wan2.1)
        target_id = target_model_id or self.model_id
        resolver = get_device_resolver()
        resolved = resolver.resolve(
            task="video",
            model_id=target_id,
            requested_strategy=self.device_strategy,
            precision="fp16",
            config=self.config,
        )
        dev_str = resolved["device"]
        is_dual_gpu = resolved.get("is_dual_gpu", False)

        logger.info(
            f"🎬 [Wan Pipeline] Nạp '{target_id}' BẢN CHUẨN FP16 "
            f"[Thiết bị: {dev_str} | 2-GPU Dual Scaling: {is_dual_gpu} | Lý do: {resolved['reason']}]..."
        )

        try:
            from diffusers import WanPipeline, AutoencoderKLWan

            load_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }

            # Phân bổ thiết bị: 2 GPU (GPU 0: UMT5, GPU 1: Transformer + VAE) hoặc 1 GPU (CPU Offload)
            if is_dual_gpu:
                logger.info("⚖️ [Wan 2-GPU Scaling] Cấu hình 2 GPU: UMT5 chia tải 2 GPU | GPU 0 -> VAE FP32 (1.2GB) | GPU 1 -> Transformer FP16 (2.6GB)...")
                from transformers import UMT5EncoderModel, AutoTokenizer
                from diffusers import WanTransformer3DModel, AutoencoderKLWan

                # 1. Text Encoder UMT5 chia tải đều trên 2 GPU (GPU 0 & GPU 1)
                logger.info("🎬 [2-GPU] Nạp UMT5EncoderModel FP16 qua device_map='auto' (6GiB trên GPU 0, 6GiB trên GPU 1)...")
                max_mem = {0: "6GiB", 1: "6GiB"}
                text_encoder = UMT5EncoderModel.from_pretrained(
                    target_id,
                    subfolder="text_encoder",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_mem,
                    low_cpu_mem_usage=True,
                )

                # 2. Transformer on GPU 1 (cuda:1)
                logger.info("🎬 [2-GPU] Nạp WanTransformer3DModel FP16 lên cuda:1...")
                transformer = WanTransformer3DModel.from_pretrained(
                    target_id,
                    subfolder="transformer",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to("cuda:1")

                # 3. VAE on GPU 0 (cuda:0) in FP32 (bảo toàn 100% độ tương phản/màu sắc theo chuẩn Diffusers, giải phóng VRAM GPU 1 cho Attention)
                logger.info("🎬 [2-GPU] Nạp AutoencoderKLWan FP32 lên cuda:0 (tận dụng ~8.5GB VRAM trống trên GPU 0)...")
                vae = AutoencoderKLWan.from_pretrained(
                    target_id,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).to("cuda:0")
                try:
                    vae.enable_tiling()
                    logger.info("✅ [Wan 2-GPU] Đã kích hoạt vae.enable_tiling() thành công!")
                except Exception as te:
                    logger.warning(f"⚠️ Không thể kích hoạt enable_tiling trên VAE: {te}")
                try:
                    vae.enable_slicing()
                    logger.info("✅ [Wan 2-GPU] Đã kích hoạt vae.enable_slicing() thành công!")
                except Exception:
                    pass

                # Tự động chuyển latents từ cuda:1 sang cuda:0 và cast sang FP32 khi pipeline gọi vae.decode
                orig_decode = vae.decode
                def _wrapped_decode(z, *args, **kwargs):
                    z = z.to(device=torch.device("cuda:0"), dtype=torch.float32)
                    return orig_decode(z, *args, **kwargs)
                vae.decode = _wrapped_decode

                tokenizer = AutoTokenizer.from_pretrained(target_id, subfolder="tokenizer")

                # Load scheduler: dùng FlowMatchEulerDiscreteScheduler với shift=3.0 (tối ưu 480p, tương thích 100% WanTransformer3D & RoPE)
                try:
                    from diffusers import FlowMatchEulerDiscreteScheduler
                    base_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
                        target_id,
                        subfolder="scheduler",
                        shift=3.0,
                    )
                    logger.info("✅ [Wan 2-GPU] Đã nạp FlowMatchEulerDiscreteScheduler(shift=3.0) thành công!")
                except Exception as fe:
                    logger.warning(f"⚠️ Không nạp được FlowMatchEulerDiscreteScheduler với shift=3.0 ({fe}), nạp mặc định...")
                    try:
                        base_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(target_id, subfolder="scheduler")
                    except Exception:
                        base_sched = None

                self._text_encoder = text_encoder
                self._tokenizer = tokenizer
                self._is_dual_gpu = True

                pipe_kwargs = {
                    "tokenizer": tokenizer,
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                    "vae": vae,
                }
                if base_sched is not None:
                    pipe_kwargs["scheduler"] = base_sched

                pipe = WanPipeline(**pipe_kwargs)
                try:
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                        pipe.vae.enable_tiling()
                except Exception:
                    pass
                # Đảm bảo pipe._execution_device luôn trỏ vào cuda:1 (nơi transformer thực thi diffusion loop)
                pipe.__class__ = type("DualGpuWanPipeline", (pipe.__class__,), {
                    "_execution_device": property(lambda self: torch.device("cuda:1"))
                })
                logger.info("✅ [Wan 2-GPU] Đã kết nối pipeline phân bổ hoàn hảo giữa GPU 0 (UMT5 + VAE FP32) và GPU 1 (UMT5 + Transformer FP16, execution_device: cuda:1)!")
            elif torch.cuda.is_available():
                target_dev_idx = 1 if torch.cuda.device_count() > 1 else 0
                if "cuda:" in dev_str:
                    try:
                        target_dev_idx = int(dev_str.split(":")[1])
                    except Exception:
                        pass
                dev_obj = torch.device(f"cuda:{target_dev_idx}")
                logger.info(f"🎯 [Wan 1-GPU Scaling] Kích hoạt enable_model_cpu_offload({dev_obj}) chuẩn FP16...")
                try:
                    pipe = WanPipeline.from_pretrained(target_id, **load_kwargs)
                except Exception as p_err:
                    vae = AutoencoderKLWan.from_pretrained(target_id, subfolder="vae", torch_dtype=torch.float16)
                    pipe = WanPipeline.from_pretrained(target_id, vae=vae, **load_kwargs)
                pipe.enable_model_cpu_offload(device=dev_obj)
            else:
                pipe = WanPipeline.from_pretrained(target_id, **load_kwargs)
                pipe.to("cpu")

            self._model = pipe
            self._pipe = pipe
            self._current_loaded_id = target_id
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ Wan2.1 Video ({target_id}) nạp thành công ở chuẩn FP16 trong {elapsed:.2f}s!")
            return pipe
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"❌ [Wan Loading Error on {target_id}]: {e}\n{tb}")
            raise RuntimeError(f"Không thể nạp Wan2.1 T2V pipeline ({target_id}): {e}\n{tb}")

    def _encode_prompt_2gpu(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        max_sequence_length: int = 226,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mã hóa prompt văn bản trên GPU 0 bằng UMT5 FP16, sau đó chuyển embeddings sang GPU 1 (cuda:1)."""
        import re, html
        def _clean_text(text: str) -> str:
            if not text:
                return ""
            try:
                import ftfy
                text = ftfy.fix_text(text)
            except Exception:
                pass
            text = html.unescape(html.unescape(text.strip()))
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        tokenizer = getattr(self, "_tokenizer", None) or getattr(self._pipe, "tokenizer", None)
        text_encoder = getattr(self, "_text_encoder", None) or getattr(self._pipe, "text_encoder", None)

        if tokenizer is None or text_encoder is None:
            raise RuntimeError("Không tìm thấy tokenizer hoặc text_encoder cho Wan 2-GPU encoding!")

        def _get_embed(text: str) -> torch.Tensor:
            clean = _clean_text(text)
            text_inputs = tokenizer(
                [clean],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            first_dev = getattr(text_encoder, "device", torch.device("cuda:0"))
            input_ids = text_inputs.input_ids.to(first_dev)
            mask = text_inputs.attention_mask.to(first_dev)
            seq_lens = mask.gt(0).sum(dim=1).long()

            with torch.no_grad():
                out = text_encoder(input_ids=input_ids, attention_mask=mask)
                embeds = out.last_hidden_state.to(device="cuda:1", dtype=dtype)

            embeds = [u[:v.item()] for u, v in zip(embeds, seq_lens)]
            embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in embeds],
                dim=0,
            )
            if torch.isnan(embeds).any() or torch.isinf(embeds).any():
                logger.warning("⚠️ [Prompt Embeds] Phát hiện NaN/Inf trong text encoder embeddings! Tự động nan_to_num...")
                embeds = torch.nan_to_num(embeds, nan=0.0, posinf=0.0, neginf=0.0)

            return embeds.to("cuda:1", dtype=dtype)

        prompt_embeds = _get_embed(prompt)
        neg_text = negative_prompt if negative_prompt is not None else ""
        neg_embeds = _get_embed(neg_text)
        logger.info(
            f"📊 [Embeddings Check] Prompt: shape={prompt_embeds.shape}, mean={prompt_embeds.mean():.4f}, std={prompt_embeds.std():.4f} | "
            f"Neg: shape={neg_embeds.shape}, mean={neg_embeds.mean():.4f}, std={neg_embeds.std():.4f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prompt_embeds, neg_embeds

    def release_from_gpu(self):
        """Giải phóng hoàn toàn Wan2.1 khỏi cả GPU 0 và GPU 1 khi nhường chỗ cho Image."""
        logger.info("🧹 Giải phóng Wan2.1 khỏi GPU 0 và GPU 1...")
        self._pipe = None
        self._model = None
        self._i2v_pipe = None
        self._text_encoder = None
        self._tokenizer = None
        self._is_dual_gpu = False
        self._is_loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_pipeline(self, target_model_id: Optional[str] = None, is_i2v: bool = False):
        target_id = target_model_id or (
            resolve_video_model_id(None, is_i2v=True) if is_i2v else self.model_id
        )

        if self.lifecycle == "always_active":
            if is_i2v and self._i2v_pipe and self._i2v_pipe != "fallback":
                return self._i2v_pipe
            if not is_i2v and self._pipe and self._pipe != "fallback":
                return self._pipe
            return self._loader(target_id, is_i2v=is_i2v)

        mem = get_memory_manager()

        def loader_wrap():
            return self._loader(target_id, is_i2v=is_i2v)

        slot_key = "video"
        return mem.switch_dynamic_slot(slot_key, loader_wrap, engine_obj=self)

    def reload_to_gpu(self):
        """Kích hoạt lại Wan2.1 Video trên GPU."""
        if self._pipe and self._pipe != "fallback":
            target_device = 1 if torch.cuda.device_count() > 1 else 0
            if torch.cuda.is_available():
                torch.cuda.set_device(target_device)

    def load(self) -> Any:
        return self.get_pipeline()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = 16,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[bytes, float]:
        """Sinh video từ prompt văn bản (T2V). Hỗ trợ toàn diện các tham số frames, fps, seed, steps."""
        with self._lock:
            target_id = resolve_video_model_id(model_variant, is_i2v=False)
            pipe = self.get_pipeline(target_id, is_i2v=False)
            t0 = time.time()

        frames = num_frames or self.num_frames
        w = width or self.width
        h = height or self.height
        fps_val = fps or 16
        tot_steps = steps or self.steps or 20

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            out_path = tmp_file.name

        target_device = 1 if torch.cuda.device_count() > 1 else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(target_device)

        try:
            if pipe != "fallback":
                generator = None
                if seed is not None:
                    dev = f"cuda:{target_device}" if torch.cuda.is_available() else "cpu"
                    generator = torch.Generator(device=dev).manual_seed(seed)

                def step_cb(pipeline, step_idx, timestep, callback_kwargs):
                    pct = int(((step_idx + 1) / max(tot_steps, 1)) * 100)
                    if progress_callback:
                        progress_callback(step_idx + 1, tot_steps, min(pct, 100))
                    if step_idx + 1 >= tot_steps and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    return callback_kwargs

                neg_p = negative_prompt if negative_prompt is not None else ""

                try:
                    with torch.inference_mode():
                        if getattr(self, "_is_dual_gpu", False) and getattr(self, "_text_encoder", None) is not None:
                            logger.info("🎬 [Wan 2-GPU] Đang sinh Prompt Embeddings qua UMT5 FP16 trên GPU 0 (max_sequence_length=226)...")
                            p_embeds, neg_embeds = self._encode_prompt_2gpu(prompt, neg_p, max_sequence_length=226)
                            logger.info(f"✅ [Wan 2-GPU] Prompt Embeddings đã chuyển sang GPU 1 (shape={p_embeds.shape}, dtype={p_embeds.dtype})")
                            call_kwargs = {
                                "prompt": None,
                                "negative_prompt": None,
                                "prompt_embeds": p_embeds,
                                "negative_prompt_embeds": neg_embeds,
                                "width": w,
                                "height": h,
                                "num_frames": frames,
                                "num_inference_steps": tot_steps,
                                "guidance_scale": guidance if guidance is not None else 5.0,
                                "generator": generator,
                                "max_sequence_length": 226,
                            }
                        else:
                            call_kwargs = {
                                "prompt": prompt,
                                "negative_prompt": neg_p,
                                "width": w,
                                "height": h,
                                "num_frames": frames,
                                "num_inference_steps": tot_steps,
                                "guidance_scale": guidance if guidance is not None else 5.0,
                                "generator": generator,
                                "max_sequence_length": 226,
                            }

                        # Đảm bảo VAE luôn kích hoạt enable_tiling() và enable_slicing() trước khi decode latents
                        if hasattr(pipe, "vae"):
                            if hasattr(pipe.vae, "enable_tiling"):
                                try:
                                    pipe.vae.enable_tiling()
                                    logger.info("✅ Đã bật pipe.vae.enable_tiling() trước khi sinh video.")
                                except Exception as te:
                                    logger.warning(f"Không thể bật pipe.vae.enable_tiling(): {te}")
                            if hasattr(pipe.vae, "enable_slicing"):
                                try:
                                    pipe.vae.enable_slicing()
                                except Exception:
                                    pass

                        try:
                            video_frames = pipe(**call_kwargs, callback_on_step_end=step_cb).frames[0]
                        except TypeError:
                            try:
                                video_frames = pipe(**call_kwargs).frames[0]
                            except TypeError:
                                call_kwargs.pop("max_sequence_length", None)
                                video_frames = pipe(**call_kwargs).frames[0]

                        # Thống kê phân bố điểm ảnh để kiểm định chất lượng trực tiếp trong log
                        try:
                            import numpy as np
                            if isinstance(video_frames, np.ndarray):
                                f_mean = float(video_frames.mean())
                                f_std = float(video_frames.std())
                                logger.info(f"🎬 [T2V Frame Stats] shape={video_frames.shape}, min={video_frames.min():.2f}, max={video_frames.max():.2f}, mean={f_mean:.2f}, std={f_std:.2f}")
                                if f_std < 10.0:
                                    logger.warning(f"⚠️ [Low Quality Warning] Generated video has low std ({f_std:.2f}), frames may lack contrast!")
                                else:
                                    logger.info(f"🎉 [Quality Check Passed] std={f_std:.2f} confirms rich visual contrast and content!")
                            elif isinstance(video_frames, list) and len(video_frames) > 0:
                                arr0 = np.array(video_frames[0])
                                f_mean = float(arr0.mean())
                                f_std = float(arr0.std())
                                logger.info(f"🎬 [T2V Frame Stats] {len(video_frames)} frames, frame0: mean={f_mean:.2f}, std={f_std:.2f}")
                        except Exception as q_err:
                            logger.debug(f"Frame stats note: {q_err}")

                    try:
                        from diffusers.utils import export_to_video
                        export_to_video(video_frames, out_path, fps=fps_val)
                    except Exception as ve:
                        logger.warning(f"export_to_video warning ({ve}), fallback to imageio...")
                        import imageio
                        imageio.mimwrite(out_path, video_frames, fps=fps_val)
                except Exception as pipe_err:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"❌ [Wan Runtime Error] {pipe_err}\n{tb}")
                    get_memory_manager().clean_gpu()
                    raise RuntimeError(f"Wan video pipeline runtime failed: {pipe_err}\n{tb}") from pipe_err


            with open(out_path, "rb") as f:
                video_bytes = f.read()

            elapsed = time.time() - t0
            logger.info(f"🎥 Sinh video T2V hoàn tất trong {elapsed:.2f}s ({len(video_bytes)} bytes)")
            return video_bytes, round(elapsed, 2)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def generate_i2v(
        self,
        prompt: str,
        image: Any,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = 16,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[bytes, float]:
        """Sinh video từ ảnh tĩnh đầu vào (Image-to-Video). Hỗ trợ SVD và Wan I2V mượt mà."""
        with self._lock:
            target_id = resolve_video_model_id(model_variant, is_i2v=True)
            pipe = self.get_pipeline(target_id, is_i2v=True)
            t0 = time.time()

        frames = num_frames or self.num_frames
        w = width or self.width
        h = height or self.height
        fps_val = fps or 16
        pil_img = self._parse_image(image).resize((w, h))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            out_path = tmp_file.name

        target_device = 1 if torch.cuda.device_count() > 1 else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(target_device)

        try:
            if pipe != "fallback":
                generator = None
                if seed is not None:
                    dev = f"cuda:{target_device}" if torch.cuda.is_available() else "cpu"
                    generator = torch.Generator(device=dev).manual_seed(seed)

                tot_steps = steps or 30
                def step_cb(pipeline, step_idx, timestep, callback_kwargs):
                    pct = int(((step_idx + 1) / max(tot_steps, 1)) * 100)
                    if progress_callback:
                        progress_callback(step_idx + 1, tot_steps, min(pct, 100))
                    if step_idx + 1 >= tot_steps and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return callback_kwargs

                neg_p = negative_prompt if negative_prompt is not None else ""

                try:
                    with torch.inference_mode():
                        pipe_name = pipe.__class__.__name__
                        if "StableVideoDiffusion" in pipe_name:
                            call_kwargs = {
                                "image": pil_img,
                                "num_frames": frames,
                                "fps": fps_val,
                                "decode_chunk_size": 4,
                                "generator": generator,
                                "num_inference_steps": tot_steps,
                            }
                            video_frames = pipe(**call_kwargs).frames[0]
                        elif "WanImageToVideo" in pipe_name or hasattr(pipe, "image_encoder"):
                            call_kwargs = {
                                "image": pil_img,
                                "prompt": prompt,
                                "negative_prompt": neg_p,
                                "width": w,
                                "height": h,
                                "num_frames": frames,
                                "num_inference_steps": tot_steps,
                                "guidance_scale": guidance if guidance is not None else 5.0,
                                "generator": generator,
                            }
                            try:
                                video_frames = pipe(**call_kwargs, callback_on_step_end=step_cb).frames[0]
                            except TypeError:
                                video_frames = pipe(**call_kwargs).frames[0]
                        else:
                            # Fallback sang T2V pipeline nếu không có I2V chuyên dụng (không truyền param image)
                            call_kwargs = {
                                "prompt": prompt,
                                "negative_prompt": neg_p,
                                "width": w,
                                "height": h,
                                "num_frames": frames,
                                "num_inference_steps": tot_steps,
                                "guidance_scale": guidance if guidance is not None else 5.0,
                                "generator": generator,
                            }
                            if steps:
                                call_kwargs["num_inference_steps"] = steps
                            video_frames = pipe(**call_kwargs).frames[0]

                    try:
                        from diffusers.utils import export_to_video
                        export_to_video(video_frames, out_path, fps=fps_val)
                    except Exception as ve:
                        logger.warning(f"export_to_video warning ({ve}), fallback to imageio...")
                        import imageio
                        imageio.mimwrite(out_path, video_frames, fps=fps_val)
                except Exception as i2v_err:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"❌ [Wan I2V Runtime Error] {i2v_err}\n{tb}")
                    get_memory_manager().clean_gpu()
                    raise RuntimeError(f"Wan I2V pipeline runtime failed: {i2v_err}\n{tb}") from i2v_err
            else:
                raise RuntimeError("Wan I2V pipeline is not loaded.")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            elapsed = time.time() - t0
            logger.info(f"🎥 Sinh video I2V hoàn tất trong {elapsed:.2f}s ({len(video_bytes)} bytes)")
            return video_bytes, round(elapsed, 2)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def generate_stream(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = 16,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
    ):
        """Yields tiến độ SSE trong lúc sinh video T2V và kết quả cuối."""
        q = queue.Queue()
        result_holder = {}

        def cb(step, total, pct):
            q.put({"type": "progress", "step": step, "total_steps": total, "progress": pct})

        def run_thread():
            try:
                v_bytes, elapsed = self.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    fps=fps,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    model_variant=model_variant,
                    progress_callback=cb,
                )
                result_holder["success"] = True
                result_holder["video_bytes"] = v_bytes
                result_holder["elapsed"] = elapsed
            except Exception as e:
                result_holder["success"] = False
                result_holder["error"] = str(e)
            finally:
                q.put(None)

        th = threading.Thread(target=run_thread)
        th.start()

        while True:
            try:
                item = q.get(timeout=2.0)
                if item is None:
                    break
                yield item
            except queue.Empty:
                yield {"type": "heartbeat"}

        th.join()
        if result_holder.get("success"):
            yield {
                "type": "complete",
                "video_bytes": result_holder["video_bytes"],
                "elapsed": result_holder["elapsed"],
            }
        else:
            yield {
                "type": "error",
                "error": result_holder.get("error", "Unknown video generation error"),
            }

    def generate_i2v_stream(
        self,
        prompt: str,
        image: Any,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = 16,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
    ):
        """Yields tiến độ SSE trong lúc sinh video I2V và kết quả cuối."""
        q = queue.Queue()
        result_holder = {}

        def cb(step, total, pct):
            q.put({"type": "progress", "step": step, "total_steps": total, "progress": pct})

        def run_thread():
            try:
                v_bytes, elapsed = self.generate_i2v(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    fps=fps,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    model_variant=model_variant,
                    progress_callback=cb,
                )
                result_holder["success"] = True
                result_holder["video_bytes"] = v_bytes
                result_holder["elapsed"] = elapsed
            except Exception as e:
                result_holder["success"] = False
                result_holder["error"] = str(e)
            finally:
                q.put(None)

        th = threading.Thread(target=run_thread)
        th.start()

        while True:
            try:
                item = q.get(timeout=2.0)
                if item is None:
                    break
                yield item
            except queue.Empty:
                yield {"type": "heartbeat"}

        th.join()
        if result_holder.get("success"):
            yield {
                "type": "complete",
                "video_bytes": result_holder["video_bytes"],
                "elapsed": result_holder["elapsed"],
            }
        else:
            yield {
                "type": "error",
                "error": result_holder.get("error", "Unknown I2V generation error"),
            }


def get_wan_engine(config: Optional[Dict[str, Any]] = None) -> WanVideoEngine:
    return WanVideoEngine(config)

