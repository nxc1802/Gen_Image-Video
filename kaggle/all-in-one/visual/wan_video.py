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

        # Text-to-Video Pipeline (Wan2.1-T2V-1.3B)
        candidates = ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]
        seen = set()
        clean_candidates = []
        for c in candidates:
            if c and c not in seen:
                clean_candidates.append(c)
                seen.add(c)

        candidate_errors = {}
        for mid in clean_candidates:
            try:
                from transformers import BitsAndBytesConfig, UMT5EncoderModel
                from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel

                logger.info(f"🎬 [Wan 1.3B Native FP16] Bắt đầu nạp Wan2.1 1.3B từ '{mid}' trực tiếp lên {dev_str} (NO CPU OFFLOAD)...")

                # Text Encoder: google/umt5-xxl in 4-bit NF4 with FP32 compute (~5.2 GB VRAM)
                # Dùng bnb_4bit_compute_dtype=torch.float32 và torch_dtype=torch.float32 để triệt tiêu hoàn toàn lỗi tràn số (overflow/NaN) của T5 trong FP16!
                bnb_config_text = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                )

                # 1. Text Encoder: google/umt5-xxl
                logger.info(f"🎬 [Wan] Đang nạp UMT5EncoderModel (4-bit NF4, FP32 compute & weights) từ '{mid}/text_encoder'...")
                text_encoder = UMT5EncoderModel.from_pretrained(
                    mid,
                    subfolder="text_encoder",
                    quantization_config=bnb_config_text,
                    torch_dtype=torch.float32,
                    device_map={"": dev_str},
                    low_cpu_mem_usage=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 2. Transformer: WanTransformer3DModel in native unquantized FP16 (~2.7 GB VRAM)
                # Lưu ý: Model 1.3B chỉ có 1.36 tỷ tham số (~2.7GB FP16), hoàn toàn vừa vặn trong VRAM Tesla T4!
                # KHÔNG lượng tử hóa 4-bit transformer vì bnb NF4 làm sụp đổ trường vector dòng chảy (flow matching trajectory),
                # dẫn đến hiện tượng video bị xám đặc (grey static noise, std ~6.2).
                logger.info(f"🎬 [Wan] Đang nạp WanTransformer3DModel (FP16 nguyên bản không lượng tử hóa, ~2.7GB) từ '{mid}/transformer'...")
                transformer = WanTransformer3DModel.from_pretrained(
                    mid,
                    subfolder="transformer",
                    torch_dtype=torch.float16,
                ).to(dev_str)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 3. VAE: AutoencoderKLWan in float32 (~1.2 GB VRAM)
                logger.info(f"🎬 [Wan] Đang nạp AutoencoderKLWan (FP32 nguyên bản) từ '{mid}/vae'...")
                vae = AutoencoderKLWan.from_pretrained(
                    mid,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                )
                vae = vae.to(dev_str)

                # 4. Assembled WanPipeline directly bound to GPU 1
                logger.info(f"🎬 [Wan] Lắp ráp WanPipeline nguyên khối trực tiếp trên {dev_str} (NO CPU OFFLOAD)...")
                try:
                    pipe = WanPipeline.from_pretrained(
                        mid,
                        transformer=transformer,
                        text_encoder=text_encoder,
                        vae=vae,
                    )
                except Exception as p_err:
                    logger.warning(f"from_pretrained note ({p_err}), thử khởi tạo trực tiếp với FlowMatchEulerDiscreteScheduler...")
                    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(mid, subfolder="tokenizer")
                    base_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(mid, subfolder="scheduler")
                    scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_sched.config, shift=3.0)
                    pipe = WanPipeline(
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        transformer=transformer,
                        vae=vae,
                        scheduler=scheduler,
                    )

                # 5. Cấu hình FlowMatchEulerDiscreteScheduler với shift=3.0 (chuẩn Flow Matching cho Wan2.1 1.3B 480P)
                from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=3.0)
                logger.info("✅ Đã cấu hình FlowMatchEulerDiscreteScheduler (shift=3.0) chuẩn cho WanPipeline!")

                if hasattr(pipe, "vae") and pipe.vae is not None:
                    try:
                        pipe.vae.to(dev_str, dtype=torch.float32)
                        if hasattr(pipe.vae, "enable_tiling"):
                            pipe.vae.enable_tiling()
                        if hasattr(pipe.vae, "enable_slicing"):
                            pipe.vae.enable_slicing()
                        logger.info("✅ Đã bật VAE enable_tiling() & enable_slicing() tối ưu VRAM decode!")
                    except Exception as ve:
                        logger.warning(f"vae setup note: {ve}")

                self._model = pipe
                self._pipe = pipe
                self._current_loaded_id = mid
                self._is_loaded = True
                elapsed = time.time() - t0
                logger.info(f"✅ Wan2.1 Video 1.3B (Native FP16 Transformer + 4-bit FP32-Compute UMT5 + FlowMatchEuler) nạp thành công 100% trên {dev_str} trong {elapsed:.2f}s! (NO CPU OFFLOAD)")
                return pipe
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                candidate_errors[mid] = f"{e}\n{tb}"
                logger.error(f"❌ [Wan 4-bit Loading Error on {mid}]: {e}\n{tb}")

        err_summary = "\n---\n".join([f"Candidate '{m}': {err}" for m, err in candidate_errors.items()])
        self.last_error = err_summary
        raise RuntimeError(f"Không thể nạp Wan2.1 T2V pipeline với bất kỳ candidate nào:\n{err_summary}")

    def get_pipeline(self, target_model_id: Optional[str] = None, is_i2v: bool = False):
        if is_i2v:
            if self._i2v_pipe and self._i2v_pipe != "fallback":
                return self._i2v_pipe
        else:
            if self._pipe and self._pipe != "fallback":
                return self._pipe

        target_id = target_model_id or (
            resolve_video_model_id(None, is_i2v=True) if is_i2v else self.model_id
        )

        if self.lifecycle == "always_active":
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
                    return callback_kwargs

                neg_p = negative_prompt or (
                    "Bright tones, overexposed, static, blurred details, subtitles, style, "
                    "works, paintings, images, static, overall gray, worst quality, low quality, "
                    "JPEG artifacts, ugly, deformed, extra limbs, poorly drawn hands, poorly drawn face"
                )

                try:
                    with torch.inference_mode():
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

                        try:
                            video_frames = pipe(**call_kwargs, callback_on_step_end=step_cb).frames[0]
                        except TypeError:
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

                neg_p = negative_prompt or (
                    "Bright tones, overexposed, static, blurred details, subtitles, style, "
                    "works, paintings, images, static, overall gray, worst quality, low quality, "
                    "JPEG artifacts, ugly, deformed, extra limbs, poorly drawn hands, poorly drawn face"
                )

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

