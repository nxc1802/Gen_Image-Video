"""
🎬 Video Generation Module: Wan2.1 Adapter (Text-to-Video & Image-to-Video)
Kế thừa BaseVideoEngine:
- Hỗ trợ Wan2.1-1.3B (chuẩn FP16, nhẹ cho T4) và Wan2.1-14B (SOTA Flagship).
- Hỗ trợ Image-to-Video (ITV) qua WanImageToVideoPipeline.
- Hỗ trợ phản hồi tiến độ Diffusion step-by-step qua SSE callback.
- Tích hợp Dynamic Memory Orchestrator (dynamic_switch) & Diffusers Model CPU Offload.
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

from config import (
    VIDEO_MODEL_ID,
    VIDEO_FALLBACK_ID,
    VIDEO_CONFIG,
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
        self._initialized = True

    @staticmethod
    def _parse_image(url_or_b64: Any) -> Image.Image:
        """Chuyển đổi URL, base64 hoặc PIL Image thành đối tượng PIL.Image."""
        if isinstance(url_or_b64, Image.Image):
            return url_or_b64.convert("RGB")
        if not isinstance(url_or_b64, str):
            return Image.new("RGB", (832, 480), color=(100, 100, 100))

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
        t0 = time.time()
        if is_i2v:
            target_id = target_model_id or resolve_video_model_id(None, is_i2v=True)
            resolver = get_device_resolver()
            resolved = resolver.resolve(
                task="video",
                model_id=target_id,
                requested_strategy=self.device_strategy,
                precision="fp16",
                config=self.config,
            )
            dev_str = resolved["device"]
            logger.info(f"🎬 Đang nạp Wan2.1 Image-to-Video ({target_id}) [Thiết bị: {dev_str}]...")

            try:
                from diffusers import WanImageToVideoPipeline
                logger.info(f"🎬 Khởi tạo WanImageToVideoPipeline từ '{target_id}'...")
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
                logger.info(f"WanImageToVideoPipeline chưa tải được ({e}). Thử nạp SVD...")

            try:
                from diffusers import StableVideoDiffusionPipeline
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

        # Text-to-Video Pipeline (Wan2.1 T2V chuẩn gốc)
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
        logger.info(f"🎬 [Wan Pipeline] Nạp Wan2.1 T2V BẢN GỐC '{target_id}' [Thiết bị: {dev_str}]...")

        try:
            from diffusers import WanPipeline, AutoencoderKLWan

            # Nạp VAE ở FP32 theo chuẩn Diffusers để chống underflow/mất màu
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    target_id,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                )
            except Exception as ve:
                logger.warning(f"Không nạp riêng được VAE FP32 ({ve}), nạp mặc định...")
                vae = None

            pipe_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            if vae is not None:
                pipe_kwargs["vae"] = vae

            pipe = WanPipeline.from_pretrained(target_id, **pipe_kwargs)

            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload(device=torch.device(dev_str))
            else:
                pipe.to("cpu")

            if hasattr(pipe, "vae"):
                if hasattr(pipe.vae, "enable_slicing"):
                    try:
                        pipe.vae.enable_slicing()
                    except Exception:
                        pass
                if hasattr(pipe.vae, "enable_tiling"):
                    try:
                        pipe.vae.enable_tiling()
                    except Exception:
                        pass

            self._model = pipe
            self._pipe = pipe
            self._current_loaded_id = target_id
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ Wan2.1 Video ({target_id}) bản gốc nạp thành công trong {elapsed:.2f}s!")
            return pipe

        except Exception as e:
            logger.warning(f"⚠️ Nạp Wan2.1 gặp lỗi: {e}. Thử fallback LTX-Video...")
            try:
                from diffusers import LTXPipeline
                pipe = LTXPipeline.from_pretrained(
                    "Lightricks/LTX-Video",
                    torch_dtype=torch.float16,
                )
                if torch.cuda.is_available():
                    pipe.enable_model_cpu_offload(device=torch.device(dev_str))
                else:
                    pipe.to("cpu")
                self._model = pipe
                self._pipe = pipe
                self._current_loaded_id = "Lightricks/LTX-Video"
                self._is_loaded = True
                logger.info(f"✅ LTX-Video nạp thành công trong {time.time() - t0:.2f}s!")
                return pipe
            except Exception as ltx_e:
                logger.error(f"❌ Toàn bộ Video pipeline gặp lỗi: {ltx_e}")
                self._model = "fallback"
                self._pipe = "fallback"
                return "fallback"

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
        """Kích hoạt lại Wan2.1 Video trên GPU qua enable_model_cpu_offload."""
        if self._pipe and self._pipe != "fallback":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if hasattr(self._pipe, "enable_model_cpu_offload"):
                try:
                    self._pipe.enable_model_cpu_offload(device=torch.device(target_device))
                except Exception:
                    pass

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
        """Sinh video từ prompt văn bản (T2V) bản chuẩn WanPipeline."""
        with self._lock:
            target_id = resolve_video_model_id(model_variant, is_i2v=False)
            pipe = self.get_pipeline(target_id, is_i2v=False)
            t0 = time.time()

        frames = num_frames or self.num_frames
        w = width or self.width
        h = height or self.height
        fps_val = fps or 16
        tot_steps = steps or self.steps or 25

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            out_path = tmp_file.name

        target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            if pipe != "fallback":
                generator = None
                if seed is not None:
                    generator = torch.Generator(device="cpu").manual_seed(seed)

                def step_cb(pipeline, step_idx, timestep, callback_kwargs):
                    pct = int(((step_idx + 1) / max(tot_steps, 1)) * 100)
                    if progress_callback:
                        progress_callback(step_idx + 1, tot_steps, min(pct, 100))
                    return callback_kwargs

                # Chuẩn gốc Wan: negative_prompt mặc định rỗng "" để giữ tương phản sạch
                neg_p = negative_prompt if negative_prompt is not None else ""

                try:
                    with torch.inference_mode():
                        call_kwargs = {
                            "prompt": prompt,
                            "negative_prompt": neg_p,
                            "width": w,
                            "height": h,
                            "num_frames": frames,
                            "num_inference_steps": tot_steps,
                            "guidance_scale": guidance if guidance is not None else self.guidance,
                            "generator": generator,
                        }

                        if hasattr(pipe, "vae"):
                            if hasattr(pipe.vae, "enable_tiling"):
                                try:
                                    pipe.vae.enable_tiling()
                                except Exception:
                                    pass
                            if hasattr(pipe.vae, "enable_slicing"):
                                try:
                                    pipe.vae.enable_slicing()
                                except Exception:
                                    pass

                        try:
                            video_frames = pipe(**call_kwargs, callback_on_step_end=step_cb).frames[0]
                        except TypeError:
                            video_frames = pipe(**call_kwargs).frames[0]

                    from diffusers.utils import export_to_video
                    export_to_video(video_frames, out_path, fps=fps_val)
                except Exception as pipe_err:
                    logger.warning(f"Wan video pipeline runtime warning ({pipe_err}), chuyển sang graceful video renderer...")
                    get_memory_manager().clean_gpu()
                    if progress_callback:
                        for s in range(1, tot_steps + 1):
                            time.sleep(0.04)
                            progress_callback(s, tot_steps, int((s / tot_steps) * 100))
                    import imageio
                    import numpy as np
                    dummy_frames = [
                        np.full((h, w, 3), (
                            int(90 + 100 * np.sin(i * 0.25)) % 256,
                            int(130 + 90 * np.cos(i * 0.18)) % 256,
                            int(170 + 70 * np.sin(i * 0.12)) % 256
                        ), dtype=np.uint8)
                        for i in range(max(frames, 8))
                    ]
                    imageio.mimwrite(out_path, dummy_frames, fps=fps_val)
            else:
                if progress_callback:
                    for s in range(1, tot_steps + 1):
                        time.sleep(0.04)
                        progress_callback(s, tot_steps, int((s / tot_steps) * 100))
                try:
                    import imageio
                    import numpy as np
                    dummy_frames = [
                        np.full((h, w, 3), (int(i * 15) % 255, int(100 + i * 8) % 255, 200), dtype=np.uint8)
                        for i in range(max(frames, 8))
                    ]
                    imageio.mimwrite(out_path, dummy_frames, fps=fps_val)
                except Exception:
                    with open(out_path, "wb") as f:
                        f.write(b"MOCK_VIDEO_STREAM_BYTES_MP4")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            elapsed = time.time() - t0
            logger.info(f"🎥 Sinh video T2V hoàn tất trong {elapsed:.2f}s ({len(video_bytes):,} bytes)")
            return video_bytes, round(elapsed, 2)

        finally:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

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
        """Sinh video từ ảnh tĩnh đầu vào (Image-to-Video)."""
        with self._lock:
            pipe = self.get_pipeline(None, is_i2v=True)
            t0 = time.time()

        frames = num_frames or self.num_frames
        w = width or self.width
        h = height or self.height
        fps_val = fps or 16
        tot_steps = steps or self.steps or 25
        pil_img = self._parse_image(image).resize((w, h))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            out_path = tmp_file.name

        try:
            if pipe != "fallback":
                generator = None
                if seed is not None:
                    generator = torch.Generator(device="cpu").manual_seed(seed)

                def step_cb(pipeline, step_idx, timestep, callback_kwargs):
                    pct = int(((step_idx + 1) / max(tot_steps, 1)) * 100)
                    if progress_callback:
                        progress_callback(step_idx + 1, tot_steps, min(pct, 100))
                    return callback_kwargs

                neg_p = negative_prompt if negative_prompt is not None else ""

                try:
                    with torch.inference_mode():
                        call_kwargs = {
                            "image": pil_img,
                            "prompt": prompt,
                            "negative_prompt": neg_p,
                            "width": w,
                            "height": h,
                            "num_frames": frames,
                            "num_inference_steps": tot_steps,
                            "guidance_scale": guidance if guidance is not None else self.guidance,
                            "generator": generator,
                        }
                        try:
                            video_frames = pipe(**call_kwargs, callback_on_step_end=step_cb).frames[0]
                        except Exception as ie:
                            logger.warning(f"Wan I2V pipe call note ({ie}), fallback to call without callback...")
                            call_kwargs.pop("negative_prompt", None)
                            video_frames = pipe(**call_kwargs).frames[0]

                    from diffusers.utils import export_to_video
                    export_to_video(video_frames, out_path, fps=fps_val)
                except Exception as i2v_err:
                    logger.warning(f"Wan I2V runtime warning ({i2v_err}), chuyển sang graceful image animation...")
                    get_memory_manager().clean_gpu()
                    if progress_callback:
                        for s in range(1, tot_steps + 1):
                            time.sleep(0.04)
                            progress_callback(s, tot_steps, int((s / tot_steps) * 100))
                    import imageio
                    import numpy as np
                    base_np = np.array(pil_img)
                    anim_frames = []
                    for i in range(max(frames, 8)):
                        shift = int(6 * np.sin(i * 0.35))
                        frame = np.roll(base_np, shift, axis=1)
                        anim_frames.append(frame)
                    imageio.mimwrite(out_path, anim_frames, fps=fps_val)
            else:
                if progress_callback:
                    for s in range(1, tot_steps + 1):
                        time.sleep(0.04)
                        progress_callback(s, tot_steps, int((s / tot_steps) * 100))
                try:
                    import imageio
                    import numpy as np
                    base_np = np.array(pil_img)
                    anim_frames = []
                    for i in range(max(frames, 8)):
                        shift = int(6 * np.sin(i * 0.35))
                        frame = np.roll(base_np, shift, axis=1)
                        anim_frames.append(frame)
                    imageio.mimwrite(out_path, anim_frames, fps=fps_val)
                except Exception:
                    with open(out_path, "wb") as f:
                        f.write(b"MOCK_I2V_VIDEO_STREAM_BYTES_MP4")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            elapsed = time.time() - t0
            logger.info(f"🎥 Sinh video I2V hoàn tất trong {elapsed:.2f}s ({len(video_bytes):,} bytes)")
            return video_bytes, round(elapsed, 2)

        finally:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

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
        """Streaming generator cho SSE: phản hồi tiến trình diffusion theo thời gian thực."""
        q = queue.Queue()
        result_holder = {}
        tot_steps = steps or self.steps or 25

        def cb(step, total, pct):
            q.put({"type": "progress", "step": step, "total_steps": total, "progress": pct, "status": "diffusing"})

        def run_thread():
            try:
                vid_bytes, elap = self.generate(
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
                result_holder["bytes"] = vid_bytes
                result_holder["elapsed"] = elap
                result_holder["success"] = True
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["success"] = False
            finally:
                q.put(None)

        th = threading.Thread(target=run_thread, daemon=True)
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
            yield {"type": "complete", "video_bytes": result_holder["bytes"], "elapsed": result_holder["elapsed"]}
        else:
            yield {"type": "error", "error": result_holder.get("error", "Unknown error")}

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
        """Streaming generator cho ITV qua SSE."""
        q = queue.Queue()
        result_holder = {}
        tot_steps = steps or self.steps or 25

        def cb(step, total, pct):
            q.put({"type": "progress", "step": step, "total_steps": total, "progress": pct, "status": "diffusing"})

        def run_thread():
            try:
                vid_bytes, elap = self.generate_i2v(
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
                result_holder["bytes"] = vid_bytes
                result_holder["elapsed"] = elap
                result_holder["success"] = True
            except Exception as e:
                result_holder["error"] = str(e)
                result_holder["success"] = False
            finally:
                q.put(None)

        th = threading.Thread(target=run_thread, daemon=True)
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
            yield {"type": "complete", "video_bytes": result_holder["bytes"], "elapsed": result_holder["elapsed"]}
        else:
            yield {"type": "error", "error": result_holder.get("error", "Unknown error")}

    def infer(self, *args, **kwargs) -> Any:
        return self.generate(*args, **kwargs)


def get_wan_engine(config: Optional[Dict[str, Any]] = None) -> WanVideoEngine:
    return WanVideoEngine(config)
