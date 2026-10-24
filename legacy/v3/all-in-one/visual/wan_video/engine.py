"""
🎬 ComfyUI-Powered Video Generation Engine (Wan2.2 TI2V-5B, T2V-A14B, I2V-A14B)
================================================================================
Kế thừa BaseVideoEngine, thay thế hoàn toàn cơ chế Diffusers đa GPU cũ:
1. Giao tiếp qua ComfyUI REST API trên GPU chuyên dụng (cuda:1).
2. Tận dụng cơ chế Dynamic VRAM và Offloading tự động của ComfyUI (Zero Multi-GPU OOM).
3. Hỗ trợ đầy đủ:
   - Wan2.2-TI2V-5B (DiT 48-channel latent, T2V & I2V)
   - Wan2.2-T2V-A14B (Two-Stage High/Low Noise MoE)
   - Wan2.2-I2V-A14B (Two-Stage MoE kèm WanFirstLastFrameToVideo)
4. Động hóa 100% tham số (prompt, negative_prompt, width, height, num_frames, steps, cfg, shift, fps, sampler, scheduler, seed).
"""

import base64
import logging
import time
from typing import Any, Dict, Optional, Tuple

from config import MODELS_CONFIG
from core.base_engine import BaseVideoEngine
from core.comfyui_manager import get_comfyui_manager
from .workflow import (
    build_wan22_i2v_a14b_workflow,
    build_wan22_t2v_a14b_workflow,
    build_wan22_ti2v_workflow,
)

logger = logging.getLogger("ComfyUIVideoEngine")


class ComfyUIVideoEngine(BaseVideoEngine):
    """Engine sinh video Wan2.2 vận hành trên ComfyUI backend chuyên biệt 1 GPU."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(cfg)
        self.model_id = cfg.get("id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        from config import resolve_model_gpu_device
        dev_str, gpu_idx = resolve_model_gpu_device("video")
        self.device = dev_str
        self.gpu_device_id = gpu_idx
        self.config = cfg
        self.manager = get_comfyui_manager(cuda_device=self.gpu_device_id)

        self.default_variant = cfg.get("default_variant", "ti2v_5b")
        self.width = cfg.get("width", 832)
        self.height = cfg.get("height", 480)
        self.num_frames = cfg.get("num_frames", 17)
        self.default_steps = cfg.get("steps", 20)
        self.default_cfg = cfg.get("guidance", 5.0)
        self.default_shift = cfg.get("shift", 8.0)
        self.default_fps = cfg.get("fps", 24)
        self.default_sampler = cfg.get("sampler_name", "euler")
        self.default_scheduler = cfg.get("scheduler", "simple")

        logger.info(f"🎬 ComfyUIVideoEngine sẵn sàng: {self.model_id} trên {self.device} (GPU {self.gpu_device_id}, ComfyUI port {self.manager.port})")

    def load_to_gpu(self) -> None:
        """Đảm bảo server ComfyUI đã khởi động."""
        if not self.manager.is_running():
            logger.info("⚡ Đang khởi động ComfyUI cho Video Engine...")
            ok = self.manager.start()
            if not ok:
                raise RuntimeError("ComfyUI server không khởi động thành công")
        self._is_loaded = True

    def load(self) -> Any:
        """Thỏa mãn interface của BaseModelEngine."""
        self.load_to_gpu()
        return self

    def infer(self, *args, **kwargs) -> Any:
        """Thỏa mãn interface của BaseModelEngine."""
        return self.generate(*args, **kwargs)

    def release_from_gpu(self) -> None:
        """Đánh dấu trạng thái."""
        self._is_loaded = False

    def _resolve_params(
        self,
        num_frames: Optional[int],
        width: Optional[int],
        height: Optional[int],
        model_variant: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        rt_cfg = MODELS_CONFIG.get("video", {})
        return {
            "variant": (model_variant or rt_cfg.get("default_variant", self.default_variant)).lower(),
            "negative_prompt": kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", ""),
            "steps_val": kwargs.get("steps") or rt_cfg.get("steps", self.default_steps),
            "cfg_val": kwargs.get("guidance") if kwargs.get("guidance") is not None else rt_cfg.get("guidance", self.default_cfg),
            "shift_val": kwargs.get("shift") if kwargs.get("shift") is not None else rt_cfg.get("shift", self.default_shift),
            "fps_val": kwargs.get("fps") or rt_cfg.get("fps", self.default_fps),
            "sampler": kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler),
            "scheduler": kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler),
            "frames_val": num_frames or rt_cfg.get("num_frames", self.num_frames),
            "width_val": width or rt_cfg.get("width", self.width),
            "height_val": height or rt_cfg.get("height", self.height),
        }

    def generate(
        self,
        prompt: str,
        num_frames: int = 17,
        width: int = 832,
        height: int = 480,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bytes, float]:
        """Sinh video Text-to-Video qua ComfyUI."""
        self.load_to_gpu()
        p = self._resolve_params(num_frames, width, height, model_variant, kwargs)

        # Lựa chọn workflow dựa trên model variant
        if "a14b" in p["variant"] or "14b" in p["variant"]:
            logger.info("🎬 Chọn kiến trúc Wan2.2 T2V-A14B Two-Stage MoE...")
            workflow = build_wan22_t2v_a14b_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                high_steps=kwargs.get("high_steps"),
                low_steps=kwargs.get("low_steps"),
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
            )
        else:
            logger.info("🎬 Chọn kiến trúc Wan2.2 TI2V-5B...")
            workflow = build_wan22_ti2v_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
                start_image_name=None,
            )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Video T2V prompt {prompt_id} tới ComfyUI...")

        outputs = self.manager.wait_for_completion(prompt_id)
        if not outputs:
            raise RuntimeError(f"ComfyUI không xuất video cho prompt {prompt_id}")

        first_out = outputs[0]
        video_bytes = self.manager.get_view_bytes(
            filename=first_out["filename"],
            subfolder=first_out.get("subfolder", ""),
            folder_type=first_out.get("type", "output"),
        )

        elapsed = time.time() - start_t
        logger.info(f"✅ Hoàn thành sinh video T2V trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
        return video_bytes, elapsed

    def generate_i2v(
        self,
        prompt: str,
        image: Any,
        num_frames: int = 17,
        width: int = 832,
        height: int = 480,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bytes, float]:
        """Sinh video Image-to-Video qua ComfyUI."""
        self.load_to_gpu()

        input_filename = self.manager.save_input_image(image)
        p = self._resolve_params(num_frames, width, height, model_variant, kwargs)

        if "a14b" in p["variant"] or "14b" in p["variant"]:
            logger.info("🎬 Chọn kiến trúc Wan2.2 I2V-A14B Two-Stage MoE...")
            workflow = build_wan22_i2v_a14b_workflow(
                start_image_name=input_filename,
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                high_steps=kwargs.get("high_steps"),
                low_steps=kwargs.get("low_steps"),
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
            )
        else:
            logger.info("🎬 Chọn kiến trúc Wan2.2 TI2V-5B (Image Conditioning)...")
            workflow = build_wan22_ti2v_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
                start_image_name=input_filename,
            )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Video I2V prompt {prompt_id} tới ComfyUI...")

        outputs = self.manager.wait_for_completion(prompt_id)
        if not outputs:
            raise RuntimeError(f"ComfyUI không xuất video cho prompt {prompt_id}")

        first_out = outputs[0]
        video_bytes = self.manager.get_view_bytes(
            filename=first_out["filename"],
            subfolder=first_out.get("subfolder", ""),
            folder_type=first_out.get("type", "output"),
        )

        elapsed = time.time() - start_t
        logger.info(f"✅ Hoàn thành sinh video I2V trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
        return video_bytes, elapsed

    def generate_stream(
        self,
        prompt: str,
        num_frames: int = 17,
        width: int = 832,
        height: int = 480,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình sinh video cho SSE kèm heartbeat định kỳ chống timeout."""
        self.load_to_gpu()
        p = self._resolve_params(num_frames, width, height, model_variant, kwargs)

        if "a14b" in p["variant"] or "14b" in p["variant"]:
            workflow = build_wan22_t2v_a14b_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                high_steps=kwargs.get("high_steps"),
                low_steps=kwargs.get("low_steps"),
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
            )
        else:
            workflow = build_wan22_ti2v_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
                start_image_name=None,
            )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": p["steps_val"], "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Video T2V stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 1200.0
        import json, urllib.request, urllib.error
        est_step = 1
        while time.time() - start_t < timeout:
            time.sleep(2.0)
            elapsed = time.time() - start_t
            if est_step < max(1, p["steps_val"] - 1):
                est_step += 1
            yield {
                "type": "progress",
                "step": est_step,
                "total_steps": p["steps_val"],
                "progress": min(95, int((est_step / max(1, p["steps_val"])) * 100)),
                "status": "executing",
                "stage": f"Denoising video frames ({p['width_val']}x{p['height_val']}, {p['frames_val']}f)...",
                "elapsed": round(elapsed, 1),
            }

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MMO-Factory"})
                with urllib.request.urlopen(req, timeout=10) as res:
                    history = json.loads(res.read().decode("utf-8"))

                if prompt_id not in history:
                    continue

                data = history[prompt_id]
                status = data.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {msgs}")

                outputs = data.get("outputs", {})
                video_results = []
                image_results = []
                for node_id, node_out in outputs.items():
                    if "gifs" in node_out:
                        video_results.extend(node_out["gifs"])
                    if "videos" in node_out:
                        video_results.extend(node_out["videos"])
                    if "images" in node_out:
                        image_results.extend(node_out["images"])
                results = video_results + image_results

                if results:
                    first_out = results[0]
                    video_bytes = self.manager.get_view_bytes(
                        filename=first_out["filename"],
                        subfolder=first_out.get("subfolder", ""),
                        folder_type=first_out.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    logger.info(f"✅ Hoàn thành sinh video T2V stream trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
                    yield {"type": "progress", "step": p["steps_val"], "total_steps": p["steps_val"], "progress": 100, "status": "completed"}
                    yield {"type": "complete", "video_bytes": video_bytes, "elapsed": elapsed}
                    return
            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")

    def generate_i2v_stream(
        self,
        prompt: str,
        image: Any,
        num_frames: int = 17,
        width: int = 832,
        height: int = 480,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình sinh video I2V cho SSE kèm heartbeat định kỳ."""
        self.load_to_gpu()

        input_filename = self.manager.save_input_image(image)
        p = self._resolve_params(num_frames, width, height, model_variant, kwargs)

        if "a14b" in p["variant"] or "14b" in p["variant"]:
            workflow = build_wan22_i2v_a14b_workflow(
                start_image_name=input_filename,
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                high_steps=kwargs.get("high_steps"),
                low_steps=kwargs.get("low_steps"),
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
            )
        else:
            workflow = build_wan22_ti2v_workflow(
                prompt=prompt,
                negative_prompt=p["negative_prompt"],
                width=p["width_val"],
                height=p["height_val"],
                num_frames=p["frames_val"],
                steps=p["steps_val"],
                cfg=p["cfg_val"],
                shift=p["shift_val"],
                sampler_name=p["sampler"],
                scheduler=p["scheduler"],
                seed=seed,
                fps=p["fps_val"],
                start_image_name=input_filename,
            )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": p["steps_val"], "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Video I2V stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 1200.0
        import json, urllib.request, urllib.error
        est_step = 1
        while time.time() - start_t < timeout:
            time.sleep(2.0)
            elapsed = time.time() - start_t
            if est_step < max(1, p["steps_val"] - 1):
                est_step += 1
            yield {
                "type": "progress",
                "step": est_step,
                "total_steps": p["steps_val"],
                "progress": min(95, int((est_step / max(1, p["steps_val"])) * 100)),
                "status": "executing",
                "stage": f"Denoising I2V frames ({p['width_val']}x{p['height_val']}, {p['frames_val']}f)...",
                "elapsed": round(elapsed, 1),
            }

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MMO-Factory"})
                with urllib.request.urlopen(req, timeout=10) as res:
                    history = json.loads(res.read().decode("utf-8"))

                if prompt_id not in history:
                    continue

                data = history[prompt_id]
                status = data.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {msgs}")

                outputs = data.get("outputs", {})
                video_results = []
                image_results = []
                for node_id, node_out in outputs.items():
                    if "gifs" in node_out:
                        video_results.extend(node_out["gifs"])
                    if "videos" in node_out:
                        video_results.extend(node_out["videos"])
                    if "images" in node_out:
                        image_results.extend(node_out["images"])
                results = video_results + image_results

                if results:
                    first_out = results[0]
                    video_bytes = self.manager.get_view_bytes(
                        filename=first_out["filename"],
                        subfolder=first_out.get("subfolder", ""),
                        folder_type=first_out.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    logger.info(f"✅ Hoàn thành sinh video I2V stream trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
                    yield {"type": "progress", "step": p["steps_val"], "total_steps": p["steps_val"], "progress": 100, "status": "completed"}
                    yield {"type": "complete", "video_bytes": video_bytes, "elapsed": elapsed}
                    return
            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")


def get_wan_engine(config: Optional[Dict[str, Any]] = None) -> ComfyUIVideoEngine:
    return ComfyUIVideoEngine(config)


def get_wan22_engine(config: Optional[Dict[str, Any]] = None) -> ComfyUIVideoEngine:
    return ComfyUIVideoEngine(config)
