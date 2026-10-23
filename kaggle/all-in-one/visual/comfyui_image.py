"""
🖼️ ComfyUI-Powered Image Generation & Inpainting Engine (Qwen-Image-2.1 GGUF)
================================================================================
Kế thừa BaseImageEngine, thay thế hoàn toàn cơ chế Diffusers thủ công cũ:
1. Giao tiếp qua ComfyUI REST API trên GPU chuyên dụng (cuda:1).
2. Tận dụng 100% Dynamic VRAM & CPU Offload của ComfyUI (Zero OOM).
3. Động hóa toàn bộ tham số (prompt, negative_prompt, steps, cfg, sampler, scheduler, seed).
4. Hỗ trợ cả Text-to-Image và Masked Inpainting / Image-to-Image.
"""

import base64
import logging
import time
from typing import Any, Dict, Optional, Tuple

from config import MODELS_CONFIG
from core.base_engine import BaseImageEngine
from core.comfyui_manager import get_comfyui_manager
from core.comfyui_workflows import build_qwen_image_workflow

logger = logging.getLogger("ComfyUIImageEngine")


class ComfyUIImageEngine(BaseImageEngine):
    """Engine sinh ảnh & inpaint Qwen-Image 2.1 vận hành trên ComfyUI backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(cfg)
        self.model_id = cfg.get("id", "abenzerps/Qwen-Image-2.1-GGUF")
        self.device = cfg.get("device", "cuda:1")
        self.config = cfg
        self.manager = get_comfyui_manager()
        self.default_steps = cfg.get("steps", 20)
        self.default_guidance = cfg.get("guidance", 1.0)
        self.default_sampler = cfg.get("sampler_name", "euler")
        self.default_scheduler = cfg.get("scheduler", "simple")

        logger.info(f"🎨 ComfyUIImageEngine sẵn sàng: {self.model_id} trên {self.device}")

    def load_to_gpu(self) -> None:
        """Đảm bảo server ComfyUI đã khởi động và models đã được nạp."""
        if not self.manager.is_running():
            logger.info("⚡ Đang khởi động ComfyUI cho Image Engine...")
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
        """ComfyUI tự động quản lý Dynamic VRAM nên không cần tắt server, chỉ đánh dấu trạng thái."""
        self._is_loaded = False

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, float]:
        """Sinh ảnh từ prompt văn bản qua ComfyUI."""
        self.load_to_gpu()
        rt_cfg = MODELS_CONFIG.get("image", {})

        # Phân tích kích thước
        width = kwargs.get("width") or rt_cfg.get("width")
        height = kwargs.get("height") or rt_cfg.get("height")
        if not width or not height:
            try:
                parts = (size or rt_cfg.get("size", "1024x1024")).lower().split("x")
                width = int(parts[0])
                height = int(parts[1])
            except Exception:
                width, height = 1024, 1024

        negative_prompt = kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", "")
        steps_val = steps or rt_cfg.get("steps", self.default_steps)
        cfg_val = guidance if guidance is not None else rt_cfg.get("guidance", self.default_guidance)
        sampler = kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler)
        scheduler = kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler)

        # Multimodal Instruction Edit / Reference image support
        ref_img = kwargs.get("reference_image") or kwargs.get("instruction_image")
        ref_filename = self.manager.save_input_image(ref_img) if ref_img else None

        # Xây dựng workflow động
        workflow = build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            cfg=cfg_val,
            sampler_name=sampler,
            scheduler=scheduler,
            seed=seed,
            input_image_name=None,
            reference_image_name=ref_filename,
        )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Image prompt {prompt_id} tới ComfyUI...")

        outputs = self.manager.wait_for_completion(prompt_id)
        if not outputs:
            raise RuntimeError(f"ComfyUI không trả về ảnh nào cho prompt {prompt_id}")

        first_img = outputs[0]
        img_bytes = self.manager.get_view_bytes(
            filename=first_img["filename"],
            subfolder=first_img.get("subfolder", ""),
            folder_type=first_img.get("type", "output"),
        )

        elapsed = time.time() - start_t
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"✅ Hoàn thành sinh ảnh trong {elapsed:.2f}s (size={width}x{height}, steps={steps_val})")
        return b64_str, elapsed

    def inpaint(
        self,
        prompt: str,
        image: Any,
        mask_image: Any = None,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, float]:
        """Inpainting hoặc Image-to-Image qua ComfyUI."""
        self.load_to_gpu()
        rt_cfg = MODELS_CONFIG.get("image", {})

        # Lưu ảnh đầu vào và mask vào ComfyUI input
        input_filename = self.manager.save_input_image(image) if image else None
        mask_filename = self.manager.save_input_image(mask_image) if mask_image else None
        ref_img = kwargs.get("reference_image") or kwargs.get("instruction_image")
        ref_filename = self.manager.save_input_image(ref_img) if ref_img else None

        width = kwargs.get("width") or rt_cfg.get("width")
        height = kwargs.get("height") or rt_cfg.get("height")
        if not width or not height:
            try:
                parts = (size or rt_cfg.get("size", "1024x1024")).lower().split("x")
                width = int(parts[0])
                height = int(parts[1])
            except Exception:
                width, height = 1024, 1024

        negative_prompt = kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", "")
        steps_val = steps or rt_cfg.get("steps", self.default_steps)
        cfg_val = guidance if guidance is not None else rt_cfg.get("guidance", self.default_guidance)
        sampler = kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler)
        scheduler = kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler)
        denoise_val = (
            kwargs.get("strength")
            if kwargs.get("strength") is not None
            else (kwargs.get("denoise") if kwargs.get("denoise") is not None else rt_cfg.get("denoise", 0.8))
        )

        workflow = build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            cfg=cfg_val,
            sampler_name=sampler,
            scheduler=scheduler,
            seed=seed,
            input_image_name=input_filename,
            mask_image_name=mask_filename,
            reference_image_name=ref_filename,
            denoise=denoise_val,
        )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        outputs = self.manager.wait_for_completion(prompt_id)

        first_img = outputs[0]
        img_bytes = self.manager.get_view_bytes(
            filename=first_img["filename"],
            subfolder=first_img.get("subfolder", ""),
            folder_type=first_img.get("type", "output"),
        )

        elapsed = time.time() - start_t
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return b64_str, elapsed

    def generate_stream(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình cho SSE kèm heartbeat định kỳ chống timeout."""
        self.load_to_gpu()
        rt_cfg = MODELS_CONFIG.get("image", {})

        width = kwargs.get("width") or rt_cfg.get("width")
        height = kwargs.get("height") or rt_cfg.get("height")
        if not width or not height:
            try:
                parts = (size or rt_cfg.get("size", "1024x1024")).lower().split("x")
                width = int(parts[0])
                height = int(parts[1])
            except Exception:
                width, height = 1024, 1024

        negative_prompt = kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", "")
        steps_val = steps or rt_cfg.get("steps", self.default_steps)
        cfg_val = guidance if guidance is not None else rt_cfg.get("guidance", self.default_guidance)
        sampler = kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler)
        scheduler = kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler)

        ref_img = kwargs.get("reference_image") or kwargs.get("instruction_image")
        ref_filename = self.manager.save_input_image(ref_img) if ref_img else None

        workflow = build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            cfg=cfg_val,
            sampler_name=sampler,
            scheduler=scheduler,
            seed=seed,
            input_image_name=None,
            reference_image_name=ref_filename,
        )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": steps_val, "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Image stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 600.0
        import json, urllib.request, urllib.error
        est_step = 1
        while time.time() - start_t < timeout:
            time.sleep(1.5)
            elapsed = time.time() - start_t
            if est_step < max(1, steps_val - 1):
                est_step += 1
            yield {
                "type": "progress",
                "step": est_step,
                "total_steps": steps_val,
                "progress": min(95, int((est_step / max(1, steps_val)) * 100)),
                "status": "executing",
                "stage": f"Denoising latents ({width}x{height})...",
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
                results = []
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        results.extend(node_out["images"])

                if results:
                    first_img = results[0]
                    img_bytes = self.manager.get_view_bytes(
                        filename=first_img["filename"],
                        subfolder=first_img.get("subfolder", ""),
                        folder_type=first_img.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    logger.info(f"✅ Hoàn thành sinh ảnh stream trong {elapsed:.2f}s (size={width}x{height})")
                    yield {"type": "progress", "step": steps_val, "total_steps": steps_val, "progress": 100, "status": "completed"}
                    yield {"type": "complete", "b64_json": b64_str, "elapsed": elapsed}
                    return
            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")

    def inpaint_stream(
        self,
        prompt: str,
        image: Any,
        mask_image: Any = None,
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình cho Image Edit / Inpainting qua ComfyUI."""
        self.load_to_gpu()
        rt_cfg = MODELS_CONFIG.get("image", {})

        # Lưu ảnh đầu vào và mask vào ComfyUI input
        input_filename = self.manager.save_input_image(image) if image else None
        mask_filename = self.manager.save_input_image(mask_image) if mask_image else None
        ref_img = kwargs.get("reference_image") or kwargs.get("instruction_image")
        ref_filename = self.manager.save_input_image(ref_img) if ref_img else None

        width = kwargs.get("width") or rt_cfg.get("width")
        height = kwargs.get("height") or rt_cfg.get("height")
        if not width or not height:
            try:
                parts = (size or rt_cfg.get("size", "1024x1024")).lower().split("x")
                width = int(parts[0])
                height = int(parts[1])
            except Exception:
                width, height = 1024, 1024

        negative_prompt = kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", "")
        steps_val = steps or rt_cfg.get("steps", self.default_steps)
        cfg_val = guidance if guidance is not None else rt_cfg.get("guidance", self.default_guidance)
        sampler = kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler)
        scheduler = kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler)
        denoise_val = (
            kwargs.get("strength")
            if kwargs.get("strength") is not None
            else (kwargs.get("denoise") if kwargs.get("denoise") is not None else rt_cfg.get("denoise", 0.8))
        )

        workflow = build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            cfg=cfg_val,
            sampler_name=sampler,
            scheduler=scheduler,
            seed=seed,
            input_image_name=input_filename,
            mask_image_name=mask_filename,
            reference_image_name=ref_filename,
            denoise=denoise_val,
        )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": steps_val, "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi Image Edit/Inpaint stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 600.0
        import json, urllib.request, urllib.error
        est_step = 1
        while time.time() - start_t < timeout:
            time.sleep(1.5)
            elapsed = time.time() - start_t
            if est_step < max(1, steps_val - 1):
                est_step += 1
            yield {
                "type": "progress",
                "step": est_step,
                "total_steps": steps_val,
                "progress": min(95, int((est_step / max(1, steps_val)) * 100)),
                "status": "executing",
                "stage": f"Editing / Inpainting latents (denoise={denoise_val})...",
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
                results = []
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        results.extend(node_out["images"])

                if results:
                    first_img = results[0]
                    img_bytes = self.manager.get_view_bytes(
                        filename=first_img["filename"],
                        subfolder=first_img.get("subfolder", ""),
                        folder_type=first_img.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    logger.info(f"✅ Hoàn thành Image Edit/Inpaint stream trong {elapsed:.2f}s")
                    yield {"type": "progress", "step": steps_val, "total_steps": steps_val, "progress": 100, "status": "completed"}
                    yield {"type": "complete", "b64_json": b64_str, "elapsed": elapsed}
                    return
            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")


def get_qwen_image_engine(config: Optional[Dict[str, Any]] = None) -> ComfyUIImageEngine:
    return ComfyUIImageEngine(config)
