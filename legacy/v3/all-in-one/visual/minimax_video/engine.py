"""
🎬 ComfyUI-Powered MiniMax H3 Video & Audio Generation Engine
================================================================================
Kế thừa BaseVideoEngine, vận hành mô hình MiniMax H3 (T2VA, FL2VA, Ref2VA)
hoàn toàn trên 1 GPU Tesla T4 (cuda:1 hoặc cuda:0) kết hợp ComfyUI Dynamic Offloading.
"""

import base64
import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from config import MODELS_CONFIG
from core.base_engine import BaseVideoEngine
from core.comfyui_manager import get_comfyui_manager
from .workflow import build_minimax_h3_workflow

logger = logging.getLogger("ComfyUIMiniMaxEngine")


class ComfyUIMiniMaxEngine(BaseVideoEngine):
    """
    Engine sinh video & audio MiniMax H3 (FL2VA + Ref2VA + Turbo LoRA) trên ComfyUI backend.
    Tích hợp cơ chế Shared Encoder (Qwen3-VL 32B + Video/Audio VAE) và Dynamic UNet Hot-Swap
    có bảo vệ ngưỡng System RAM (RAM Guard) trên Kaggle Dual-T4.
    """

    # Theo dõi UNet đang nạp trên từng GPU để tự động giải phóng khi chuyển đổi FL2VA <-> Ref2VA
    _active_unet_per_gpu: Dict[int, str] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(cfg)
        self.model_id = cfg.get("id", "Comfy-Org/MiniMax-H3")
        from config import GPU_COUNT, resolve_model_gpu_device
        dev_str, gpu_idx = resolve_model_gpu_device("minimax")
        if "gpu_device" in cfg and cfg["gpu_device"] is not None:
            req_gpu = int(cfg["gpu_device"])
            self.gpu_device_id = min(req_gpu, max(0, GPU_COUNT - 1)) if GPU_COUNT > 0 else 0
            self.device = f"cuda:{self.gpu_device_id}" if GPU_COUNT > 0 else "cpu"
        else:
            self.device = dev_str
            self.gpu_device_id = gpu_idx

        self.config = cfg
        self.manager = get_comfyui_manager(cuda_device=self.gpu_device_id)

        self.width = cfg.get("width", 768)
        self.height = cfg.get("height", 448)
        self.num_frames = cfg.get("num_frames", 33)
        self.default_steps = cfg.get("steps", 14)
        self.default_turbo_steps = cfg.get("turbo_steps", 6)
        self.default_cfg = cfg.get("guidance", 1.0)
        self.default_shift = cfg.get("shift", 8.0)
        self.default_fps = cfg.get("fps", 24)
        self.default_sampler = cfg.get("sampler_name", "euler")
        self.default_scheduler = cfg.get("scheduler", "simple")
        self.default_enable_audio = cfg.get("enable_audio", False)

        logger.info(
            f"🎬 ComfyUIMiniMaxEngine sẵn sàng: {self.model_id} trên {self.device} "
            f"(GPU {self.gpu_device_id}, ComfyUI port {self.manager.port})"
        )

    def load_to_gpu(self) -> None:
        """Đảm bảo server ComfyUI đã khởi động."""
        if not self.manager.is_running():
            logger.info("⚡ Đang khởi động ComfyUI cho MiniMax H3 Engine...")
            ok = self.manager.start()
            if not ok:
                raise RuntimeError("ComfyUI server không khởi động thành công")
        self._is_loaded = True

    def load(self) -> Any:
        self.load_to_gpu()
        return self

    def infer(self, *args, **kwargs) -> Any:
        return self.generate(*args, **kwargs)

    def release_from_gpu(self) -> None:
        self._is_loaded = False

    def _model_exists(self, filename: str, dirs: List[Any]) -> bool:
        for d in dirs:
            if (d / filename).exists():
                return True
        return filename in self.manager.linked_models

    def _resolve_params(
        self,
        num_frames: Optional[int],
        width: Optional[int],
        height: Optional[int],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        rt_cfg = MODELS_CONFIG.get("minimax", MODELS_CONFIG.get("video", {}))
        audio_val = kwargs.get("enable_audio")
        if audio_val is None:
            audio_val = kwargs.get("audio", rt_cfg.get("enable_audio", self.default_enable_audio))

        mode_hint = str(kwargs.get("mode") or kwargs.get("model_variant") or "").lower()
        turbo_flag = kwargs.get("turbo")
        if turbo_flag is None:
            turbo_flag = kwargs.get("use_turbo")
        if turbo_flag is None:
            turbo_flag = ("turbo" in mode_hint) or bool(kwargs.get("lora_name")) or bool(rt_cfg.get("use_turbo", False))
        use_turbo = bool(turbo_flag)

        raw_steps = kwargs.get("steps")
        if raw_steps is not None:
            # Nếu bật turbo mà caller vẫn để mặc định 14 steps -> tự động tối ưu xuống 6 steps
            if use_turbo and int(raw_steps) == self.default_steps:
                steps_val = self.default_turbo_steps
            else:
                steps_val = int(raw_steps)
        else:
            steps_val = self.default_turbo_steps if use_turbo else int(rt_cfg.get("steps", self.default_steps))

        return {
            "negative_prompt": kwargs.get("negative_prompt") or rt_cfg.get("negative_prompt", ""),
            "steps_val": max(1, steps_val),
            "cfg_val": kwargs.get("guidance") if kwargs.get("guidance") is not None else rt_cfg.get("guidance", self.default_cfg),
            "shift_val": kwargs.get("shift") if kwargs.get("shift") is not None else rt_cfg.get("shift", self.default_shift),
            "fps_val": kwargs.get("fps") or rt_cfg.get("fps", self.default_fps),
            "sampler": kwargs.get("sampler_name") or rt_cfg.get("sampler_name", self.default_sampler),
            "scheduler": kwargs.get("scheduler") or rt_cfg.get("scheduler", self.default_scheduler),
            "frames_val": num_frames or rt_cfg.get("num_frames", self.num_frames),
            "width_val": width or rt_cfg.get("width", self.width),
            "height_val": height or rt_cfg.get("height", self.height),
            "audio_val": bool(audio_val),
            "use_turbo": use_turbo,
            "lora_strength": float(kwargs.get("lora_strength") if kwargs.get("lora_strength") is not None else rt_cfg.get("lora_strength", 1.0)),
            "ref_image_size": str(kwargs.get("ref_image_size") or rt_cfg.get("ref_image_size", "match")),
            "mode": str(kwargs.get("mode") or "auto"),
        }

    def _resolve_model_names(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str, str, Optional[str]]:
        """
        Tự động phát hiện phiên bản trọng số MiniMax H3 (FL2VA vs Ref2VA) và Turbo LoRA.
        - Tự động chọn Ref2VA UNet khi có reference_images / reference_image hoặc mode='ref2va'.
        - Tự động chọn FL2VA UNet khi chạy T2V / I2V / FL2V / FL2VA.
        """
        kw = kwargs or {}
        rt_cfg = MODELS_CONFIG.get("minimax", {})

        mode_hint = f"{kw.get('mode', '')} {kw.get('model_variant', '')}".lower()
        has_ref_input = bool(kw.get("reference_images") or kw.get("reference_image"))
        wants_ref2va = ("ref2v" in mode_hint) or (has_ref_input and "fl2v" not in mode_hint)

        fl2va_unets = [
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "MiniMax-H3_FL2VA-INT8-ConvRot.safetensors",
            "MiniMax-H3-FL2VA-Q4_K_M.gguf",
            "minimax_h3_fl2va_pruned-Q4_K.gguf",
            "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            "minimax_h3_fl2va_pruned_w6a8.safetensors",
        ]
        ref2va_unets = [
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "MiniMax-H3_Ref2VA-INT8-ConvRot.safetensors",
            "minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors",
            "minimax_h3_ref2va_pruned-Q6_K.gguf",
            "minimax-h3-ref2va-U16G.gguf",
        ]

        unet_dirs = [self.manager.unet_dir, self.manager.diffusion_dir]
        explicit_unet = kw.get("unet_name") or rt_cfg.get("unet_name")
        if explicit_unet:
            unet_name = explicit_unet
        else:
            ordered_unets = (ref2va_unets + fl2va_unets) if wants_ref2va else (fl2va_unets + ref2va_unets)
            unet_name = ordered_unets[0]
            for c in ordered_unets:
                if self._model_exists(c, unet_dirs):
                    unet_name = c
                    break

        clip_name = kw.get("clip_name") or rt_cfg.get("clip_name") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        candidate_clips = [
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "qwen3_vl_32b_q4_k_m.gguf",
            "qwen3vl-32B-MiniMax-H3-Q2_K.gguf",
        ]
        if not (kw.get("clip_name") or rt_cfg.get("clip_name")):
            for c in candidate_clips:
                if self._model_exists(c, [self.manager.clip_dir, self.manager.text_encoders_dir]):
                    clip_name = c
                    break

        video_vae = kw.get("video_vae_name") or rt_cfg.get("video_vae_name") or "minimax_h3_video_vae_fp16.safetensors"
        candidate_v_vaes = [
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_video_vae_int8_convrot.safetensors",
            "MiniMax-H3-video_vae_fp16.safetensors",
        ]
        if not (kw.get("video_vae_name") or rt_cfg.get("video_vae_name")):
            for c in candidate_v_vaes:
                if self._model_exists(c, [self.manager.vae_dir]):
                    video_vae = c
                    break

        audio_vae = kw.get("audio_vae_name") or rt_cfg.get("audio_vae_name") or "minimax_h3_audio_vae_fp32.safetensors"
        candidate_a_vaes = [
            "minimax_h3_audio_vae_fp32.safetensors",
            "MiniMax-H3-audio_vae_fp32.safetensors",
        ]
        if not (kw.get("audio_vae_name") or rt_cfg.get("audio_vae_name")):
            for c in candidate_a_vaes:
                if self._model_exists(c, [self.manager.vae_dir]):
                    audio_vae = c
                    break

        lora_name: Optional[str] = kw.get("lora_name") or rt_cfg.get("lora_name")
        if not lora_name:
            candidate_loras = [
                "minimax_h3_turbo_4step_pruned_comfyui.safetensors",
                "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
                "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            ]
            for c in candidate_loras:
                if self._model_exists(c, [self.manager.loras_dir]):
                    lora_name = c
                    break

        logger.info(
            f"🔍 MiniMax resolved (wants_ref2va={wants_ref2va}): unet={unet_name}, "
            f"clip={clip_name}, video_vae={video_vae}, audio_vae={audio_vae}, lora={lora_name}"
        )
        return unet_name, clip_name, video_vae, audio_vae, lora_name

    def _ensure_unet_ready(self, target_unet: str) -> None:
        """
        RAM Guard & Dynamic UNet Hot-Swap trên GPU Kaggle (31.35 GB System RAM):
        Nếu chuyển đổi giữa FL2VA UNet và Ref2VA UNet trên cùng ComfyUI instance,
        hoặc khi System RAM khả dụng xuống thấp (< 3.2 GB), tự động gọi /free để xả UNet cũ
        khỏi VRAM và Host RAM trước khi nạp UNet mới.
        """
        prev_unet = ComfyUIMiniMaxEngine._active_unet_per_gpu.get(self.gpu_device_id)
        avail_ram_gb = 10.0
        try:
            import psutil
            avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass

        if prev_unet is not None and prev_unet != target_unet:
            logger.info(
                f"🔄 [Hot-Swap GPU {self.gpu_device_id}] Chuyển đổi DiT UNet: '{prev_unet}' -> '{target_unet}' "
                f"(RAM trống: {avail_ram_gb:.2f} GB). Đang giải phóng UNet cũ..."
            )
            self.manager.free_vram(unload_models=True, free_memory=True)
            gc.collect()
            time.sleep(0.5)
        elif prev_unet is not None and avail_ram_gb < 3.2:
            logger.warning(
                f"⚠️ [RAM Guard GPU {self.gpu_device_id}] System RAM trống thấp ({avail_ram_gb:.2f} GB < 3.2 GB). "
                f"Chủ động gọi ComfyUI /free để dọn cache..."
            )
            self.manager.free_vram(unload_models=False, free_memory=True)
            gc.collect()

        ComfyUIMiniMaxEngine._active_unet_per_gpu[self.gpu_device_id] = target_unet

    def _save_reference_images(self, kwargs: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Lưu cả reference_image đơn lẻ và danh sách reference_images (Multi-Reference) vào ComfyUI input."""
        ref_names: List[str] = []
        raw_list = kwargs.get("reference_images")
        if isinstance(raw_list, list):
            for idx, item in enumerate(raw_list):
                if item:
                    fname = self.manager.save_input_image(item)
                    ref_names.append(fname)

        single_ref = kwargs.get("reference_image")
        single_ref_name: Optional[str] = None
        if single_ref:
            single_ref_name = self.manager.save_input_image(single_ref)
            if single_ref_name not in ref_names:
                ref_names.insert(0, single_ref_name)
        elif ref_names:
            single_ref_name = ref_names[0]

        return single_ref_name, ref_names

    def _save_guide_frames(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        """Lưu chuỗi frame hướng dẫn (guide_frames) đa khung hình để neo vector vận tốc vào ComfyUI input."""
        raw_frames = kwargs.get("guide_frames")
        if not raw_frames or not isinstance(raw_frames, list):
            return None
        saved: List[str] = []
        for f in raw_frames:
            if f:
                fname = self.manager.save_input_image(f)
                saved.append(fname)
        return saved or None

    def generate(
        self,
        prompt: str,
        num_frames: int = 33,
        width: int = 768,
        height: int = 448,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bytes, float]:
        """Sinh video Text-to-Video (T2V / T2VA) hoặc Reference-to-Video (Ref2V / Ref2VA) qua MiniMax H3."""
        self.load_to_gpu()
        if model_variant:
            kwargs["model_variant"] = model_variant
        p = self._resolve_params(num_frames, width, height, kwargs)
        u_name, c_name, v_vae, a_vae, l_name = self._resolve_model_names(kwargs)
        self._ensure_unet_ready(u_name)

        ref_image_name, ref_image_names = self._save_reference_images(kwargs)
        guide_frame_names = self._save_guide_frames(kwargs)

        workflow = build_minimax_h3_workflow(
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
            enable_audio=p["audio_val"],
            start_image_name=None,
            reference_image_name=ref_image_name,
            reference_image_names=ref_image_names,
            guide_frame_names=guide_frame_names,
            ref_image_size=p["ref_image_size"],
            mode=p["mode"],
            use_turbo=p["use_turbo"],
            lora_name=l_name,
            lora_strength=p["lora_strength"],
            unet_name=u_name,
            clip_name=c_name,
            video_vae_name=v_vae,
            audio_vae_name=a_vae,
        )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(
            f"📤 Đã gửi MiniMax H3 prompt {prompt_id} tới ComfyUI "
            f"(unet={u_name}, audio={p['audio_val']}, turbo={p['use_turbo']}, steps={p['steps_val']}, refs={len(ref_image_names)}, guide_frames={len(guide_frame_names or [])})..."
        )

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
        logger.info(f"✅ Hoàn thành sinh video MiniMax H3 trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
        return video_bytes, elapsed

    def generate_i2v(
        self,
        prompt: str,
        image: Any,
        num_frames: int = 33,
        width: int = 768,
        height: int = 448,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bytes, float]:
        """Sinh video Image-to-Video (I2V, I2VA, FL2V, FL2VA, Ref2V, Ref2VA & Multi-Reference) qua MiniMax H3."""
        self.load_to_gpu()
        if model_variant:
            kwargs["model_variant"] = model_variant
        input_filename = self.manager.save_input_image(image) if image else None

        end_image_name = None
        last_frame = kwargs.get("last_frame") or kwargs.get("end_image")
        if last_frame:
            end_image_name = self.manager.save_input_image(last_frame)

        ref_image_name, ref_image_names = self._save_reference_images(kwargs)
        guide_frame_names = self._save_guide_frames(kwargs)

        p = self._resolve_params(num_frames, width, height, kwargs)
        u_name, c_name, v_vae, a_vae, l_name = self._resolve_model_names(kwargs)
        self._ensure_unet_ready(u_name)

        workflow = build_minimax_h3_workflow(
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
            enable_audio=p["audio_val"],
            start_image_name=input_filename,
            end_image_name=end_image_name,
            reference_image_name=ref_image_name,
            reference_image_names=ref_image_names,
            guide_frame_names=guide_frame_names,
            ref_image_size=p["ref_image_size"],
            mode=p["mode"],
            use_turbo=p["use_turbo"],
            lora_name=l_name,
            lora_strength=p["lora_strength"],
            unet_name=u_name,
            clip_name=c_name,
            video_vae_name=v_vae,
            audio_vae_name=a_vae,
        )

        start_t = time.time()
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(
            f"📤 Đã gửi MiniMax H3 I2V/Ref2V prompt {prompt_id} tới ComfyUI "
            f"(unet={u_name}, audio={p['audio_val']}, turbo={p['use_turbo']}, steps={p['steps_val']}, "
            f"start={'yes' if input_filename else 'no'}, end={'yes' if end_image_name else 'no'}, refs={len(ref_image_names)}, guide_frames={len(guide_frame_names or [])})..."
        )

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
        logger.info(f"✅ Hoàn thành sinh video MiniMax H3 I2V/Ref2V trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
        return video_bytes, elapsed

    def generate_stream(
        self,
        prompt: str,
        num_frames: int = 17,
        width: int = 416,
        height: int = 240,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình sinh video MiniMax H3 cho SSE kèm heartbeat chống timeout."""
        self.load_to_gpu()
        if model_variant:
            kwargs["model_variant"] = model_variant
        p = self._resolve_params(num_frames, width, height, kwargs)
        u_name, c_name, v_vae, a_vae, l_name = self._resolve_model_names(kwargs)
        self._ensure_unet_ready(u_name)
        ref_image_name, ref_image_names = self._save_reference_images(kwargs)
        guide_frame_names = self._save_guide_frames(kwargs)

        workflow = build_minimax_h3_workflow(
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
            enable_audio=p["audio_val"],
            start_image_name=None,
            reference_image_name=ref_image_name,
            reference_image_names=ref_image_names,
            guide_frame_names=guide_frame_names,
            ref_image_size=p["ref_image_size"],
            mode=p["mode"],
            use_turbo=p["use_turbo"],
            lora_name=l_name,
            lora_strength=p["lora_strength"],
            unet_name=u_name,
            clip_name=c_name,
            video_vae_name=v_vae,
            audio_vae_name=a_vae,
        )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": p["steps_val"], "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi MiniMax H3 stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 1800.0
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
                "stage": f"Denoising MiniMax H3 Video+Audio ({p['width_val']}x{p['height_val']}, {p['frames_val']}f)...",
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
                for node_id, node_out in outputs.items():
                    if "gifs" in node_out:
                        video_results.extend(node_out["gifs"])
                    if "videos" in node_out:
                        video_results.extend(node_out["videos"])

                if video_results:
                    first_out = video_results[0]
                    video_bytes = self.manager.get_view_bytes(
                        filename=first_out["filename"],
                        subfolder=first_out.get("subfolder", ""),
                        folder_type=first_out.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    logger.info(f"✅ Hoàn thành sinh video MiniMax H3 stream trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
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
        width: int = 416,
        height: int = 240,
        seed: Optional[int] = None,
        model_variant: Optional[str] = None,
        **kwargs,
    ):
        """Streaming tiến trình sinh video I2V / Ref2VA MiniMax H3 cho SSE kèm heartbeat."""
        self.load_to_gpu()
        if model_variant:
            kwargs["model_variant"] = model_variant
        input_filename = self.manager.save_input_image(image) if image else None

        end_image_name = None
        last_frame = kwargs.get("last_frame") or kwargs.get("end_image")
        if last_frame:
            end_image_name = self.manager.save_input_image(last_frame)

        ref_image_name, ref_image_names = self._save_reference_images(kwargs)

        p = self._resolve_params(num_frames, width, height, kwargs)
        u_name, c_name, v_vae, a_vae, l_name = self._resolve_model_names(kwargs)
        self._ensure_unet_ready(u_name)

        workflow = build_minimax_h3_workflow(
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
            enable_audio=p["audio_val"],
            start_image_name=input_filename,
            end_image_name=end_image_name,
            reference_image_name=ref_image_name,
            reference_image_names=ref_image_names,
            ref_image_size=p["ref_image_size"],
            mode=p["mode"],
            use_turbo=p["use_turbo"],
            lora_name=l_name,
            lora_strength=p["lora_strength"],
            unet_name=u_name,
            clip_name=c_name,
            video_vae_name=v_vae,
            audio_vae_name=a_vae,
        )

        start_t = time.time()
        yield {"type": "progress", "step": 0, "total_steps": p["steps_val"], "progress": 0, "status": "queueing"}
        prompt_id = self.manager.queue_prompt(workflow)
        logger.info(f"📤 Đã gửi MiniMax H3 I2V stream prompt {prompt_id} tới ComfyUI...")

        url = f"{self.manager.base_url}/history/{prompt_id}"
        timeout = 1800.0
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
                "stage": f"Denoising MiniMax H3 I2V ({p['width_val']}x{p['height_val']}, {p['frames_val']}f)...",
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
                for node_id, node_out in outputs.items():
                    if "gifs" in node_out:
                        video_results.extend(node_out["gifs"])
                    if "videos" in node_out:
                        video_results.extend(node_out["videos"])

                if video_results:
                    first_out = video_results[0]
                    video_bytes = self.manager.get_view_bytes(
                        filename=first_out["filename"],
                        subfolder=first_out.get("subfolder", ""),
                        folder_type=first_out.get("type", "output"),
                    )
                    elapsed = time.time() - start_t
                    logger.info(f"✅ Hoàn thành sinh video MiniMax H3 I2V stream trong {elapsed:.2f}s ({len(video_bytes) / 1024:.1f} KB)")
                    yield {"type": "progress", "step": p["steps_val"], "total_steps": p["steps_val"], "progress": 100, "status": "completed"}
                    yield {"type": "complete", "video_bytes": video_bytes, "elapsed": elapsed}
                    return
            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")


def get_minimax_engine(config: Optional[Dict[str, Any]] = None, gpu_device: Optional[int] = None) -> ComfyUIMiniMaxEngine:
    cfg = dict(config or {})
    if gpu_device is not None:
        cfg["gpu_device"] = gpu_device
        cfg["device"] = f"cuda:{gpu_device}"
    return ComfyUIMiniMaxEngine(cfg)
