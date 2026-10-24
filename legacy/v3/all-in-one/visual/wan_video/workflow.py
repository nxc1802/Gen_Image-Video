"""
🎬 ComfyUI Workflow Generator for Wan2.2 Video (TI2V-5B, T2V-A14B, I2V-A14B)
"""

import random
from typing import Any, Dict, Optional


def build_wan22_ti2v_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 17,
    steps: int = 20,
    cfg: float = 5.0,
    shift: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    seed: Optional[int] = None,
    fps: int = 24,
    start_image_name: Optional[str] = None,
    unet_name: str = "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    clip_name: str = "umt5-xxl-encoder-Q4_K_M.gguf",
    vae_name: str = "Wan2.2_VAE.safetensors",
    denoise: float = 1.0,
) -> Dict[str, Any]:
    """Sinh workflow Wan2.2 TI2V-5B chuẩn từ wan2.2_TI2V.py."""
    current_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)

    nodes = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_name},
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": clip_name, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt or "", "clip": ["2", 0]},
        },
        "8": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["1", 0], "shift": shift},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Wan2.2_TI2V_5B",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }

    if start_image_name:
        nodes["4"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_name},
        }
        nodes["7"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "start_image": ["4", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }
    else:
        # Nếu không có ảnh, dùng Wan22ImageToVideoLatent chuẩn từ wan2.2_TI2V.py
        nodes["7"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }

    return nodes


def build_wan22_t2v_a14b_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 17,
    steps: int = 20,
    high_steps: Optional[int] = None,
    low_steps: Optional[int] = None,
    cfg: float = 3.5,
    shift: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    seed: Optional[int] = None,
    fps: int = 24,
    high_unet: str = "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    low_unet: str = "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
    clip_name: str = "umt5-xxl-encoder-Q4_K_M.gguf",
    vae_name: str = "Wan2.1_VAE.safetensors",
) -> Dict[str, Any]:
    """Sinh workflow Wan2.2 T2V-A14B Two-Stage MoE chuẩn từ wan2.2_T2V.py."""
    current_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)
    steps = max(2, int(steps or 4))
    split_step = high_steps if high_steps is not None else max(1, steps // 2)
    final_step = low_steps if low_steps is not None else steps

    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": high_unet}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": low_unet}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": shift}},
        "5": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip_name, "type": "wan"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative_prompt or ""}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "9": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        },
        # High Noise Pass (0 -> split_step)
        "10": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["9", 0],
                "add_noise": "enable",
                "noise_seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": split_step,
                "return_with_leftover_noise": "enable",
            },
        },
        # Low Noise Pass (split_step -> final_step)
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["10", 0],
                "add_noise": "disable",
                "noise_seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": split_step,
                "end_at_step": final_step,
                "return_with_leftover_noise": "disable",
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["8", 0]},
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Wan2.2_T2V_A14B",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def build_wan22_i2v_a14b_workflow(
    start_image_name: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 17,
    steps: int = 20,
    high_steps: Optional[int] = None,
    low_steps: Optional[int] = None,
    cfg: float = 3.5,
    shift: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    seed: Optional[int] = None,
    fps: int = 24,
    end_image_name: Optional[str] = None,
    high_unet: str = "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
    low_unet: str = "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    clip_name: str = "umt5-xxl-encoder-Q4_K_M.gguf",
    vae_name: str = "Wan2.1_VAE.safetensors",
) -> Dict[str, Any]:
    """Sinh workflow Wan2.2 I2V-A14B Two-Stage MoE chuẩn từ wan2.2_I2V.py."""
    current_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)
    steps = max(2, int(steps or 4))
    split_step = high_steps if high_steps is not None else max(1, steps // 2)
    final_step = low_steps if low_steps is not None else steps

    nodes = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": high_unet}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": low_unet}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": shift}},
        "5": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip_name, "type": "wan"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative_prompt or ""}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "9": {"class_type": "LoadImage", "inputs": {"image": start_image_name}},
        # WanFirstLastFrameToVideo ràng buộc khung hình đầu/cuối vào latents
        "10": {
            "class_type": "WanFirstLastFrameToVideo",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["8", 0],
                "start_image": ["9", 0],
                "end_image": ["9", 0] if not end_image_name else ["9_end", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        },
        # High Noise Pass
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "add_noise": "enable",
                "noise_seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": split_step,
                "return_with_leftover_noise": "enable",
            },
        },
        # Low Noise Pass
        "12": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["4", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["11", 0],
                "add_noise": "disable",
                "noise_seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": split_step,
                "end_at_step": final_step,
                "return_with_leftover_noise": "disable",
            },
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["8", 0]},
        },
        "14": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["13", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Wan2.2_I2V_A14B",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }

    if end_image_name and end_image_name != start_image_name:
        nodes["9_end"] = {
            "class_type": "LoadImage",
            "inputs": {"image": end_image_name},
        }

    return nodes
