"""
🎨 ComfyUI Dynamic Workflow Generators
================================================================================
Trực tiếp chuyển hóa toàn bộ logic chuẩn từ thư mục 'kaggle/original/' thành các
hàm sinh Workflow JSON động:
1. Qwen-Image 2.1 (Text-to-Image & Inpainting)
2. Wan2.2 TI2V-5B (Text/Image-to-Video)
3. Wan2.2 T2V-A14B (Two-Stage High/Low Noise MoE Text-to-Video)
4. Wan2.2 I2V-A14B (Two-Stage High/Low Noise MoE Image-to-Video)

Tất cả các hyperparameter (width, height, steps, cfg, shift, sampler, scheduler,
seed, fps, prompts) đều được truyền động qua tham số hàm (từ API request)
chứ tuyệt đối KHÔNG hardcode.
"""

import random
from typing import Any, Dict, Optional


def build_qwen_image_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    seed: Optional[int] = None,
    unet_name: str = "qwen-image-2.1-Q4_K_M.gguf",
    clip_name: str = "qwen3vl_8b_fp8_scaled.safetensors",
    vae_name: str = "qwen_image_2.1_vae_bf16.safetensors",
    input_image_name: Optional[str] = None,
    mask_image_name: Optional[str] = None,
    reference_image_name: Optional[str] = None,
    denoise: float = 1.0,
) -> Dict[str, Any]:
    """Sinh workflow tạo ảnh Qwen-Image 2.1 GGUF chuẩn hỗ trợ T2I, I2I, Inpainting và Multimodal Edit."""
    current_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)

    nodes = {
        "461": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["475", 0],
                "filename_prefix": "Qwen_Image_2.1",
            },
        },
        "471": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["472", 0],
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "resolution": width,
            },
        },
        "472": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "473": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "475": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["476", 0],
                "vae": ["473", 0],
            },
        },
        "476": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["478", 0],
                "positive": ["471", 0],
                "negative": ["471", 1],
                "latent_image": ["474", 0],
                "seed": current_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0 if mask_image_name else denoise,
            },
        },
        "478": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": unet_name,
            },
        },
    }

    # Hỗ trợ Multimodal Instruction Editing (truyền ảnh tham chiếu vào image_1 của TextEncodeQwenImage21)
    ref_img = reference_image_name or (input_image_name if denoise == 1.0 and not mask_image_name else None)
    if ref_img:
        nodes["ref_img"] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_img},
        }
        nodes["471"]["inputs"]["image_1"] = ["ref_img", 0]

    if input_image_name:
        nodes["474_img"] = {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        }
        if mask_image_name:
            # Masked Inpainting / Outpainting (Sử dụng ImageToMask trích xuất kênh đỏ và VAEEncodeForInpaint chống viền seam)
            nodes["474_mask"] = {
                "class_type": "LoadImage",
                "inputs": {"image": mask_image_name},
            }
            nodes["474_mask_conv"] = {
                "class_type": "ImageToMask",
                "inputs": {
                    "image": ["474_mask", 0],
                    "channel": "red",
                },
            }
            nodes["474"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["474_img", 0],
                    "vae": ["473", 0],
                    "mask": ["474_mask_conv", 0],
                    "grow_mask_by": 6,
                },
            }
        else:
            # Standard Image-to-Image (I2I)
            nodes["474"] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["474_img", 0],
                    "vae": ["473", 0],
                },
            }
    else:
        # Text-to-Image tiêu chuẩn
        nodes["474"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }

    return nodes


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


def build_minimax_h3_workflow(*args, **kwargs):
    from visual.minimax_video.workflow import build_minimax_h3_workflow as _b
    return _b(*args, **kwargs)

