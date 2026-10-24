"""
🎨 ComfyUI Workflow Generator for Qwen-Image 2.1 (T2I & Inpainting)
"""

import random
from typing import Any, Dict, Optional


def build_qwen_image_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
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
