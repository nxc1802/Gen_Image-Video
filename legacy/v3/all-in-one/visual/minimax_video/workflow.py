"""
🎬 ComfyUI Workflow Generator for MiniMax H3 (T2VA, FL2VA, Ref2VA & Turbo LoRA)
Tích hợp native Stereo Audio, Multi-Reference Conditioning (<Picture i>) & Dynamic VRAM Offloading.
"""

import random
from typing import Any, Dict, List, Optional


def _build_image_batch(frame_names: List[str], prefix: str, nodes: Dict[str, Any]) -> Optional[List[Any]]:
    """Tạo chuỗi node LoadImage + ImageBatch để tạo ra một tensor IMAGE đa khung hình [B, H, W, 3]."""
    if not frame_names:
        return None
    if len(frame_names) == 1:
        node_id = f"{prefix}_0"
        nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": frame_names[0]},
        }
        return [node_id, 0]

    load_ids = []
    for idx, fname in enumerate(frame_names):
        node_id = f"{prefix}_{idx}"
        nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": fname},
        }
        load_ids.append(node_id)

    prev_ref = [load_ids[0], 0]
    for idx in range(1, len(load_ids)):
        batch_id = f"{prefix}_batch_{idx}"
        nodes[batch_id] = {
            "class_type": "ImageBatch",
            "inputs": {
                "image1": prev_ref,
                "image2": [load_ids[idx], 0],
            },
        }
        prev_ref = [batch_id, 0]
    return prev_ref


def build_minimax_h3_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 448,
    num_frames: int = 33,
    steps: int = 14,
    cfg: float = 1.0,
    shift: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    seed: Optional[int] = None,
    fps: int = 24,
    enable_audio: bool = False,
    start_image_name: Optional[str] = None,
    end_image_name: Optional[str] = None,
    reference_image_name: Optional[str] = None,
    reference_image_names: Optional[List[str]] = None,
    guide_frame_names: Optional[List[str]] = None,
    ref_image_size: str = "match",
    mode: str = "auto",
    use_turbo: bool = False,
    lora_name: Optional[str] = None,
    lora_strength: float = 1.0,
    unet_name: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors",
    denoise: float = 1.0,
) -> Dict[str, Any]:
    """
    Sinh workflow MiniMax H3 chuẩn cho ComfyUI:
    - Hỗ trợ 2 nhánh DiT Backbone:
      1. FL2VA (MiniMaxH3ImageToVideo): T2V, T2VA, I2V, I2VA, FL2V, FL2VA, Video Chaining
      2. Ref2VA (MiniMaxH3ReferenceToVideo): Multi-Reference (<Picture 1>, <Picture 2>, ...) + Video Reference (<Video 1>) + Keyframe Guides (MiniMaxH3AddGuide)
    - Hỗ trợ Turbo LoRA (LoraLoaderModelOnly): Giảm số bước lấy mẫu xuống 4-8 steps.
    - Hỗ trợ Multi-Frame Guide Clip (ImageBatch 5 frames): Bảo toàn 100% vector vận tốc thời gian v⃗(t).
    """
    current_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)
    steps_val = max(1, int(steps or (6 if use_turbo else 14)))

    # MiniMax H3 VAE downscale 16x * DiT patch size 2x = bắt buộc width và height phải chia hết cho 32
    width = max(32, round(int(width or 768) / 32) * 32)
    height = max(32, round(int(height or 448) / 32) * 32)

    unet_loader = "UnetLoaderGGUF" if unet_name.endswith(".gguf") else "UNETLoader"
    clip_loader = "CLIPLoaderGGUF" if clip_name.endswith(".gguf") else "CLIPLoader"

    unet_inputs: Dict[str, Any] = {"unet_name": unet_name}
    if unet_loader == "UNETLoader":
        unet_inputs["weight_dtype"] = "default"

    clip_inputs: Dict[str, Any] = {"clip_name": clip_name, "type": "minimax"}
    if clip_loader == "CLIPLoader":
        clip_inputs["device"] = "default"

    combine_inputs: Dict[str, Any] = {
        "images": ["11", 0],
        "frame_rate": fps,
        "loop_count": 0,
        "filename_prefix": "MiniMax_H3_Output",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": True,
        "trim_to_audio": False,
        "pingpong": False,
        "save_output": True,
    }

    nodes: Dict[str, Any] = {
        # 1. DiT Backbone Loader (GGUF or SafeTensors)
        "1": {
            "class_type": unet_loader,
            "inputs": unet_inputs,
        },
        # 2. Text Encoder Loader (Shared Qwen3-VL-32B)
        "2": {
            "class_type": clip_loader,
            "inputs": clip_inputs,
        },
        # 3. Video VAE Loader
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
        },
    }

    # 1b. Optional Turbo LoRA Loader
    active_model_ref = ["1", 0]
    if use_turbo and lora_name:
        nodes["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name,
                "strength_model": float(lora_strength),
            },
        }
        active_model_ref = ["lora", 0]

    # 4. Audio VAE & Decode (nếu enable_audio=True)
    if enable_audio:
        nodes["4"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
        }
        nodes["13"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["10", 0], "vae": ["4", 0]},
        }
        combine_inputs["audio"] = ["13", 0]

    # Tập hợp danh sách Reference Images (cho Ref2V / Ref2VA)
    ref_list: List[str] = []
    if reference_image_names:
        for r_name in reference_image_names:
            if r_name and r_name not in ref_list:
                ref_list.append(r_name)
    if reference_image_name and reference_image_name not in ref_list:
        ref_list.append(reference_image_name)

    mode_norm = (mode or "auto").strip().lower()
    has_ref_unet = "ref2v" in unet_name.lower()
    use_ref2va_node = (
        mode_norm in ("ref2va", "ref2v")
        or (mode_norm == "auto" and bool(ref_list) and has_ref_unet)
    )

    # Nếu bật chế độ Ref2VA nhưng user truyền ảnh qua start_image_name mà chưa truyền ref_list
    if use_ref2va_node and not ref_list and start_image_name and not guide_frame_names:
        ref_list.append(start_image_name)
        start_image_name = None

    # Tạo batch ảnh đa khung hình nếu có guide_frame_names
    guide_batch_ref = _build_image_batch(guide_frame_names or [], "guide_clip", nodes)

    positive_cond_ref = ["9", 0]
    latent_ref = ["9", 1]

    if use_ref2va_node and (ref_list or guide_batch_ref):
        # ----------------------------------------------------------------------
        # NHÁNH A: Ref2VA Multi-Reference Conditioning (MiniMaxH3ReferenceToVideo)
        # ----------------------------------------------------------------------
        effective_prompt = prompt
        if "<Picture" not in effective_prompt and ref_list:
            if len(ref_list) == 1:
                effective_prompt = f"Featuring the exact subject and visual style of <Picture 1>: {prompt}"
            else:
                pic_tags = ", ".join(f"<Picture {i + 1}>" for i in range(len(ref_list)))
                effective_prompt = f"Combining the exact character, outfit, and scene elements from {pic_tags}: {prompt}"

        ref2v_inputs: Dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": effective_prompt,
            "width": width,
            "height": height,
            "length": num_frames,
            "ref_image_size": ref_image_size if ref_image_size in ("match", "max") else "match",
        }
        if enable_audio and "4" in nodes:
            ref2v_inputs["audio_vae"] = ["4", 0]

        for idx, ref_fname in enumerate(ref_list[:9]):
            node_id = f"ref_img_{idx}"
            nodes[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": ref_fname},
            }
            ref2v_inputs[f"ref_images.ref_image_{idx}"] = [node_id, 0]

        if guide_batch_ref is not None:
            # Truyền chuỗi đa khung hình vào ref_videos.ref_video_0 như một Video Reference <Video 1>
            ref2v_inputs["ref_videos.ref_video_0"] = guide_batch_ref

        nodes["9"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": ref2v_inputs,
        }

        # Neo cứng chuỗi đa khung hình (hoặc frame đơn lẻ) vào frame_idx=0 bằng MiniMaxH3AddGuide
        target_guide_img = guide_batch_ref
        if target_guide_img is None and start_image_name:
            nodes["start_img"] = {
                "class_type": "LoadImage",
                "inputs": {"image": start_image_name},
            }
            target_guide_img = ["start_img", 0]

        if target_guide_img is not None:
            nodes["guide_start"] = {
                "class_type": "MiniMaxH3AddGuide",
                "inputs": {
                    "positive": positive_cond_ref,
                    "vae": ["3", 0],
                    "latent": latent_ref,
                    "image": target_guide_img,
                    "frame_idx": 0,
                },
            }
            positive_cond_ref = ["guide_start", 0]

        if end_image_name:
            nodes["end_img"] = {
                "class_type": "LoadImage",
                "inputs": {"image": end_image_name},
            }
            nodes["guide_end"] = {
                "class_type": "MiniMaxH3AddGuide",
                "inputs": {
                    "positive": positive_cond_ref,
                    "vae": ["3", 0],
                    "latent": latent_ref,
                    "image": ["end_img", 0],
                    "frame_idx": -1,
                },
            }
            positive_cond_ref = ["guide_end", 0]

        if negative_prompt and float(cfg) > 1.0:
            nodes["6"] = {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "length": num_frames,
                    "ref_image_size": "match",
                },
            }
            negative_cond_ref = ["6", 0]
        else:
            negative_cond_ref = positive_cond_ref

    else:
        # ----------------------------------------------------------------------
        # NHÁNH B: FL2VA Keyframe Conditioning (MiniMaxH3ImageToVideo)
        # ----------------------------------------------------------------------
        minimax_inputs: Dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": num_frames,
        }

        effective_start = start_image_name or (ref_list[0] if ref_list else None)
        if effective_start:
            nodes["start_img"] = {
                "class_type": "LoadImage",
                "inputs": {"image": effective_start},
            }
            minimax_inputs["first_frame"] = ["start_img", 0]

        if end_image_name:
            nodes["end_img"] = {
                "class_type": "LoadImage",
                "inputs": {"image": end_image_name},
            }
            minimax_inputs["last_frame"] = ["end_img", 0]

        nodes["9"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": minimax_inputs,
        }

        # Nếu có guide_batch_ref (chuỗi 5 frame), ghép thêm MiniMaxH3AddGuide để neo tensor không-thời gian 3D
        if guide_batch_ref is not None:
            nodes["guide_start"] = {
                "class_type": "MiniMaxH3AddGuide",
                "inputs": {
                    "positive": ["9", 0],
                    "vae": ["3", 0],
                    "latent": ["9", 1],
                    "image": guide_batch_ref,
                    "frame_idx": 0,
                },
            }
            positive_cond_ref = ["guide_start", 0]
        else:
            positive_cond_ref = ["9", 0]

        if negative_prompt and float(cfg) > 1.0:
            nodes["6"] = {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "length": num_frames,
                },
            }
            negative_cond_ref = ["6", 0]
        else:
            negative_cond_ref = positive_cond_ref

    # 10. Denoising KSampler (Native MiniMax Flow Matching Sampling)
    nodes["10"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": active_model_ref,
            "positive": positive_cond_ref,
            "negative": negative_cond_ref,
            "latent_image": latent_ref,
            "seed": current_seed,
            "steps": steps_val,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    }

    # 11. Video Latent Decode
    nodes["11"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
    }

    # 12. Mux video frames thành chuẩn MP4 H.264
    nodes["12"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": combine_inputs,
    }

    return nodes

