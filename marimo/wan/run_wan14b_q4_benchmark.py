#!/usr/bin/env python3
"""
🚀 Wan2.1-T2V-14B 4-bit (NF4) Quantization Benchmark with 3 Test Prompts
-------------------------------------------------------------------------
Chạy 3 kịch bản Prompt trên Wan2.1-T2V-14B nén 4-bit NF4 (bitsandbytes):
- Độ phân giải: 1280x720 (720P HD)
- Số frame: 33 frames @ 16 fps (~2.06s)
- Số bước lấy mẫu (steps): 30 steps
- CFG Scale: 6.0
- Seed: 42
- Tối ưu: 4-bit NF4 DiT + Host RAM Swap (Text Encoder) + VAE Tiling/Slicing
"""

import os
import gc
import time
import json
from typing import Dict, Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import imageio

# 1. Patch diffusers bitsandbytes import check
import diffusers.utils.import_utils as iu
import bitsandbytes as bnb

iu._bitsandbytes_available = True
iu._bitsandbytes_version = bnb.__version__
iu.is_bitsandbytes_version.cache_clear()

import diffusers.quantizers.bitsandbytes.utils as bnb_utils
import diffusers.quantizers.bitsandbytes.bnb_quantizer as bnb_quantizer
bnb_utils.bnb = bnb
bnb_quantizer.bnb = bnb

from diffusers import BitsAndBytesConfig, WanTransformer3DModel, WanPipeline

TEST_PROMPTS = [
    {
        "id": "p1_cafe",
        "name": "Chân dung đời thường & Biểu cảm",
        "prompt": (
            "A close-up cinematic shot of a young woman in a soft knitted cream sweater sitting by a sunlit "
            "wooden cafe table, gently holding a steaming ceramic mug with both hands, taking a slow delicate sip "
            "and looking out the rainy window with a calm natural smile, soft warm morning lighting, "
            "raindrops on window glass, shallow depth of field, photorealistic 8k, fluid realistic subtle motion"
        ),
    },
    {
        "id": "p2_puppy",
        "name": "Động vật & Chi tiết sợi lông / Ánh sáng tự nhiên",
        "prompt": (
            "A cute golden retriever puppy sitting on vibrant green grass in a sunlit park, "
            "tilting its head curiously with wide eyes, soft wind rustling its golden fur, "
            "warm afternoon golden hour sunlight, macro cinematic 4k, fluid lifelike subtle motion"
        ),
    },
    {
        "id": "p3_dragon",
        "name": "Điện ảnh hoành tráng & Hiệu ứng ánh sáng phức tạp",
        "prompt": (
            "A magnificent mechanical golden dragon soaring gracefully through misty neon thunderstorm clouds "
            "above a cyberpunk metropolis at dusk, cinematic camera tracking, volumetric lightning flashes, "
            "4k ultra-detailed, photorealistic fluid motion, vivid reflections"
        ),
    },
]

NEGATIVE_PROMPT = ""
SEED = 42

def get_vram_gb(device=0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
        "reserved": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
        "peak": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
    }

def main():
    print(f"\n{'='*80}")
    print("🚀 KHỞI ĐỘNG BENCHMARK WAN2.1-14B 4-BIT (NF4 QUANTIZATION)")
    print("   Độ phân giải: 1280x720 | 33 frames @ 16 fps | 30 steps | CFG 6.0 | Seed 42")
    print(f"{'='*80}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_vram = get_vram_gb()
    print(f"[*] Base VRAM: {base_vram['allocated']} GB Alloc | {base_vram['reserved']} GB Res")

    # Cấu hình BitsAndBytes 4-bit NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    t_load_start = time.time()
    print("[*] Đang tải WanTransformer3DModel 14B với 4-bit NF4 quantization...")
    transformer_4bit = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    vram_after_transformer = get_vram_gb()
    print(f"[*] WanTransformer3DModel 4-bit tải xong! VRAM: {vram_after_transformer['allocated']} GB")

    print("[*] Đang lắp ráp WanPipeline...")
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        transformer=transformer_4bit,
        torch_dtype=torch.bfloat16,
    )
    t_load = time.time() - t_load_start
    print(f"[*] WanPipeline 14B 4-bit hoàn tất trong {t_load:.2f}s!")

    # Bật VAE Tiling và Slicing
    if hasattr(pipe, "vae"):
        pipe.vae.to("cuda")
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    # Text encoder đặt trên CPU (Host RAM Swap)
    pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    static_vram = get_vram_gb()
    print(f"[*] Static VRAM (4-bit DiT + VAE on GPU, Text Encoder on CPU):")
    print(f"    Allocated: {static_vram['allocated']} GB | Reserved: {static_vram['reserved']} GB")

    file_prefix = "wan21_14b_q4_opt"
    prompt_results = []

    for idx, p_info in enumerate(TEST_PROMPTS):
        p_id = p_info["id"]
        p_text = p_info["prompt"]
        p_title = p_info["name"]
        out_mp4 = f"{file_prefix}_{p_id}.mp4"

        print(f"\n--- [Prompt {idx+1}/3: {p_title}] ---")
        t_p_start = time.time()

        # 1. Text Encode với Fast GPU Swap
        torch.cuda.reset_peak_memory_stats()
        t_enc_start = time.time()
        pipe.text_encoder.to("cuda")
        with torch.inference_mode():
            p_embeds, neg_embeds = pipe.encode_prompt(
                prompt=p_text,
                negative_prompt=NEGATIVE_PROMPT or None,
                do_classifier_free_guidance=True,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
        t_enc = time.time() - t_enc_start

        # 2. Swap Text Encoder về CPU Host RAM
        t_swap_start = time.time()
        pipe.text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        t_swap = time.time() - t_swap_start

        # 3. Denoising DiT
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        t_denoise_start = time.time()
        with torch.inference_mode():
            pipe_out = pipe(
                prompt_embeds=p_embeds,
                negative_prompt_embeds=neg_embeds,
                width=1280,
                height=720,
                num_frames=33,
                num_inference_steps=30,
                guidance_scale=6.0,
                generator=generator,
            )
        t_denoise = time.time() - t_denoise_start
        denoise_vram = get_vram_gb()

        # 4. Xuất video MP4 (H.264 CRF 17 High Profile)
        t_export_start = time.time()
        frames_rgb = pipe_out.frames[0]
        writer = imageio.get_writer(
            out_mp4,
            fps=16,
            codec="libx264",
            ffmpeg_params=[
                "-crf", "17",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
            ],
        )
        for f in frames_rgb:
            f_np = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
            writer.append_data(f_np)
        writer.close()
        t_export = time.time() - t_export_start
        t_total = time.time() - t_p_start

        sz_kb = round(os.path.getsize(out_mp4) / 1024, 1)
        print(f"✅ Hoàn thành: {out_mp4} ({sz_kb} KB)")
        print(f"   Denoise: {t_denoise:.2f}s | Tổng: {t_total:.2f}s | Peak Alloc: {denoise_vram['peak']} GB | Reserved: {denoise_vram['reserved']} GB")

        prompt_results.append({
            "prompt_id": p_id,
            "title": p_title,
            "prompt": p_text,
            "file": out_mp4,
            "size_kb": sz_kb,
            "timing": {
                "text_encode_s": round(t_enc, 2),
                "swap_s": round(t_swap, 2),
                "denoise_s": round(t_denoise, 2),
                "export_s": round(t_export, 2),
                "total_s": round(t_total, 2),
            },
            "vram_gb": {
                "peak_allocated": denoise_vram["peak"],
                "peak_reserved": denoise_vram["reserved"],
            },
        })

        del p_embeds
        del neg_embeds
        del pipe_out
        gc.collect()
        torch.cuda.empty_cache()

    # Dọn dẹp
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "timestamp": time.time(),
        "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "model_name": "Wan2.1-T2V-14B (4-bit NF4 Quantized)",
        "quantization": {
            "method": "bitsandbytes",
            "type": "nf4",
            "compute_dtype": "bfloat16",
        },
        "config": {
            "width": 1280,
            "height": 720,
            "frames": 33,
            "fps": 16,
            "steps": 30,
            "cfg": 6.0,
            "seed": SEED,
        },
        "load_time_s": round(t_load, 2),
        "static_vram_gb": static_vram["allocated"],
        "prompts": prompt_results,
    }

    out_json = "wan21_14b_q4_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Đã hoàn tất toàn bộ 3 test prompts 14B 4-bit và lưu tại {out_json}!")

if __name__ == "__main__":
    main()
