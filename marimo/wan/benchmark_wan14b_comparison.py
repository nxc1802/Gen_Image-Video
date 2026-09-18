#!/usr/bin/env python3
"""
📊 Wan2.1 1.3B vs 14B Comparative Benchmark (Everyday Prompt & VRAM Optimization)
--------------------------------------------------------------------------------
Thực nghiệm so sánh đối đầu giữa Wan2.1-T2V-1.3B và Wan2.1-T2V-14B:
- Cùng Prompt đời thường: Đánh giá chi tiết biểu cảm, khuôn mặt, ánh sáng và cử động tự nhiên
- Cùng Resolution: 832x480 (480p), 17 frames @ 16 fps, 10 steps, Guidance 5.0, Seed 42
- Cùng áp dụng trọn bộ VRAM Optimization:
  1. Host RAM Swap (Text Encoder UMT5-XXL swap sang CPU RAM ngay sau khi encode)
  2. Precision bfloat16
  3. PyTorch SDPA FlashAttention
  4. 3D VAE Tiling & Slicing
  5. CUDA Allocator expandable_segments:True & Garbage Collection
"""

import os
import sys
import gc
import time
import json
from typing import Dict, Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from diffusers import WanPipeline
import imageio

PROMPT = (
    "A close-up cinematic shot of a young woman in a soft knitted cream sweater sitting by a sunlit "
    "wooden cafe table, gently holding a steaming ceramic mug with both hands, taking a slow delicate sip "
    "and looking out the rainy window with a calm natural smile, soft warm morning lighting, "
    "raindrops on window glass, shallow depth of field, photorealistic 8k, fluid realistic subtle motion"
)
NEGATIVE_PROMPT = ""
WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 17
FPS = 16
STEPS = 10
GUIDANCE_SCALE = 5.0
SEED = 42

def get_vram_gb(device=0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
        "reserved": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
        "peak": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
    }

def run_model_benchmark(model_id: str, model_tag: str, output_mp4: str) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"🚀 BẮT ĐẦU BENCHMARK: {model_id} ({model_tag})")
    print(f"{'='*75}")
    
    # 0. Dọn dẹp VRAM trước khi nạp
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_baseline = get_vram_gb()
    print(f"[*] VRAM Baseline: Alloc={vram_baseline['allocated']} GB | Peak={vram_baseline['peak']} GB")

    t_load_start = time.time()
    print(f"[*] Đang nạp {model_id} (bfloat16)...")
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    t_load = time.time() - t_load_start
    print(f"[*] Nạp hoàn tất trong {t_load:.2f}s")

    # Kích hoạt VAE tiling & slicing
    if hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        print("[*] Đã kích hoạt 3D VAE Tiling & Slicing")

    # Cấu hình Host RAM Swap
    # Transformer & VAE lên GPU, Text Encoder lưu trên CPU Host RAM
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.text_encoder.to("cpu")
    
    vram_loaded_swap = get_vram_gb()
    print(f"[*] VRAM Static sau khi swap Text Encoder ra CPU RAM: Alloc={vram_loaded_swap['allocated']} GB | Res={vram_loaded_swap['reserved']} GB")

    # 1. Giai đoạn Text Encoding (Đẩy nhanh Text Encoder lên GPU để encode prompt)
    t_enc_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipe.text_encoder.to("cuda")
    vram_text_enc_pre = get_vram_gb()

    with torch.inference_mode():
        p_embeds, neg_embeds = pipe.encode_prompt(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT or None,
            do_classifier_free_guidance=(GUIDANCE_SCALE > 1.0),
            device=torch.device("cuda"),
            dtype=pipe.transformer.dtype,
        )
    vram_text_enc_peak = get_vram_gb()
    t_enc = time.time() - t_enc_start
    print(f"[STAGE 1 - Text Encode] Thời gian: {t_enc:.2f}s | Peak Alloc: {vram_text_enc_peak['peak']} GB")

    # 2. Swap ngay Text Encoder về Host CPU RAM để giải phóng toàn bộ VRAM
    t_swap_start = time.time()
    pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    t_swap = time.time() - t_swap_start
    vram_post_swap = get_vram_gb()
    print(f"[SWAP -> Host RAM] Thời gian: {t_swap:.2f}s | VRAM sau giải phóng: Alloc={vram_post_swap['allocated']} GB | Peak reset")

    # 3. Giai đoạn Denoising với Transformer DiT
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    
    t_denoise_start = time.time()
    with torch.inference_mode():
        pipe_output = pipe(
            prompt_embeds=p_embeds,
            negative_prompt_embeds=neg_embeds,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
    t_denoise = time.time() - t_denoise_start
    vram_denoise_peak = get_vram_gb()
    print(f"[STAGE 2 - Denoising DiT] Thời gian: {t_denoise:.2f}s | Peak Alloc: {vram_denoise_peak['peak']} GB | Reserved: {vram_denoise_peak['reserved']} GB")

    # 4. Xuất video MP4
    video_frames = pipe_output.frames[0]
    t_export_start = time.time()
    writer = imageio.get_writer(
        output_mp4,
        fps=FPS,
        codec="libx264",
        ffmpeg_params=[
            "-crf", "17",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
        ],
    )
    for frame in video_frames:
        frame_np = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        writer.append_data(frame_np)
    writer.close()
    t_export = time.time() - t_export_start
    file_size_kb = round(os.path.getsize(output_mp4) / 1024, 1)
    print(f"[EXPORT VIDEO] Đã lưu {output_mp4} ({file_size_kb} KB) trong {t_export:.2f}s")

    # Thu thập thông số
    total_time = t_enc + t_swap + t_denoise + t_export
    result = {
        "model_id": model_id,
        "model_tag": model_tag,
        "parameters_b": 1.3 if "1.3B" in model_id else 14.3,
        "output_file": output_mp4,
        "file_size_kb": file_size_kb,
        "timing": {
            "load_model_s": round(t_load, 2),
            "text_encode_s": round(t_enc, 2),
            "swap_to_cpu_s": round(t_swap, 2),
            "denoise_s": round(t_denoise, 2),
            "export_s": round(t_export, 2),
            "total_inference_s": round(total_time, 2),
        },
        "vram_gb": {
            "baseline_allocated": vram_baseline["allocated"],
            "static_after_swap": vram_loaded_swap["allocated"],
            "text_encoder_peak": vram_text_enc_peak["peak"],
            "post_swap_allocated": vram_post_swap["allocated"],
            "denoise_peak_allocated": vram_denoise_peak["peak"],
            "denoise_peak_reserved": vram_denoise_peak["reserved"],
        }
    }

    # Giải phóng hoàn toàn pipeline khỏi bộ nhớ trước lượt tiếp theo
    del pipe
    del p_embeds
    del neg_embeds
    del pipe_output
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[*] Đã giải phóng hoàn toàn {model_id} khỏi GPU.")
    return result

def main():
    print(f"Bắt đầu thực nghiệm so sánh Wan2.1 1.3B vs 14B...")
    print(f"Prompt: {PROMPT}")
    print(f"Config: {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames @ {FPS} fps, {STEPS} steps, Seed {SEED}")

    # 1. Benchmark Wan2.1 1.3B
    res_1_3b = run_model_benchmark(
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        model_tag="Wan2.1 1.3B (Everyday Benchmark)",
        output_mp4="wan21_1_3b_everyday.mp4",
    )

    # 2. Benchmark Wan2.1 14B
    res_14b = run_model_benchmark(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        model_tag="Wan2.1 14B (Everyday Benchmark)",
        output_mp4="wan21_14b_everyday.mp4",
    )

    # 3. Tổng hợp kết quả
    comparison = {
        "prompt": PROMPT,
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "frames": NUM_FRAMES,
            "fps": FPS,
            "steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
        },
        "models": {
            "wan21_1_3b": res_1_3b,
            "wan21_14b": res_14b,
        }
    }

    json_path = "wan21_comparison_1_3b_vs_14b.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*75}")
    print(f"🏁 TỔNG KẾT SO SÁNH WAN2.1 1.3B VS 14B (EVERYDAY PROMPT)")
    print(f"{'='*75}")
    print(f"{'Tiêu chí':<32} | {'Wan2.1 1.3B':<18} | {'Wan2.1 14B':<18}")
    print(f"{'-'*32} | {'-'*18} | {'-'*18}")
    v1_alloc = res_1_3b['vram_gb']['denoise_peak_allocated']
    v2_alloc = res_14b['vram_gb']['denoise_peak_allocated']
    v1_res = res_1_3b['vram_gb']['denoise_peak_reserved']
    v2_res = res_14b['vram_gb']['denoise_peak_reserved']
    t1_enc = res_1_3b['timing']['text_encode_s']
    t2_enc = res_14b['timing']['text_encode_s']
    t1_den = res_1_3b['timing']['denoise_s']
    t2_den = res_14b['timing']['denoise_s']
    t1_tot = res_1_3b['timing']['total_inference_s']
    t2_tot = res_14b['timing']['total_inference_s']
    sz1 = res_1_3b['file_size_kb']
    sz2 = res_14b['file_size_kb']

    print(f"{'Số lượng tham số':<32} | {'1.3B DiT':<18} | {'14.3B DiT':<18}")
    print(f"{'Denoise Peak VRAM (Allocated)':<32} | {f'{v1_alloc} GB':<18} | {f'{v2_alloc} GB':<18}")
    print(f"{'Denoise Peak VRAM (Reserved)':<32} | {f'{v1_res} GB':<18} | {f'{v2_res} GB':<18}")
    print(f"{'Thời gian Text Encode':<32} | {f'{t1_enc}s':<18} | {f'{t2_enc}s':<18}")
    print(f"{'Thời gian Denoising (10 steps)':<32} | {f'{t1_den}s':<18} | {f'{t2_den}s':<18}")
    print(f"{'Tổng thời gian sinh video':<32} | {f'{t1_tot}s':<18} | {f'{t2_tot}s':<18}")
    print(f"{'Kích thước file video':<32} | {f'{sz1} KB':<18} | {f'{sz2} KB':<18}")
    print(f"\nĐã lưu báo cáo JSON tại {json_path}!")

if __name__ == "__main__":
    main()
