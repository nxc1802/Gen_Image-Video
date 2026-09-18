"""
Wan2.2 Comprehensive Benchmark Suite: Wan2.2-TI2V-5B vs Wan2.2-T2V-A14B
4-Bit NF4 Quantization + Host RAM Swap + FP32 VAE Tiling/Slicing
UniPCMultistepScheduler (20 steps) on 3 Standard Prompts
"""
import os
import gc
import sys
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

from diffusers import (
    BitsAndBytesConfig,
    WanTransformer3DModel,
    WanPipeline,
    AutoencoderKLWan,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel

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

NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurry details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts"
)
SEED = 42
OUTPUT_DIR = "/root/wan22_benchmark/outputs"
METRICS_FILE = "/root/wan22_benchmark/wan22_benchmark_metrics.json"

def get_vram_gb(device=0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
        "reserved": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
        "peak": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
    }

def encode_all_prompts(repo_id: str, device: str = "cuda") -> Dict[str, Any]:
    print(f"\n[*] Đang mã hóa {len(TEST_PROMPTS)} prompts qua UMT5 Text Encoder ({repo_id})...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)

    pipe_temp = WanPipeline.from_pretrained(
        repo_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    )

    encoded = {}
    for p in TEST_PROMPTS:
        t0 = time.time()
        prompt_embeds, neg_embeds = pipe_temp.encode_prompt(
            prompt=p["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            do_classifier_free_guidance=True,
            device=device,
            dtype=torch.bfloat16,
        )
        encode_time = time.time() - t0
        # Move embeddings to CPU to preserve GPU memory during DiT loading
        encoded[p["id"]] = {
            "prompt_embeds": prompt_embeds.cpu(),
            "neg_embeds": neg_embeds.cpu(),
            "encode_time": round(encode_time, 3),
        }
        print(f"  [+] Đã encode '{p['id']}' ({encode_time:.2f}s)")

    # Clean up text encoder completely from GPU
    del pipe_temp, text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"[*] Text Encoder đã giải phóng hoàn toàn khỏi GPU VRAM.")
    return encoded

def run_benchmark_model(
    model_name: str,
    repo_id: str,
    width: int,
    height: int,
    num_frames: int = 33,
    num_steps: int = 20,
    guidance_scale: float = 5.0,
    guidance_scale_2: float = 5.0,
    is_two_stage: bool = False,
    expand_timesteps: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*80}")
    print(f"🚀 BẮT ĐẦU BENCHMARK: {model_name}")
    print(f"   Repo: {repo_id}")
    print(f"   Độ phân giải: {width}x{height} | {num_frames} frames @ 16 fps | {num_steps} steps")
    print(f"   Guidance: {guidance_scale}" + (f" / {guidance_scale_2}" if is_two_stage else ""))
    print(f"{'='*80}")

    # 1. Encode prompts
    encoded_prompts = encode_all_prompts(repo_id, device=device)

    # 2. Configure 4-bit NF4 Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Load Transformer(s) in 4-bit NF4
    t_load_start = time.time()
    print(f"\n[*] Đang nạp Transformer 1 trong 4-bit NF4 ({repo_id})...")
    transformer_1 = WanTransformer3DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    transformer_2 = None
    if is_two_stage:
        print(f"[*] Đang nạp Transformer 2 (Low-Noise Stage) trong 4-bit NF4 ({repo_id})...")
        transformer_2 = WanTransformer3DModel.from_pretrained(
            repo_id,
            subfolder="transformer_2",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

    pipe = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        scheduler=scheduler,
        transformer=transformer_1,
        transformer_2=transformer_2,
        boundary_ratio=0.875 if is_two_stage else None,
        expand_timesteps=expand_timesteps,
    )
    load_time = time.time() - t_load_start
    vram_after_dit = get_vram_gb()
    print(f"[*] DiT Transformer tải hoàn tất trong {load_time:.2f}s!")
    print(f"    VRAM tĩnh sau khi nạp DiT: {vram_after_dit['allocated']} GB Alloc | {vram_after_dit['reserved']} GB Res")

    results = []
    generated_latents = {}

    # 4. Denoising loop for each prompt
    for idx, p_info in enumerate(TEST_PROMPTS, 1):
        p_id = p_info["id"]
        p_name = p_info["name"]
        print(f"\n--- [Step 1/2 Denoise] ({idx}/3) {model_name} | Prompt: {p_id} ({p_name}) ---")

        p_data = encoded_prompts[p_id]
        p_embeds = p_data["prompt_embeds"].to(device)
        n_embeds = p_data["neg_embeds"].to(device)

        torch.cuda.reset_peak_memory_stats()
        t_denoise_start = time.time()

        call_kwargs = {
            "prompt_embeds": p_embeds,
            "negative_prompt_embeds": n_embeds,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator(device=device).manual_seed(SEED),
            "output_type": "latent",
            "return_dict": False,
        }
        if is_two_stage:
            call_kwargs["guidance_scale_2"] = guidance_scale_2

        latents = pipe(**call_kwargs)[0]
        denoise_time = time.time() - t_denoise_start
        peak_denoise_vram = get_vram_gb()

        print(f"  [+] Denoise hoàn tất trong {denoise_time:.2f}s! ({denoise_time/num_steps:.2f}s/step)")
        print(f"      Peak VRAM Denoise: {peak_denoise_vram['peak']} GB")

        generated_latents[p_id] = {
            "latents": latents.cpu(),
            "denoise_time": round(denoise_time, 2),
            "peak_vram_denoise": peak_denoise_vram["peak"],
            "encode_time": p_data["encode_time"],
        }

    # 5. Clean up Transformer(s) from GPU to give full room to VAE
    print(f"\n[*] Giải phóng DiT Transformer khỏi GPU để nạp FP32 VAE...")
    del pipe, transformer_1
    if transformer_2 is not None:
        del transformer_2
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 6. Load VAE in FP32 with Tiling & Slicing
    print(f"[*] Nạp AutoencoderKLWan FP32 lên GPU...")
    vae = AutoencoderKLWan.from_pretrained(repo_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    # 7. Decode all prompts
    for idx, p_info in enumerate(TEST_PROMPTS, 1):
        p_id = p_info["id"]
        p_name = p_info["name"]
        print(f"\n--- [Step 2/2 Decode] ({idx}/3) {model_name} | Prompt: {p_id} ---")

        l_info = generated_latents[p_id]
        latents = l_info["latents"].to(device=device, dtype=torch.float32)

        torch.cuda.reset_peak_memory_stats()
        t_decode_start = time.time()

        with torch.no_grad():
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
            norm_latents = latents / latents_std + latents_mean
            video = vae.decode(norm_latents, return_dict=False)[0]

            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.cpu().float().permute(0, 2, 3, 4, 1).squeeze(0)
            frames = [Image.fromarray((f.numpy() * 255).astype(np.uint8)) for f in video]

        decode_time = time.time() - t_decode_start
        peak_decode_vram = get_vram_gb()
        print(f"  [+] VAE Decode hoàn tất trong {decode_time:.2f}s ({len(frames)} frames)")
        print(f"      Peak VRAM Decode: {peak_decode_vram['peak']} GB")

        # Save video and keyframe PNG
        prefix = "wan22_5b_q4_opt" if not is_two_stage else "wan22_a14b_q4_opt"
        out_mp4 = os.path.join(OUTPUT_DIR, f"{prefix}_{p_id}.mp4")
        out_png = os.path.join(OUTPUT_DIR, f"{prefix}_{p_id}_frame16.png")

        export_to_video(frames, out_mp4, fps=16)
        frames[min(16, len(frames)-1)].save(out_png)

        file_size_mb = round(os.path.getsize(out_mp4) / (1024**2), 2)
        total_time = round(l_info["encode_time"] + l_info["denoise_time"] + decode_time, 2)
        overall_peak_vram = max(l_info["peak_vram_denoise"], peak_decode_vram["peak"])

        metric_record = {
            "model_name": model_name,
            "repo_id": repo_id,
            "prompt_id": p_id,
            "prompt_name": p_name,
            "resolution": f"{width}x{height}",
            "frames": num_frames,
            "steps": num_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2 if is_two_stage else None,
            "encode_time_s": l_info["encode_time"],
            "denoise_time_s": l_info["denoise_time"],
            "decode_time_s": round(decode_time, 2),
            "total_time_s": total_time,
            "peak_vram_gb": overall_peak_vram,
            "video_size_mb": file_size_mb,
            "video_path": out_mp4,
            "keyframe_path": out_png,
        }
        results.append(metric_record)
        print(f"  [+] Đã lưu MP4: {out_mp4} ({file_size_mb} MB) | Total: {total_time}s | Peak VRAM: {overall_peak_vram} GB")

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    print(f"\n{'#'*80}")
    print("🔥 KHỞI ĐỘNG SUITE BENCHMARK WAN2.2: TI2V-5B VÀ T2V-A14B (4-BIT NF4)")
    print("   Áp dụng: 4-bit NF4 Quantization + Host RAM Swap + FP32 VAE Tiling/Slicing")
    print(f"{'#'*80}")

    all_metrics = {}

    # Batch 1: Wan2.2-TI2V-5B
    metrics_5b = run_benchmark_model(
        model_name="Wan2.2-TI2V-5B-4bit",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        width=832,
        height=480,
        num_frames=33,
        num_steps=20,
        guidance_scale=5.0,
        is_two_stage=False,
        expand_timesteps=True,
    )
    all_metrics["wan22_5b_q4"] = metrics_5b

    # Batch 2: Wan2.2-T2V-A14B
    metrics_a14b = run_benchmark_model(
        model_name="Wan2.2-T2V-A14B-4bit",
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        width=1280,
        height=720,
        num_frames=33,
        num_steps=20,
        guidance_scale=5.0,
        guidance_scale_2=5.0,
        is_two_stage=True,
        expand_timesteps=False,
    )
    all_metrics["wan22_a14b_q4"] = metrics_a14b

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"🎉 TOÀN BỘ BENCHMARK WAN2.2 HOÀN TẤT!")
    print(f"   Kết quả JSON đã lưu tại: {METRICS_FILE}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
