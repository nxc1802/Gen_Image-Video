#!/usr/bin/env python3
"""
FLUX.1 Supabase GPU Queue Worker
----------------------------------------------------------------
Chạy trực tiếp trên máy GPU (Molab / Cloud GPU).
Chỉ thực hiện kết nối OUTBOUND qua HTTPS (cổng 443) tới Supabase.
Không mở cổng, không dùng tunnel, không bao giờ bị bot phát hiện hay kill pod!
"""

import argparse
import base64
import gc
import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from supabase import Client, create_client

pipeline = None
system_info = {}


def init_gpu():
    global system_info
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        system_info = {
            "device": "cuda",
            "name": dev_name,
            "vram_gb": round(vram_gb, 1),
            "bf16": torch.cuda.is_bf16_supported(),
        }
    else:
        system_info = {"device": "cpu", "name": "CPU", "vram_gb": 0.0, "bf16": False}
    print(f"🖥️ Thiết bị: {system_info['name']} | VRAM: {system_info['vram_gb']}GB | bfloat16: {system_info['bf16']}")


def load_model(model_id: str = "black-forest-labs/FLUX.1-dev", vram_strategy: str = "full_gpu"):
    global pipeline
    from diffusers import FluxPipeline

    token = os.environ.get("HF_TOKEN") or None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = torch.bfloat16 if system_info.get("bf16", False) else torch.float32
    print(f"⏳ Đang nạp {model_id} ({dtype})...")
    start_load = time.time()

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )

    if vram_strategy == "full_gpu" and system_info["device"] == "cuda":
        pipe.to("cuda")
        print("⚡ Mô hình chạy 100% trên GPU VRAM (Tốc độ tối đa).")
    elif vram_strategy == "cpu_offload":
        pipe.enable_model_cpu_offload()
        print("⚡ Mô hình chạy với Model CPU Offload.")
    else:
        pipe.to("cpu")

    pipeline = pipe
    print(f"✅ Nạp mô hình hoàn tất trong {time.time() - start_load:.1f}s!")
    return pipeline


def parse_resolution(size_str: Optional[str]) -> tuple[int, int]:
    if not size_str:
        return 1024, 1024
    try:
        parts = size_str.lower().replace(" ", "").split("x")
        w = (int(parts[0]) // 16) * 16
        h = (int(parts[1]) // 16) * 16
        return max(256, min(2048, w)), max(256, min(2048, h))
    except Exception:
        return 1024, 1024


def run_worker(supabase: Client, poll_interval: float = 1.0):
    print("=" * 70)
    print("🚀 FLUX.1 SUPABASE WORKER ĐANG HOẠT ĐỘNG")
    print("   Đang lắng nghe hàng đợi 'image_jobs' từ Supabase...")
    print("   (Chỉ kết nối Outbound HTTPS, an toàn 100% trước mọi tường lửa)")
    print("=" * 70)

    job_counter = 0

    while True:
        try:
            # Lấy 1 job có status = 'pending' cũ nhất
            res = (
                supabase.table("image_jobs")
                .select("*")
                .eq("status", "pending")
                .order("created_at")
                .limit(1)
                .execute()
            )

            if not res.data or len(res.data) == 0:
                time.sleep(poll_interval)
                continue

            job = res.data[0]
            job_id = job["id"]

            # Khóa job: chuyển status sang 'processing'
            lock_res = (
                supabase.table("image_jobs")
                .update({"status": "processing"})
                .eq("id", job_id)
                .eq("status", "pending")
                .execute()
            )

            # Nếu bị worker khác nhận trước, bỏ qua
            if not lock_res.data:
                continue

            job_counter += 1
            prompt = job.get("prompt", "")
            size_str = job.get("size", "1024x1024")
            w, h = parse_resolution(size_str)
            model_name = (job.get("model") or "flux-1-dev").lower()
            is_schnell = "schnell" in model_name

            steps = job.get("steps") or (4 if is_schnell else 28)
            guidance = job.get("guidance")
            if guidance is None:
                guidance = 0.0 if is_schnell else 3.5

            seed = job.get("seed")
            if seed is None:
                seed = int(time.time() * 1000) % 2147483647

            print(f"\n[JOB #{job_counter}] Bắt đầu xử lý Job #{job_id}")
            print(f"  Prompt   : {prompt[:70]}...")
            print(f"  Config   : {w}x{h} | {steps} steps | Guidance: {guidance} | Seed: {seed}")

            dev = system_info.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            generator = torch.Generator(device=dev).manual_seed(seed)
            max_seq = 256 if steps <= 4 else 512

            start_t = time.time()
            try:
                result = pipeline(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    max_sequence_length=max_seq,
                )
                img = result.images[0]
                elapsed = time.time() - start_t

                # Chuyển đổi sang base64 PNG
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

                # Cập nhật thành công lên Supabase
                supabase.table("image_jobs").update({
                    "status": "completed",
                    "result_b64": b64_str,
                    "inference_time_sec": round(elapsed, 2),
                    "device_name": system_info.get("name", "GPU"),
                }).eq("id", job_id).execute()

                print(f"  ✅ Hoàn tất trong {elapsed:.2f}s! Đã cập nhật lên Supabase.")

            except Exception as gen_err:
                print(f"  ❌ Lỗi suy luận: {gen_err}")
                supabase.table("image_jobs").update({
                    "status": "failed",
                    "error_message": str(gen_err),
                }).eq("id", job_id).execute()

        except Exception as loop_err:
            print(f"⚠️ Cảnh báo vòng lặp worker: {loop_err}")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="FLUX.1 Supabase GPU Queue Worker")
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL", "https://fxepzlszglckfsscport.supabase.co"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY"))
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--vram", default="full_gpu", choices=["full_gpu", "cpu_offload"])
    args = parser.parse_args()

    if not args.supabase_key:
        print("❌ Thiếu Supabase Key! Vui lòng truyền --supabase-key hoặc export SUPABASE_KEY='...'")
        sys.exit(1)

    init_gpu()
    load_model(args.model, args.vram)

    sb_client = create_client(args.supabase_url, args.supabase_key)
    run_worker(sb_client)


if __name__ == "__main__":
    main()
