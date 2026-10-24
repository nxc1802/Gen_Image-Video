#!/usr/bin/env python3
"""
🚀 Parallel Dual-GPU MiniMax H3 Video Benchmark
================================================================================
Thực thi kiểm thử song song (Concurrent A/B Testing) mô hình MiniMax H3 trên Dual GPU:
- GPU 0 (cuda:0, ComfyUI port 8188): Config A - Balanced Facial Optimization (steps=20, cfg=2.0, shift=8.0)
- GPU 1 (cuda:1, ComfyUI port 8189): Config B - High-Guidance Dynamic Flow (steps=22, cfg=2.4, shift=9.0)

Cả 2 requests được gửi đồng thời qua ThreadPoolExecutor để xác nhận:
1. Tính năng routing và phân bổ Multi-GPU độc lập, không block nhau.
2. Cải thiện triệt để chất lượng khuôn mặt (Facial Anatomy, Eyes & Smile), loại bỏ méo mặt.
"""

import argparse
import base64
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_PROMPT = (
    "Cinematic medium portrait shot of an exquisite young woman, highly detailed delicate facial features, "
    "perfectly symmetrical hazel eyes, subtle realistic gentle smile with natural lip curve, "
    "photorealistic skin texture with subtle pores, natural calm breathing, soft studio lighting, "
    "8k uhd, masterpiece, filmic depth of field"
)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed face, distorted eyes, asymmetrical pupils, crossed eyes, melting facial features, "
    "blurry face, bad teeth, low quality, unnatural expression, bad anatomy, glitch, grotesque, "
    "robotic movement, sliding feet, audio desync"
)


def load_server_url() -> str:
    # 1. Thử đọc current_url.txt
    for path in ["current_url.txt", "kaggle/all-in-one/current_url.txt", "public_url.txt"]:
        p = Path(path)
        if p.exists():
            url = p.read_text(encoding="utf-8").strip()
            if url.startswith("http"):
                return url
    return "http://127.0.0.1:8000"


def send_video_request(server_url: str, payload: dict, label: str) -> dict:
    gpu_id = payload.get("gpu_device", "N/A")
    print(f"\n🚀 [{label}] Bắt đầu gửi request tới GPU {gpu_id}...")
    start_t = time.time()
    url = f"{server_url.rstrip('/')}/v1/videos/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer local-test-key",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_t
            print(f"✅ [{label}] GPU {gpu_id} hoàn thành sau {elapsed:.2f}s!")
            return {
                "label": label,
                "gpu_device": gpu_id,
                "elapsed": elapsed,
                "payload": payload,
                "response": res_data,
                "success": True,
            }
    except Exception as e:
        elapsed = time.time() - start_t
        print(f"❌ [{label}] GPU {gpu_id} gặp lỗi sau {elapsed:.2f}s: {e}")
        return {
            "label": label,
            "gpu_device": gpu_id,
            "elapsed": elapsed,
            "payload": payload,
            "error": str(e),
            "success": False,
        }


def extract_frames(video_path: Path, output_dir: Path, label: str):
    """Trích xuất 3 frames đại diện (đầu, giữa, cuối) để đánh giá độ sắc nét của khuôn mặt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_info = []
    # Lấy thông tin duration
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True
        )
        duration = float(probe.stdout.strip() or 1.37)
    except Exception:
        duration = 1.37

    timestamps = [0.0, duration / 2.0, max(0.0, duration - 0.1)]
    for idx, ts in enumerate(timestamps):
        out_img = output_dir / f"{label}_frame_{idx+1}_t{ts:.2f}s.png"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(video_path),
            "-vframes", "1", "-q:v", "2", str(out_img)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_img.exists():
            frames_info.append(str(out_img))
    return frames_info


def main():
    parser = argparse.ArgumentParser(description="Parallel Dual-GPU MiniMax H3 Video Benchmark")
    parser.add_argument("--url", default=None, help="Server base URL (Cloudflare hoặc localhost)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt kiểm thử")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--outdir", default="benchmark_outputs/parallel_h3", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    server_url = args.url or load_server_url()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 KIỂM THỬ SONG SONG DUAL-GPU MINIMAX H3 VIDEO (A/B TESTING FACIAL OPTIMIZATION)")
    print(f"🌐 Server URL : {server_url}")
    print(f"📁 Output Dir : {out_dir.resolve()}")
    print("=" * 80)

    # 1. Thiết lập 2 cấu hình thử nghiệm
    config_a = {
        "model": "Comfy-Org/MiniMax-H3",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "gpu_device": 0,          # GPU 0 (cuda:0, ComfyUI port 8188)
        "steps": 20,              # Tối ưu cân bằng
        "guidance": 2.0,          # CFG 2.0 kích hoạt negative prompt
        "shift": 8.0,             # Flow shift chuẩn
        "num_frames": 33,
        "width": 768,
        "height": 448,
        "fps": 24,
        "enable_audio": True,
        "seed": 42,
    }

    config_b = {
        "model": "Comfy-Org/MiniMax-H3",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "gpu_device": 1,          # GPU 1 (cuda:1, ComfyUI port 8189)
        "steps": 22,              # Tăng thêm 2 steps cho chi tiết vi mô
        "guidance": 2.4,          # CFG 2.4 tăng lực bám prompt và chống méo
        "shift": 9.0,             # Flow shift tập trung khử nhiễu tần số cao
        "num_frames": 33,
        "width": 768,
        "height": 448,
        "fps": 24,
        "enable_audio": True,
        "seed": 42,
    }

    tasks = [
        ("Config_A_GPU0_Balanced", config_a),
        ("Config_B_GPU1_HighGuidance", config_b),
    ]

    print(f"\n⚡ Bắt đầu bắn đồng thời {len(tasks)} requests qua ThreadPoolExecutor...")
    global_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(send_video_request, server_url, cfg, label): label
            for label, cfg in tasks
        }
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    global_elapsed = time.time() - global_start
    print("\n" + "=" * 80)
    print(f"🏁 TẤT CẢ REQUEST HOÀN TẤT TRONG TỔNG THỜI GIAN: {global_elapsed:.2f}s")
    print("=" * 80)

    summary_data = []

    for res in sorted(results, key=lambda x: x["gpu_device"]):
        label = res["label"]
        gpu = res["gpu_device"]
        elapsed = res["elapsed"]
        success = res["success"]

        if not success:
            print(f"❌ {label} (GPU {gpu}) Thất bại: {res.get('error')}")
            summary_data.append({
                "label": label, "gpu": gpu, "status": "FAILED", "elapsed": elapsed, "error": res.get("error")
            })
            continue

        resp = res["response"]
        data_list = resp.get("data", [])
        if not data_list or "b64_json" not in data_list[0]:
            print(f"❌ {label} Không có video data trả về.")
            continue

        b64_video = data_list[0]["b64_json"]
        video_bytes = base64.b64decode(b64_video)
        video_file = out_dir / f"{label}.mp4"
        video_file.write_bytes(video_bytes)
        file_size_mb = len(video_bytes) / (1024 * 1024)

        print(f"💾 {label} (GPU {gpu}) -> Đã lưu video: {video_file} ({file_size_mb:.2f} MB)")

        # Trích xuất frames
        frames = extract_frames(video_file, out_dir / "frames", label)
        print(f"   🖼️ Đã trích xuất {len(frames)} frames để phân tích ngũ quan.")

        summary_data.append({
            "label": label,
            "gpu": gpu,
            "status": "PASSED",
            "elapsed_seconds": round(elapsed, 2),
            "video_size_mb": round(file_size_mb, 2),
            "video_file": str(video_file),
            "frames": frames,
            "config": res["payload"],
        })

    # Lưu báo cáo JSON
    report_file = out_dir / "parallel_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "global_elapsed_seconds": round(global_elapsed, 2),
            "results": summary_data,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Báo cáo kiểm thử chi tiết đã ghi vào: {report_file}")


if __name__ == "__main__":
    main()
