#!/usr/bin/env python3
"""
🎬 Test Wan2.2 TI2V-5B Video Generation on Kaggle Studio
Stream SSE progress, track denoise steps, retrieve MP4 bytes and report performance metrics.
"""

import base64
import json
import os
import sys
import time
import socket
import requests
from pathlib import Path

# Patch socket.getaddrinfo nếu môi trường mạng chặn DNS của Cloudflare
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except Exception:
        if "trycloudflare.com" in str(host):
            return _orig_getaddrinfo("104.16.231.132", port, *args, **kwargs)
        raise
socket.getaddrinfo = _patched_getaddrinfo

# Xác định Base URL
url_arg = None
for arg in sys.argv[1:]:
    if arg.startswith("http"):
        url_arg = arg
        break

if not url_arg:
    pub_file = Path(__file__).parent / "public_url.txt"
    if pub_file.exists():
        url_arg = pub_file.read_text(encoding="utf-8").strip()

if not url_arg:
    url_arg = "http://localhost:8000"

clean_url = url_arg.rstrip("/")
if clean_url.endswith("/v1"):
    BASE_URL = clean_url[:-3]
    API_URL = clean_url
else:
    BASE_URL = clean_url
    API_URL = f"{clean_url}/v1"

OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = str(OUTPUT_DIR / "test_ti2v_5b.mp4")

print("=" * 72)
print("🎬 KIỂM THỬ VIDEO GENERATION: Wan2.2-TI2V-5B (Kaggle Dual T4x2)")
print("=" * 72)
print(f"🔗 Target Endpoint: {API_URL}/videos/generations")
print(f"📁 Video Output:    {OUTPUT_PATH}")

# 1. Kiểm tra Health & Models
print("\n📡 [1/2] Kiểm tra kết nối tới Kaggle Server...")
try:
    health_resp = requests.get(f"{BASE_URL}/health", timeout=15)
    print(f"✅ Health status: {health_resp.status_code} - {health_resp.text}")
except Exception as e:
    print(f"⚠️ Không gọi được /health ({e}), tiếp tục thử API...")

try:
    models_resp = requests.get(f"{API_URL}/models", timeout=15)
    if models_resp.status_code == 200:
        models_data = models_resp.json()
        print("📋 Danh sách model khả dụng trên server:")
        for m in models_data.get("data", []):
            print(f"   - {m.get('id')} ({m.get('type')})")
except Exception as e:
    print(f"⚠️ Lưu ý khi kiểm tra /models: {e}")

# 2. Gửi yêu cầu sinh video
prompt = "A high quality cinematic shot of a majestic luminescent jellyfish drifting gracefully through deep dark neon ocean water, bioluminescent particles floating around, 8k resolution, photorealistic, sharp focus, 3d render style"
payload = {
    "model": "ti2v_5b",
    "prompt": prompt,
    "negative_prompt": "",
    "num_frames": 17,
    "fps": 16,
    "width": 832,
    "height": 480,
    "steps": 20,
    "guidance": 5.0,
    "seed": 42,
    "stream": True,
}

print("\n🚀 [2/2] Bắt đầu gửi yêu cầu sinh video Wan2.2-TI2V-5B:")
print(f"📝 Prompt:          {prompt}")
print("⚙️ Specs:           832x480 | 17 frames | 20 steps (UniPC) | CFG 5.0 | Seed 42 | 4-bit NF4 FP16")
print("-" * 72)

t_start = time.time()
first_chunk_time = None
last_step_time = time.time()

try:
    resp = requests.post(f"{API_URL}/videos/generations", json=payload, stream=True, timeout=900)
    print(f"📡 Response HTTP Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"❌ Server trả về mã lỗi {resp.status_code}: {resp.text}")
        sys.exit(1)

    video_b64 = None
    step_times = []

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith(":"):
            now = time.time()
            print(f"\r💓 [SSE Keepalive] Server đang xử lý... Elapsed: {now - t_start:.1f}s", end="", flush=True)
            continue
        if line.startswith("event:"):
            continue
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                print("\n🏁 Nhận tín hiệu kết thúc stream [DONE]")
                break
            try:
                msg = json.loads(data_str)
                now = time.time()
                if first_chunk_time is None:
                    first_chunk_time = now

                if "step" in msg:
                    s = msg.get("step")
                    tot = msg.get("total_steps", 20)
                    pct = msg.get("progress", int((s/tot)*100))
                    st = msg.get("status", "diffusing")
                    dt = now - last_step_time
                    step_times.append(dt)
                    last_step_time = now
                    print(f"\r⏳ [Wan2.2-5B {st}] Step {s}/{tot} ({pct}%) | Step time: {dt:.2f}s | Elapsed: {now - t_start:.1f}s", end="", flush=True)

                elif "data" in msg:
                    video_b64 = msg["data"][0].get("b64_json")
                    server_elapsed = msg.get("x_inference_time_seconds", 0)
                    print(f"\n🎉 Nhận gói dữ liệu video hoàn tất! (Thời gian server: {server_elapsed:.2f}s)")

                elif "error" in msg:
                    print(f"\n❌ LỖI TỪ SERVER: {msg['error']}")
                    sys.exit(2)

            except Exception as pe:
                print(f"\n⚠️ Lỗi parse chunk: {pe} | Dữ liệu: {data_str[:120]}")

    total_time = time.time() - t_start
    if video_b64:
        video_bytes = base64.b64decode(video_b64)
        with open(OUTPUT_PATH, "wb") as f:
            f.write(video_bytes)
        file_size_kb = len(video_bytes) / 1024

        print("\n" + "=" * 72)
        print("🏆 KẾT QUẢ TEST WAN2.2-TI2V-5B THÀNH CÔNG RỰC RỠ!")
        print("=" * 72)
        print(f"📦 Kích thước file:     {file_size_kb:.1f} KB ({len(video_bytes):,} bytes)")
        print(f"⏱️ Tổng thời gian:      {total_time:.2f}s")
        if step_times:
            avg_step = sum(step_times) / len(step_times)
            print(f"⚡ Tốc độ trung bình:   {avg_step:.2f}s / step")
        print(f"🎬 File đã lưu tại:     {OUTPUT_PATH}")
        print("=" * 72)
    else:
        print("\n❌ Không nhận được payload video từ SSE stream!")
        sys.exit(3)

except requests.exceptions.RequestException as re:
    print(f"\n❌ Lỗi kết nối mạng: {re}")
    sys.exit(4)
