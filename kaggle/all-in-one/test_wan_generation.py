#!/usr/bin/env python3
"""
Test Wan2.1 1.3B Video Generation on Kernel 31
Stream SSE progress, retrieve video MP4 bytes, save to test_outputs/k31_wan_video.mp4
"""

import base64
import json
import os
import sys
import time
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://season-charge-wake-exchange.trycloudflare.com/v1"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "test_outputs/k35_wan_video.mp4"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

prompt = "A majestic neon dragon soaring across a cyberpunk metropolis, volumetric fog, cinematic lighting, 4k ultra detailed"
payload = {
    "model": "1.3b",
    "prompt": prompt,
    "num_frames": 17,
    "fps": 16,
    "width": 832,
    "height": 480,
    "steps": 25,
    "guidance": 5.0,
    "seed": 42,
    "stream": True,
}

print(f"🎬 Gửi yêu cầu sinh video Wan2.1 1.3B tới: {BASE_URL}/videos/generations")
print(f"📝 Prompt: {prompt}")
print(f"⚙️ Config: 17 frames, 832x480, 25 steps (UniPCMultistepScheduler), guidance 5.0, seed 42")
t0 = time.time()

try:
    resp = requests.post(f"{BASE_URL}/videos/generations", json=payload, stream=True, timeout=900)
    print(f"📡 Response Status: {resp.status_code}")
    
    video_b64 = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            ev_type = line[6:].strip()
            continue
        if line.startswith("data:"):
            d_str = line[5:].strip()
            if d_str == "[DONE]":
                print("\n🏁 Nhận tín hiệu [DONE]")
                break
            try:
                d_json = json.loads(d_str)
                if "step" in d_json:
                    s = d_json.get("step")
                    tot = d_json.get("total_steps")
                    pct = d_json.get("progress")
                    st = d_json.get("status", "diffusing")
                    print(f"\r⏳ [Wan Diffusion] Bước {s}/{tot} ({pct}%) [{st}] - {time.time()-t0:.1f}s elapsed", end="", flush=True)
                elif "data" in d_json:
                    video_b64 = d_json["data"][0].get("b64_json")
                    inf_time = d_json.get("x_inference_time_seconds", 0)
                    print(f"\n🎉 Nhận video dữ liệu! Thời gian xử lý: {inf_time:.2f}s")
                elif "error" in d_json:
                    print(f"\n❌ Lỗi từ server: {d_json}")
            except Exception as e:
                print(f"\nParse note: {e} - Raw: {d_str[:100]}")

    elapsed = time.time() - t0
    if video_b64:
        vid_bytes = base64.b64decode(video_b64)
        with open(OUTPUT_PATH, "wb") as f:
            f.write(vid_bytes)
        print(f"\n✅ Đã lưu video thành công: {OUTPUT_PATH} ({len(vid_bytes):,} bytes) trong {elapsed:.2f}s")
    else:
        print(f"\n❌ Không nhận được video base64 payload!")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Exception: {e}")
    sys.exit(1)
