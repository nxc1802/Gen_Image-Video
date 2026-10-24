#!/usr/bin/env python3
"""
🐉 Auto-Listener & Dragon Showcase Runner
================================================================================
Tự động lắng nghe tín hiệu Kaggle online qua ntfy.sh và kích hoạt Dragon Showcase Pipeline:
1. Bắt Cloudflare Public URL từ ntfy.sh.
2. Kiểm tra xác nhận /health online.
3. Chạy Qwen-Image 2.1 (GPU 0) sinh ảnh rồng Game of Thrones siêu sắc nét (Deep Focus, No Blur Background).
4. Chạy MiniMax H3 (GPU 1) sinh hoạt ảnh rồng phun lửa bay lượn với Camera Tracking Arc & Roar Audio.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


NTFY_TOPICS = [
    "studio-ai-url-cuongnguyen1802",
    "studio-ai-debug-cuongnguyen1802",
    "studio-ai-url-nguynxuncngde180528",
]


def poll_for_active_url(timeout_seconds: int = 1200) -> str:
    print("=" * 80)
    print("📡 [AUTO-LISTENER] BẮT ĐẦU LẮNG NGHE TÍN HIỆU TỪ KAGGLE DUAL-GPU VM...")
    print(f"   Thời gian chờ tối đa: {timeout_seconds}s")
    print("=" * 80)

    deadline = time.time() + timeout_seconds
    seen_messages = set()

    while time.time() < deadline:
        for topic in NTFY_TOPICS:
            try:
                req = urllib.request.Request(f"https://ntfy.sh/{topic}/raw?poll=1&since=5m")
                with urllib.request.urlopen(req, timeout=4) as resp:
                    content = resp.read().decode("utf-8").strip()
                    if content:
                        for line in content.splitlines():
                            line_str = line.strip()
                            if line_str and line_str not in seen_messages:
                                seen_messages.add(line_str)
                                print(f"[{time.strftime('%H:%M:%S')}] 📡 [{topic}] {line_str}", flush=True)

                                match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line_str)
                                if match:
                                    cand = match.group(0)
                                    print(f"🔍 Kiểm tra kết nối ứng viên: {cand} ...")
                                    try:
                                        with urllib.request.urlopen(f"{cand}/health", timeout=3) as h_resp:
                                            if h_resp.status == 200:
                                                print(f"\n🎉 [ONLINE] Xác nhận Cloudflare Tunnel hoạt động hoàn hảo: {cand}\n")
                                                # Lưu URL
                                                Path("current_url.txt").write_text(cand, encoding="utf-8")
                                                return cand
                                    except Exception:
                                        pass
            except Exception:
                pass
        time.sleep(6.0)

    return ""


def main():
    server_url = poll_for_active_url(timeout_seconds=900)
    if not server_url:
        print("❌ Hết thời gian chờ server online.")
        sys.exit(1)

    # Chạy run_dragon_pipeline.py
    cmd = [sys.executable, "run_dragon_pipeline.py", "--url", server_url]
    print(f"🚀 Kích hoạt quy trình sinh ảnh và video: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
