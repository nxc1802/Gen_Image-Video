#!/usr/bin/env python3
"""
🎬 Auto-Listener & 5-Part Video Chain Runner
================================================================================
Tự động lắng nghe tín hiệu Kaggle online và chạy chuỗi 5 video liên tiếp!
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


NTFY_TOPICS = [
    "studio-ai-url-cuongnguyen1802",
    "studio-ai-debug-cuongnguyen1802",
    "studio-ai-url-nguynxuncngde180528",
]


def poll_for_active_url(timeout_seconds: int = 1200) -> str:
    print("=" * 80)
    print("📡 [AUTO-LISTENER] ĐANG LẮNG NGHE TÍN HIỆU SERVER ONLINE...")
    print("=" * 80)

    deadline = time.time() + timeout_seconds
    seen = set()

    while time.time() < deadline:
        for topic in NTFY_TOPICS:
            try:
                req = urllib.request.Request(f"https://ntfy.sh/{topic}/raw?poll=1&since=5m")
                with urllib.request.urlopen(req, timeout=4) as resp:
                    content = resp.read().decode("utf-8").strip()
                    if content:
                        for line in content.splitlines():
                            l = line.strip()
                            if l and l not in seen:
                                seen.add(l)
                                print(f"[{time.strftime('%H:%M:%S')}] 📡 {l}", flush=True)
                                m = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", l)
                                if m:
                                    cand = m.group(0)
                                    print(f"🔍 Kiểm tra {cand}...")
                                    try:
                                        with urllib.request.urlopen(f"{cand}/health", timeout=3) as h:
                                            if h.status == 200:
                                                print(f"🎉 ONLINE: {cand}")
                                                Path("current_url.txt").write_text(cand, encoding="utf-8")
                                                return cand
                                    except Exception:
                                        pass
            except Exception:
                pass
        time.sleep(6)
    return ""


def main():
    url = poll_for_active_url(2400)
    if not url:
        print("❌ Timeout.")
        sys.exit(1)

    cmd = [sys.executable, "run_video_chain_5parts.py", "--url", url]
    print(f"🚀 Kích hoạt: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
