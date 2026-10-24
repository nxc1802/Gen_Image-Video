#!/usr/bin/env python3
"""
🚀 End-to-End Verification Suite for MiniMax H3:
1. Ref2VA Multi-Reference (<Picture 1>, <Picture 2>) + Turbo LoRA (6 Steps) + Stereo Audio
2. Dynamic UNet Hot-Swap (Ref2VA <-> FL2VA) + RAM Guard on Kaggle Dual-T4
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = WORKSPACE / "benchmark_outputs" / "ref2va_turbo_showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "sk-studio-local"


def wait_for_kaggle_url(timeout_sec: int = 600) -> str:
    print("⏳ Đang lắng nghe Cloudflare Tunnel URL từ Kaggle (ntfy)...")
    t0 = time.time()
    topics = ["studio-ai-url-nguynxuncngde180528", "studio-ai-url-cuongnguyen1802"]
    while time.time() - t0 < timeout_sec:
        for topic in topics:
            try:
                req = urllib.request.Request(
                    f"https://ntfy.sh/{topic}/json?poll=1&since=10m",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    lines = [l.strip() for l in resp.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
                for line in reversed(lines):
                    data = json.loads(line)
                    msg = data.get("message", "")
                    if "trycloudflare.com" in msg:
                        import re
                        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", msg)
                        if m:
                            cand_url = m.group(0)
                            # Kiểm tra health thực tế
                            try:
                                h_req = urllib.request.Request(f"{cand_url}/health", headers={"User-Agent": "Mozilla/5.0"})
                                with urllib.request.urlopen(h_req, timeout=5) as h_res:
                                    if h_res.status == 200:
                                        (WORKSPACE / "current_url.txt").write_text(cand_url + "\n", encoding="utf-8")
                                        print(f"✅ Đã kết nối thành công tới Kaggle Server: {cand_url}")
                                        return cand_url
                            except Exception:
                                pass
            except Exception:
                pass
        time.sleep(6)
    raise TimeoutError("Hết thời gian chờ Kaggle Server khởi động.")


def api_post(base_url: str, path: str, payload: dict, timeout: int = 900) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "MMO-Studio-Client",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore").strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith(":")]
        return json.loads(lines[-1] if lines else raw)


def main():
    base_url = None
    url_file = WORKSPACE / "current_url.txt"
    if url_file.exists():
        cand = url_file.read_text(encoding="utf-8").strip()
        if cand:
            try:
                with urllib.request.urlopen(f"{cand}/health", timeout=5) as r:
                    if r.status == 200:
                        base_url = cand
            except Exception:
                pass

    if not base_url:
        base_url = wait_for_kaggle_url(timeout_sec=600)

    # 1. Kiểm tra /health và danh sách linked_models
    with urllib.request.urlopen(f"{base_url}/health", timeout=10) as r:
        health = json.loads(r.read().decode("utf-8"))
    print("\n📊 [1/4] Trạng thái /health & Linked Models:")
    gpu1_diag = health.get("comfyui_diagnostics", {}).get("gpu_1", {})
    linked = gpu1_diag.get("linked_models", {})
    for k, v in linked.items():
        print(f"   🔗 {k} -> {v}")

    # 2. Chuẩn bị 2 ảnh Reference (<Picture 1>: Nhân vật, <Picture 2>: Trang phục/Bối cảnh)
    print("\n🖼️ [2/4] Chuẩn bị bộ 2 ảnh Reference (<Picture 1> & <Picture 2>)...")
    ref1_path = WORKSPACE / "benchmark_outputs" / "disney_showcase" / "disney_witch_image.png"
    ref2_path = WORKSPACE / "benchmark_outputs" / "dragon_showcase" / "dragon_qwen_image.png"

    if not ref1_path.exists():
        ref1_path = WORKSPACE / "benchmark_outputs" / "qwen_t2i.png"
    if not ref2_path.exists():
        ref2_path = WORKSPACE / "benchmark_outputs" / "qwen_instruction_edit.png"

    b64_ref1 = base64.b64encode(ref1_path.read_bytes()).decode("utf-8")
    b64_ref2 = base64.b64encode(ref2_path.read_bytes()).decode("utf-8")
    print(f"   ✅ <Picture 1>: {ref1_path.name} ({ref1_path.stat().st_size / 1024:.1f} KB)")
    print(f"   ✅ <Picture 2>: {ref2_path.name} ({ref2_path.stat().st_size / 1024:.1f} KB)")

    # 3. Chạy MiniMax H3 Ref2VA (Multi-Reference) + Turbo LoRA (6 Steps) + Audio
    print("\n🎬 [3/4] Đang sinh video MiniMax H3 Ref2VA (Multi-Reference <Picture 1> + <Picture 2>) với Turbo LoRA (6 steps)...")
    t0 = time.time()
    ref2va_payload = {
        "model": "Comfy-Org/MiniMax-H3-Ref2VA-Turbo",
        "mode": "ref2va",
        "turbo": True,
        "steps": 6,
        "width": 768,
        "height": 448,
        "num_frames": 33,
        "fps": 24,
        "enable_audio": True,
        "reference_images": [
            f"data:image/png;base64,{b64_ref1}",
            f"data:image/png;base64,{b64_ref2}",
        ],
        "prompt": (
            "3D Disney Pixar cinematic animation: The cute young apprentice witch from <Picture 1> "
            "stands heroically beside the majestic armoured dragon from <Picture 2>, waving her glowing wand "
            "as golden sparks swirl around them. Smooth dynamic camera orbit, synchronized magical chimes and gentle dragon rumble."
        ),
    }
    res_ref2va = api_post(base_url, "/v1/videos/generations", ref2va_payload, timeout=900)
    elapsed_ref2va = time.time() - t0
    vid_b64 = res_ref2va["data"][0]["b64_json"]
    out_ref2va = OUT_DIR / "minimax_ref2va_turbo_multi_image.mp4"
    out_ref2va.write_bytes(base64.b64decode(vid_b64))
    print(f"✅ [Ref2VA + Turbo 6-step] Hoàn thành trong {elapsed_ref2va:.2f}s! Lưu tại: {out_ref2va} ({out_ref2va.stat().st_size / 1024:.1f} KB)")

    # 4. Kiểm tra Dynamic Hot-Swap sang FL2VA + Turbo LoRA (6 Steps)
    print("\n🔄 [4/4] Kiểm tra Dynamic Hot-Swap (Ref2VA -> FL2VA + Turbo LoRA 6 steps) và RAM Guard...")
    t1 = time.time()
    fl2va_payload = {
        "model": "Comfy-Org/MiniMax-H3-Turbo",
        "mode": "fl2va",
        "turbo": True,
        "steps": 6,
        "width": 768,
        "height": 448,
        "num_frames": 33,
        "fps": 24,
        "enable_audio": True,
        "image": f"data:image/png;base64,{b64_ref1}",
        "prompt": (
            "3D Disney Pixar animation: The young witch smiles brightly and casts a shimmering golden starlight spell "
            "from her hands, camera gently dollies in with sparkling magical chime sound effects."
        ),
    }
    res_fl2va = api_post(base_url, "/v1/videos/generations", fl2va_payload, timeout=900)
    elapsed_fl2va = time.time() - t1
    vid_fl2va_b64 = res_fl2va["data"][0]["b64_json"]
    out_fl2va = OUT_DIR / "minimax_fl2va_turbo_hotswap.mp4"
    out_fl2va.write_bytes(base64.b64decode(vid_fl2va_b64))
    print(f"✅ [FL2VA Hot-Swap + Turbo 6-step] Hoàn thành trong {elapsed_fl2va:.2f}s! Lưu tại: {out_fl2va} ({out_fl2va.stat().st_size / 1024:.1f} KB)")

    # Kiểm tra RAM sau khi Hot-Swap
    with urllib.request.urlopen(f"{base_url}/health", timeout=10) as r:
        h_after = json.loads(r.read().decode("utf-8"))
    ram_info = h_after.get("vram_status", {}).get("ram", {})
    print(f"\n🧠 Trạng thái System RAM sau Hot-Swap: {ram_info}")


if __name__ == "__main__":
    main()
