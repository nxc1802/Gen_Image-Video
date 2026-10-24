#!/usr/bin/env python3
"""
✨ Disney 3D Animation & Camera Trajectory Pipeline
================================================================================
Thử nghiệm tính năng điều khiển góc quay camera theo quỹ đạo đường line (Camera Trajectory):
1. Qwen-Image 2.1 (GPU 0):
   - Tạo ảnh nhân vật 3D Disney/Pixar (cô bé phù thủy nhỏ dễ thương nâng quả cầu ma thuật phát sáng trong thư viện cổ tích).
2. MiniMax H3 (GPU 1):
   - Nhận ảnh gốc làm mỏ neo không gian 3D.
   - Thực thi điều hướng camera theo đường cong xoắn ốc (Spiral Sweeping Arc Tracking Shot):
     Camera lượn từ góc thấp phía trước vòng qua sườn, xoay 180 độ ra sau vai và nâng cao lên (Elevated crane shot).
   - Tạo âm thanh ma thuật lung linh và tiếng cười khúc khích dễ thương.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DISNEY_IMAGE_PROMPT = (
    "A charming Disney Pixar 3D animated style scene of an adorable little apprentice witch girl with big expressive emerald green eyes, "
    "rosy cheeks, a delighted cute smile, messy brown hair in a loose ponytail, wearing a cozy oversized knitted purple sweater and a small pointy witch hat. "
    "She is holding a glowing warm golden magical orb in her cupped hands, sparkling starlight dust floating in the air. "
    "Background is a cozy, warm fairytale wooden library with bookshelves full of ancient glowing spellbooks, potion bottles, and soft magical candlelight. "
    "Disney Pixar 3D character render, Octane 3D, subsurface scattering on skin, rich textures, vibrant warm colors, sharp focus, masterpiece"
)

DISNEY_IMAGE_NEGATIVE = (
    "low quality, blurry, 2d, flat drawing, deformed hands, ugly face, distorted eyes, dull colors, photorealistic human skin"
)

DISNEY_VIDEO_PROMPT = (
    "Cinematic Disney 3D camera trajectory: The camera follows a designated smooth curved trajectory line, "
    "beginning from a low-angle frontal close-up of the smiling girl, smoothly sweeping in a dynamic 3D orbital arc around her side, "
    "and rising upward into an elevated high-angle over-the-shoulder perspective. "
    "The glowing magical orb pulses softly with swirling golden starlight particles that drift in the air along the camera curve. "
    "The girl giggles cutely with subtle natural facial micro-expressions, lively eye blinking, and breathing movements. "
    "Rich 3D parallax depth, Pixar movie cinematography, smooth motion trajectory, magical chimes and soft giggle sound effect, masterpiece, 8k"
)

DISNEY_VIDEO_NEGATIVE = (
    "static camera, jerky camera, robotic movement, sudden jump cuts, deformed face, melting features, audio desync, blurry"
)


def load_server_url() -> str:
    for path in ["current_url.txt", "kaggle/all-in-one/current_url.txt", "public_url.txt"]:
        p = Path(path)
        if p.exists():
            url = p.read_text(encoding="utf-8").strip()
            if url.startswith("http"):
                return url
    return "https://sao-carpet-epic-relevance.trycloudflare.com"


def generate_disney_image(server_url: str, out_dir: Path) -> Path:
    print("\n" + "=" * 80)
    print("🎨 [BƯỚC 1/2] QWEN-IMAGE 2.1 (GPU 0): TẠO ẢNH NHÂN VẬT 3D DISNEY/PIXAR")
    print("=" * 80)
    print(f"📝 Prompt: {DISNEY_IMAGE_PROMPT[:120]}...")

    url = f"{server_url.rstrip('/')}/v1/images/generations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-test-key"}
    payload = {
        "prompt": DISNEY_IMAGE_PROMPT,
        "negative_prompt": DISNEY_IMAGE_NEGATIVE,
        "size": "1280x720",
        "width": 1280,
        "height": 720,
        "steps": 20,
        "guidance": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "model": "qwen-image-2.1",
    }

    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
        res = json.loads(response.read().decode("utf-8"))

    elapsed = time.time() - t0
    b64_img = res["data"][0]["b64_json"]
    img_bytes = base64.b64decode(b64_img)

    out_file = out_dir / "disney_witch_image.png"
    out_file.write_bytes(img_bytes)
    print(f"✅ [BƯỚC 1 HOÀN TẤT] Đã lưu ảnh Disney: {out_file} ({len(img_bytes)/1024:.1f} KB, {elapsed:.2f}s)")
    return out_file


def generate_disney_video(server_url: str, image_path: Path, out_dir: Path) -> Path:
    print("\n" + "=" * 80)
    print("🎬 [BƯỚC 2/2] MINIMAX H3 (GPU 1): THỰC THI CAMERA TRAJECTORY THEO ĐƯỜNG LINE CONG (I2VA)")
    print("=" * 80)
    print(f"📝 Prompt: {DISNEY_VIDEO_PROMPT[:120]}...")
    print("🎥 Quỹ đạo: Low-Angle Front -> 3D Orbital Arc -> High-Angle Over-The-Shoulder")

    img_b64 = f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"
    url = f"{server_url.rstrip('/')}/v1/videos/generations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-test-key"}

    payload = {
        "model": "Comfy-Org/MiniMax-H3",
        "image": img_b64,
        "prompt": DISNEY_VIDEO_PROMPT,
        "negative_prompt": DISNEY_VIDEO_NEGATIVE,
        "width": 768,
        "height": 448,
        "num_frames": 33,
        "fps": 24,
        "steps": 16,
        "guidance": 1.6,         # Guidance bám chặt chuyển động camera trajectory
        "shift": 8.0,
        "enable_audio": True,    # Kích hoạt âm thanh ma thuật
    }

    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=600) as response:
        content = response.read().decode("utf-8").strip()
        res = json.loads(content)

    elapsed = time.time() - t0
    b64_vid = res["data"][0]["b64_json"]
    vid_bytes = base64.b64decode(b64_vid)

    out_file = out_dir / "disney_camera_trajectory.mp4"
    out_file.write_bytes(vid_bytes)
    print(f"✅ [BƯỚC 2 HOÀN TẤT] Đã lưu video: {out_file} ({len(vid_bytes)/(1024*1024):.2f} MB, {elapsed:.2f}s)")

    # Trích xuất frames
    frames_dir = out_dir / "disney_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, ts in enumerate([0.0, 0.7, 1.3]):
        f_out = frames_dir / f"disney_frame_{idx+1}_t{ts}s.png"
        subprocess.run(["ffmpeg", "-y", "-ss", str(ts), "-i", str(out_file), "-vframes", "1", "-q:v", "2", str(f_out)], capture_output=True)
        if f_out.exists():
            print(f"   🖼️ Frame {idx+1}: {f_out}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Disney Camera Trajectory Pipeline")
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--outdir", default="benchmark_outputs/disney_showcase", help="Output directory")
    args = parser.parse_args()

    server_url = args.url or load_server_url()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("✨ KHỞI CHẠY THỬ NGHIỆM ĐIỀU KHIỂN CAMERA THEO ĐƯỜNG LINE (DISNEY PIXAR)")
    print(f"🌐 Server URL : {server_url}")
    print(f"📁 Output Dir : {out_dir.resolve()}")
    print("=" * 80)

    t_start = time.time()
    img_file = generate_disney_image(server_url, out_dir)
    vid_file = generate_disney_video(server_url, img_file, out_dir)
    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print(f"🎉 THỬ NGHIỆM HOÀN TẤT TRONG TỔNG THỜI GIAN: {total_time:.2f}s!")
    print(f"📁 Ảnh gốc Disney       : {img_file}")
    print(f"🎬 Video Camera Trajectory: {vid_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
