#!/usr/bin/env python3
"""
🐉 Dragon Cinematic AI Pipeline: Qwen-Image 2.1 (GPU 0) -> MiniMax H3 I2VA (GPU 1)
================================================================================
Quy trình điện ảnh khép kín:
1. Qwen-Image 2.1 (GPU 0, cuda:0, port 8188):
   - Sinh ảnh con rồng đen Drogon-style cực hiện đại, vảy gai góc, mắt dung nham, cánh dơi khổng lồ.
   - Bối cảnh hẻm núi pháo đài đá hoàng hôn hiểm trở, ĐỘ NÉT SÂU TOÀN PHẦN (Deep focus, KHÔNG MỜ BACKGROUND).
2. MiniMax H3 (GPU 1, cuda:1, port 8189):
   - Sinh hoạt ảnh I2VA con rồng bay lượn và phun luồng lửa dung nham rực sáng bầu trời.
   - Focus: Điều hướng góc quay camera (Dynamic Sweeping Arc Tracking Camera).
   - Âm thanh: Tiếng gầm thét sấm sét và tiếng lửa xé gió chân thực.
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


DRAGON_IMAGE_PROMPT = (
    "Cinematic masterpiece of a colossal black dragon inspired by Drogon from Game of Thrones, "
    "rugged obsidian-black and volcanic ember-red scales with sharp dorsal spikes, massive horned head with fierce crimson glowing eyes, "
    "razor-sharp dagger teeth, immense muscular bat-like wings with intricate membrane textures, "
    "perched majestically on a rugged jagged cliff overlooking a dark stormy sea and ancient stone fortress ruins. "
    "Deep depth of field, sharp focus across entire image, highly detailed background, clear sharp distant mountain peaks and storm clouds, "
    "golden-hour fiery dramatic rim lighting, photorealistic 8k CGI, no blur, no bokeh, hyper-detailed textures"
)

DRAGON_IMAGE_NEGATIVE = (
    "blurry background, shallow depth of field, bokeh, out of focus background, low quality, cartoon, plastic skin, deformed wings, bad anatomy"
)

DRAGON_VIDEO_PROMPT = (
    "Dynamic cinematic sweeping arc shot: The massive black dragon launches from the cliff into powerful flight, "
    "soaring through the stormy sky and unleashing a tremendous torrent of roaring blazing fire from its jaws. "
    "The camera smoothly pans and tracks around the dragon in a 3D arc motion, shifting perspective dynamically to follow the flight path and fiery breath. "
    "Realistic wing aerodynamic physics, fluid fire simulation illuminating the dragon's scales and dark clouds, "
    "thunderous dragon roar and roaring fire breath sound, epic cinematic blockbuster camera movement, masterpiece, 8k"
)

DRAGON_VIDEO_NEGATIVE = (
    "static camera, jerky movement, robotic motion, blurry, low resolution, deformed anatomy, flickering, audio desync"
)


def load_server_url() -> str:
    for path in ["current_url.txt", "kaggle/all-in-one/current_url.txt", "public_url.txt"]:
        p = Path(path)
        if p.exists():
            url = p.read_text(encoding="utf-8").strip()
            if url.startswith("http"):
                return url
    return "http://127.0.0.1:8000"


def wait_for_server(server_url: str, max_wait: int = 180) -> bool:
    print(f"⏳ Đang kiểm tra kết nối tới server {server_url}...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            with urllib.request.urlopen(f"{server_url.rstrip('/')}/health", timeout=3) as res:
                if res.status == 200:
                    print(f"✅ Server online sau {time.time() - t0:.1f}s!")
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def generate_dragon_image(server_url: str, out_dir: Path) -> Path:
    print("\n" + "=" * 80)
    print("🖼️ [BƯỚC 1/2] QWEN-IMAGE 2.1 (GPU 0): TẠO ẢNH RỒNG GAME OF THRONES CỰC NÉT")
    print("=" * 80)
    print(f"📝 Prompt: {DRAGON_IMAGE_PROMPT[:120]}...")
    print("🎯 Yêu cầu: Deep Depth of Field, Background hoàn toàn sắc nét (No blur)")

    url = f"{server_url.rstrip('/')}/v1/images/generations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-test-key"}
    payload = {
        "prompt": DRAGON_IMAGE_PROMPT,
        "negative_prompt": DRAGON_IMAGE_NEGATIVE,
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

    out_file = out_dir / "qwen_game_of_thrones_dragon.png"
    out_file.write_bytes(img_bytes)
    print(f"✅ [BƯỚC 1 HOÀN TẤT] Đã lưu ảnh rồng: {out_file} ({len(img_bytes)/1024:.1f} KB, {elapsed:.2f}s)")
    return out_file


def generate_dragon_video(server_url: str, image_path: Path, out_dir: Path) -> Path:
    print("\n" + "=" * 80)
    print("🎬 [BƯỚC 2/2] MINIMAX H3 (GPU 1): TẠO HOẠT ẢNH RỒNG PHUN LỬA & ĐIỀU HƯỚNG GÓC QUAY (I2VA)")
    print("=" * 80)
    print(f"📝 Prompt: {DRAGON_VIDEO_PROMPT[:120]}...")
    print("🎥 Trọng tâm: Dynamic Sweeping Arc Camera Movement + Fire Breath Physics + Epic Roar Audio")

    img_b64 = f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"
    url = f"{server_url.rstrip('/')}/v1/videos/generations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-test-key"}

    payload = {
        "model": "Comfy-Org/MiniMax-H3",
        "image": img_b64,
        "prompt": DRAGON_VIDEO_PROMPT,
        "negative_prompt": DRAGON_VIDEO_NEGATIVE,
        "width": 768,
        "height": 448,
        "num_frames": 33,
        "fps": 24,
        "steps": 16,            # 16 steps đủ độ sắc nét và mượt mà cho hiệu ứng lửa
        "guidance": 1.5,         # Guidance vừa vặn kích hoạt motion bám prompt
        "shift": 8.0,
        "enable_audio": True,    # Kích hoạt âm thanh gầm và lửa rít
    }

    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=600) as response:
        # Bỏ qua các ký tự whitespace keepalive nếu có
        content = response.read().decode("utf-8").strip()
        res = json.loads(content)

    elapsed = time.time() - t0
    b64_vid = res["data"][0]["b64_json"]
    vid_bytes = base64.b64decode(b64_vid)

    out_file = out_dir / "minimax_h3_dragon_fire_breath.mp4"
    out_file.write_bytes(vid_bytes)
    print(f"✅ [BƯỚC 2 HOÀN TẤT] Đã lưu video: {out_file} ({len(vid_bytes)/(1024*1024):.2f} MB, {elapsed:.2f}s)")

    # Trích xuất 3 frames đại diện
    frames_dir = out_dir / "dragon_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, ts in enumerate([0.0, 0.7, 1.3]):
        f_out = frames_dir / f"dragon_frame_{idx+1}_t{ts}s.png"
        subprocess.run(["ffmpeg", "-y", "-ss", str(ts), "-i", str(out_file), "-vframes", "1", "-q:v", "2", str(f_out)], capture_output=True)
        if f_out.exists():
            print(f"   🖼️ Frame {idx+1}: {f_out}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Dragon Pipeline")
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--outdir", default="benchmark_outputs/dragon_showcase", help="Output directory")
    args = parser.parse_args()

    server_url = args.url or load_server_url()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wait_for_server(server_url):
        print(f"❌ Không thể kết nối tới server tại {server_url}")
        sys.exit(1)

    t_start = time.time()
    img_file = generate_dragon_image(server_url, out_dir)
    vid_file = generate_dragon_video(server_url, img_file, out_dir)
    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE THÀNH CÔNG RỰC RỠ TRONG TỔNG THỜI GIAN: {total_time:.2f}s!")
    print(f"📁 Ảnh rồng sắc nét  : {img_file}")
    print(f"🎬 Video rồng phun lửa: {vid_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
