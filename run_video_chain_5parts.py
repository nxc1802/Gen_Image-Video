#!/usr/bin/env python3
"""
🎬 5-Part Video Extension & Chaining Pipeline (Disney Pixar Witch Story)
================================================================================
Tạo chuỗi 5 video liên tiếp liên hoàn (Continuous Video Chaining via Last-Frame):
- Part 1: Đã có sẵn (Quả cầu bay lên, camera xoay 180 độ ra sau vai).
- Part 2: Quả cầu chạm trần vỡ òa thành dải ngân hà bụi sao lấp lánh, cô bé reo vui.
- Part 3: Sách phép thuật trên giá sách tự mở cánh bay lượn như đàn chim quang học.
- Part 4: Một cuốn sách vàng đáp xuống tay cô bé, mở ra bản đồ sao 3D lơ lửng.
- Part 5: Cô bé ôm cuốn sách nhảy cẫng lên hạnh phúc, camera zoom out ra cửa sổ ngắm bầu trời đêm.
- Ghép nối: Dùng ffmpeg nối cả 5 phần thành 1 video dài ~8.2 giây mượt mà!
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


CHAIN_PROMPTS = [
    # Part 1: Đã có sẵn
    None,

    # Part 2
    (
        "Cinematic Disney 3D animation: The glowing orb reaches the ceiling and bursts into a radiant galaxy of swirling golden starlight particles, "
        "illuminating the magical wooden library. The little witch girl looks up with awe and delightful joy, waving her hands as the starlight dances around her. "
        "Whimsical starlight chime sounds, fluid 3D motion, Pixar blockbuster masterpiece, 8k"
    ),

    # Part 3
    (
        "Cinematic Disney 3D animation: Ancient spellbooks on the tall shelves sprout glowing golden wings of light and flutter gracefully off the bookshelves like magical birds, "
        "circling around the excited little girl. The camera tracks dynamically following a glowing spellbook gliding through the room. "
        "Whimsical flapping wings and magical chime audio, Disney Pixar 3D render, masterpiece"
    ),

    # Part 4
    (
        "Cinematic Disney 3D animation: The golden glowing spellbook descends gently and lands softly into the little girl's outstretched hands, "
        "instantly projecting a floating 3D luminous holographic star map in front of her. She gasps in wonder with wide sparkling green eyes. "
        "Magical hum and sparkle chime sounds, rich Pixar textures, masterpiece"
    ),

    # Part 5
    (
        "Cinematic Disney 3D animation: The little witch girl happily hugs the glowing spellbook to her chest, jumping up with a cheerful cute giggle. "
        "The camera smoothly pulls back in a wide cinematic crane retreat shot through the warm arched library window into the starry fairytale night sky. "
        "Joyful giggle and triumphant orchestral chime, Pixar masterpiece ending, 8k"
    ),
]


def load_server_url() -> str:
    for path in ["current_url.txt", "kaggle/all-in-one/current_url.txt", "public_url.txt"]:
        p = Path(path)
        if p.exists():
            url = p.read_text(encoding="utf-8").strip()
            if url.startswith("http"):
                return url
    return ""


def extract_last_frame(video_path: Path, output_frame: Path) -> Path:
    """Trích xuất chính xác frame cuối cùng của video."""
    cmd = [
        "ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video_path),
        "-vframes", "1", "-q:v", "1", str(output_frame)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_frame


def generate_extended_video(server_url: str, start_frame: Path, prompt: str, part_num: int, out_dir: Path) -> Path:
    print(f"\n🎬 [TẠO PHẦN {part_num}/5] Gửi last frame tới MiniMax H3...")
    print(f"📝 Prompt: {prompt[:110]}...")

    img_b64 = f"data:image/png;base64,{base64.b64encode(start_frame.read_bytes()).decode('utf-8')}"
    url = f"{server_url.rstrip('/')}/v1/videos/generations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-test-key"}

    payload = {
        "model": "Comfy-Org/MiniMax-H3",
        "image": img_b64,
        "prompt": prompt,
        "negative_prompt": "static, blurry, jerky camera, distorted face, bad anatomy, audio desync, robotic motion",
        "width": 768,
        "height": 448,
        "num_frames": 33,
        "fps": 24,
        "steps": 15,
        "guidance": 1.5,
        "shift": 8.0,
        "enable_audio": True,
    }

    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=900) as response:
        content = response.read().decode("utf-8").strip()
        res = json.loads(content)

    elapsed = time.time() - t0
    b64_vid = res["data"][0]["b64_json"]
    vid_bytes = base64.b64decode(b64_vid)

    out_file = out_dir / f"disney_chain_part_{part_num}.mp4"
    out_file.write_bytes(vid_bytes)
    print(f"✅ [PHẦN {part_num} HOÀN TẤT] Đã lưu: {out_file} ({len(vid_bytes)/(1024*1024):.2f} MB, {elapsed:.2f}s)")
    return out_file


def concatenate_videos(video_list: list, output_video: Path) -> Path:
    """Ghép nối danh sách các video MP4 thành 1 video duy nhất mượt mà."""
    print("\n" + "=" * 80)
    print("🎞️ ĐANG GHÉP NỐI TOÀN BỘ 5 PHẦN THÀNH 1 SIÊU PHẨM VIDEO DÀI...")
    print("=" * 80)

    concat_file = output_video.parent / "concat_list.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for v in video_list:
            f.write(f"file '{v.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", str(output_video)
    ]
    subprocess.run(cmd, check=True)
    concat_file.unlink(missing_ok=True)
    print(f"🎉 GHÉP NỐI THÀNH CÔNG! File hoàn chỉnh: {output_video}")
    return output_video


def main():
    parser = argparse.ArgumentParser(description="5-Part Video Extension Pipeline")
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--outdir", default="benchmark_outputs/disney_showcase/chain_5parts", help="Output directory")
    args = parser.parse_args()

    server_url = args.url or load_server_url()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phần 1: Tận dụng video Part 1 đã tạo trước đó
    src_part1 = Path("benchmark_outputs/disney_showcase/disney_camera_trajectory.mp4")
    part1_file = out_dir / "disney_chain_part_1.mp4"
    if src_part1.exists() and not part1_file.exists():
        part1_file.write_bytes(src_part1.read_bytes())
    
    video_chain = [part1_file]
    print("=" * 80)
    print("🎬 KHỞI ĐỘNG QUY TRÌNH CHUỖI 5 VIDEO LIÊN TIẾP (VIDEO CHAINING & EXTENSION)")
    print(f"📁 Thư mục xuất : {out_dir.resolve()}")
    print(f"🌐 Server URL   : {server_url}")
    print(f"✅ Part 1       : {part1_file} (Đã sẵn sàng)")
    print("=" * 80)

    # 2. Tuần tự sinh Part 2, 3, 4, 5
    for part_idx in range(2, 6):
        prev_video = video_chain[-1]
        last_frame_path = out_dir / f"last_frame_for_part_{part_idx}.png"
        extract_last_frame(prev_video, last_frame_path)
        print(f"🖼️ Đã trích xuất last frame từ Part {part_idx-1}: {last_frame_path}")

        prompt = CHAIN_PROMPTS[part_idx - 1]
        new_video = generate_extended_video(server_url, last_frame_path, prompt, part_idx, out_dir)
        video_chain.append(new_video)

    # 3. Ghép nối thành video dài
    final_extended = out_dir / "disney_extended_story_complete_5parts.mp4"
    concatenate_videos(video_chain, final_extended)

    print("\n" + "=" * 80)
    print("🏆 TOÀN BỘ CHUỖI 5 VIDEO ĐÃ HOÀN TẤT VÀ GHÉP NỐI THÀNH CÔNG!")
    print(f"🎥 Video hoàn chỉnh: {final_extended}")
    print("=" * 80)


if __name__ == "__main__":
    main()
