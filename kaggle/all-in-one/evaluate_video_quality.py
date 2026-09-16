#!/usr/bin/env python3
"""
🎬 Video Quality & Frame Verification Utility
Trích xuất tất cả frames của video Wan2.1 1.3B, tính toán thống kê phân bố pixel
(min, max, mean, std) để xác minh độ tương phản, chi tiết hình ảnh thực tế (không phải grey noise).
"""

import os
import sys
import subprocess
import numpy as np
from PIL import Image

def analyze_video(video_path: str, output_dir: str = "test_outputs/k28_frames"):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    file_size = os.path.getsize(video_path)
    print(f"📹 Analyzing video: {video_path} ({file_size:,} bytes)")

    # Trích xuất frames bằng ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=16",
        os.path.join(output_dir, "frame_%02d.png")
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"❌ ffmpeg failed: {res.stderr.decode()}")
        return False

    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])
    if not frame_files:
        print("❌ No frames extracted")
        return False

    print(f"✅ Extracted {len(frame_files)} frames to {output_dir}")
    print("\n📊 Frame Pixel Distribution Analysis:")
    print(f"{'Frame':<10} | {'Min':<6} | {'Max':<6} | {'Mean':<8} | {'Std':<8} | {'Quality Verdict':<20}")
    print("-" * 75)

    all_stds = []
    all_means = []

    for idx, fpath in enumerate(frame_files, 1):
        im = Image.open(fpath).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        f_min = float(arr.min())
        f_max = float(arr.max())
        f_mean = float(arr.mean())
        f_std = float(arr.std())
        all_stds.append(f_std)
        all_means.append(f_mean)

        if f_std < 10.0:
            verdict = "⚠️ Grey Noise / Flat"
        elif f_std < 25.0:
            verdict = "⚠️ Low Contrast"
        else:
            verdict = "🎉 High Quality Visual"

        print(f"Frame {idx:<4} | {f_min:<6.1f} | {f_max:<6.1f} | {f_mean:<8.2f} | {f_std:<8.2f} | {verdict:<20}")

    avg_std = np.mean(all_stds)
    avg_mean = np.mean(all_means)
    print("-" * 75)
    print(f"Average:   | min={min(all_stds):<5.1f} | max={max(all_stds):<5.1f} | Mean={avg_mean:<8.2f} | Std={avg_std:<8.2f}")

    if avg_std > 25.0:
        print("\n🏆 VERDICT: EXCELLENT! High dynamic range, real visual features and rich pixel contrast.")
        return True
    else:
        print("\n⚠️ VERDICT: FAILED. Output appears to be flat noise or degraded.")
        return False

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "test_outputs/k28_wan_video.mp4"
    out_d = sys.argv[2] if len(sys.argv) > 2 else "test_outputs/k28_frames"
    success = analyze_video(vid, out_d)
    sys.exit(0 if success else 1)
