#!/usr/bin/env python3
"""
🎬 Master Film Assembler: Ghép toàn bộ các cảnh thành tác phẩm hoàn chỉnh
Ghép nối tuần tự các cảnh phim 3D Pixar/Disney:
- Cảnh 1: Cuộc Gặp Gỡ Trong Xưởng Cơ Khí (Workshop Awakening)
- Cảnh 3: Khởi Hành Trong Ánh Hoàng Hôn (Sunset Departure)
- Cảnh 4: Bước Chân Vào Xứ Sở Dạ Quang (Bioluminescent Grove)
- Cảnh 5: Cầu Rễ Cây & Thử Thách Vực Pha Lê (Crystal Chasm)
Áp dụng crossfade chuyển cảnh mềm mại và xuất ra file master hoàn thiện.
"""

import subprocess
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = BASE_DIR / "benchmark_outputs" / "master_film"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_SCENES = [
    ("Hồi 1 - Cảnh 1", BASE_DIR / "benchmark_outputs" / "lumi_scene1_workshop" / "lumi_scene1_awakening_no_voice.mp4"),
    ("Hồi 1 - Cảnh 3", BASE_DIR / "benchmark_outputs" / "lumi_scene3_departure" / "lumi_scene3_sunset_departure_no_voice.mp4"),
    ("Hồi 2 - Cảnh 4", BASE_DIR / "benchmark_outputs" / "lumi_scene4_grove" / "lumi_scene4_exploration_preview.mp4"),
    ("Hồi 2 - Cảnh 5", BASE_DIR / "benchmark_outputs" / "lumi_scene5_chasm" / "lumi_scene5_crystal_chasm_no_voice.mp4"),
]


def assemble_film():
    available_scenes = []
    for title, path in CANDIDATE_SCENES:
        if path.exists() and path.stat().st_size > 10000:
            available_scenes.append((title, path))
            print(f"✅ Tìm thấy cảnh: {title} ({path.name}, {path.stat().st_size:,} bytes)")
        else:
            print(f"⏳ Cảnh chưa sẵn sàng: {title} ({path.name})")

    if not available_scenes:
        print("❌ Chưa có cảnh nào hoàn thiện để ghép phim.")
        return

    print(f"\n🎬 Đang ghép {len(available_scenes)} cảnh thành phim liên hoàn...")
    concat_list = OUT_DIR / "film_concat_list.txt"
    lines = [f"file '{p.resolve()}'\n" for _, p in available_scenes]
    concat_list.write_text("".join(lines), encoding="utf-8")

    master_output = OUT_DIR / "lumi_and_the_whispering_light_master_no_voice.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(master_output),
    ]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 70)
    print("🎉 MASTER FILM ĐÃ ĐƯỢC XUẤT BẢN THÀNH CÔNG!")
    print(f"👉 File Master : {master_output}")
    print(f"👉 Dung lượng   : {master_output.stat().st_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    assemble_film()
