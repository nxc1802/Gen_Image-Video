#!/usr/bin/env python3
"""
Script tạo thước phim 10 giây (no voice) liền mạch về chuyển động bay của Rồng (Game of Thrones style):
- Tận dụng:
  1. Ảnh gốc Master: benchmark_outputs/dragon_showcase/qwen_game_of_thrones_dragon.png
  2. Shot 1 có sẵn (0.0s - 1.62s): benchmark_outputs/dragon_showcase/minimax_h3_dragon_fire_breath.mp4
  3. Model FL2VA + Turbo LoRA (6 steps) đang warm sẵn trên GPU 1 của Kaggle Server:
     - Shot 2 (1.62s - 3.75s): Nối từ last_frame của Shot 1 -> Rồng ngừng phun lửa, vỗ đôi cánh da khổng lồ nghiêng mình lượn qua vách đá.
     - Shot 3 (3.75s - 5.88s): Nối từ last_frame của Shot 2 -> Rồng khép nhẹ cánh lao vút (aerodynamic dive) xuyên qua tầng mây bão và tia nắng chiếu xuống biển.
     - Shot 4 (5.88s - 8.00s): Nối từ last_frame của Shot 3 -> Rồng sải rộng đôi cánh dơi khổng lồ lượn vòng uy nghiêm trên tòa lâu đài cổ Dragonstone.
     - Shot 5 (8.00s - 10.00s): Sử dụng tính năng kép FL2VA (first_frame = last_frame của Shot 4, last_frame = qwen_game_of_thrones_dragon.png 768x448) -> Rồng hạ độ cao, vỗ cánh hãm đà và đáp xuống mỏm đá bờ biển khép kín vòng lặp 10 giây hoàn hảo!
"""

import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = BASE_DIR / "benchmark_outputs" / "dragon_10s_film"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_IMG = BASE_DIR / "benchmark_outputs" / "dragon_showcase" / "qwen_game_of_thrones_dragon.png"
SHOT1_VIDEO = BASE_DIR / "benchmark_outputs" / "dragon_showcase" / "minimax_h3_dragon_fire_breath.mp4"


def get_server_url() -> str:
    url_file = BASE_DIR / "current_url.txt"
    if url_file.exists():
        u = url_file.read_text().strip().rstrip("/")
        if u:
            try:
                with urllib.request.urlopen(f"{u}/health", timeout=10) as r:
                    if r.status == 200:
                        return u
            except Exception:
                pass
    raise RuntimeError("Không kết nối được Kaggle Server trong current_url.txt")


def extract_last_frame(video_path: Path, out_png: Path) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.08",
        "-i", str(video_path),
        "-frames:v", "1",
        "-update", "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not out_png.exists() or out_png.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract last frame from {video_path}")
    return out_png


def prepare_768x448_png(src_png: Path, out_png: Path) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_png),
        "-vf", "scale=768:448:force_original_aspect_ratio=increase,crop=768:448",
        "-frames:v", "1",
        "-update", "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_png


def encode_b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode("utf-8")


def post_video_generation(base_url: str, payload: dict, timeout: int = 1200) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/videos/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    base_url = get_server_url()
    print(f"✅ Kết nối Kaggle Server: {base_url}")

    # Copy Shot 1 (0.0s - 1.625s) có sẵn
    shot1_copy = OUT_DIR / "dragon_shot_01.mp4"
    shutil.copy2(SHOT1_VIDEO, shot1_copy)
    print(f"🎬 [Shot 1/5 - Có sẵn] {shot1_copy.name} ({shot1_copy.stat().st_size / 1024:.1f} KB)")

    # Chuẩn bị ảnh gốc 768x448 làm điểm neo kết thúc (last_frame của Shot 5)
    master_768 = OUT_DIR / "master_dragon_768x448.png"
    prepare_768x448_png(MASTER_IMG, master_768)

    shots_plan = [
        {
            "idx": 2,
            "prompt": (
                "The colossal obsidian-black and crimson-scaled Game of Thrones dragon finishes breathing fire "
                "and banks powerfully in mid-air with deep, rhythmic strokes of its massive bat-like leathery wings. "
                "Dynamic aerial tracking camera glides smoothly beside the flying dragon high above the sharp rocky coastal cliffs "
                "and dark churning ocean waves under dramatic storm clouds. Ultra-sharp background, photorealistic cinematic VFX. "
                "Pure natural sound of heavy leathery wingbeats and high-altitude rushing wind only, no voice, no speech."
            ),
            "use_end_anchor": False,
            "seed": 202602,
        },
        {
            "idx": 3,
            "prompt": (
                "The giant obsidian-black dragon folds its leathery bat-wings slightly into a fast aerodynamic swoop, "
                "gliding and banking gracefully through dramatic dark storm clouds with golden sunbeams breaking over the rugged coastline below. "
                "Fast cinematic FPV aerial camera tracks closely along the dragon's flight trajectory, showcasing rippling wing membranes "
                "and razor-sharp dorsal spines. Pure natural sound of howling wind and powerful wing whooshes only, no voice, no speech."
            ),
            "use_end_anchor": False,
            "seed": 202603,
        },
        {
            "idx": 4,
            "prompt": (
                "Pulling up from its dive, the colossal black-and-crimson dragon spreads its vast bat-like wings wide to catch the updraft, "
                "soaring majestically in a sweeping arc above the ancient stone medieval fortress perched on the jagged sea cliffs below. "
                "Smooth orbital tracking camera circles around the soaring dragon against the sharp coastal mountain backdrop. "
                "Pure natural sound of soaring wind, flapping leathery wings, and distant ocean waves only, no voice, no speech."
            ),
            "use_end_anchor": False,
            "seed": 202604,
        },
        {
            "idx": 5,
            "prompt": (
                "The colossal obsidian-black dragon descends toward the jagged rocky cliff peak overlooking the castle and sea, "
                "flaring its massive bat-like leathery wings wide to brake in mid-air before gripping the dark coastal crag with its talons, "
                "wings spread majestically under the stormy sky. Smooth descending tracking camera. "
                "Pure natural sound of heavy wingbeats, talons gripping rock, and coastal wind only, no voice, no speech."
            ),
            "use_end_anchor": True,
            "seed": 202605,
        },
    ]

    neg_prompt = (
        "human voice, speech, talking, narration, singing, vocals, dialogue, words, music, "
        "blurry background, out of focus, deformed dragon, extra wings, extra limbs, morphing, low quality"
    )

    prev_video = shot1_copy
    all_shot_videos = [shot1_copy]

    for shot in shots_plan:
        idx = shot["idx"]
        out_video = OUT_DIR / f"dragon_shot_0{idx}.mp4"
        last_frame_png = OUT_DIR / f"shot_0{idx-1}_last_frame.png"
        extract_last_frame(prev_video, last_frame_png)

        if out_video.exists() and out_video.stat().st_size > 100_000:
            print(f"⏩ [Shot {idx}/5] Đã tồn tại {out_video.name}, bỏ qua bước sinh lại.")
            all_shot_videos.append(out_video)
            prev_video = out_video
            continue

        payload = {
            "model": "Comfy-Org/MiniMax-H3-Turbo",
            "mode": "fl2va",
            "turbo": True,
            "prompt": shot["prompt"],
            "first_frame": f"data:image/png;base64,{encode_b64(last_frame_png)}",
            "width": 768,
            "height": 448,
            "num_frames": 49,
            "fps": 24,
            "steps": 6,
            "cfg_scale": 1.0,
            "seed": shot["seed"],
            "enable_audio": False,
        }
        if shot["use_end_anchor"]:
            payload["last_frame"] = f"data:image/png;base64,{encode_b64(master_768)}"

        print(f"🚀 [Shot {idx}/5] Đang sinh bằng FL2VA + Turbo LoRA (6 steps, cfg=1.0, 49 frames, end_anchor={shot['use_end_anchor']})...")
        t0 = time.time()
        res = post_video_generation(base_url, payload)
        dt = time.time() - t0
        vid_b64 = res["data"][0]["b64_json"]
        out_video.write_bytes(base64.b64decode(vid_b64))
        print(f"✅ [Shot {idx}/5] Hoàn thành trong {dt:.2f}s -> {out_video.name} ({out_video.stat().st_size / 1024:.1f} KB)")

        all_shot_videos.append(out_video)
        prev_video = out_video

    # Ghép nối 5 shots thành thước phim 10.0s liền mạch (loại bỏ 1 frame trùng ở đầu các shot 2..5)
    print("\n🎞️ Đang ghép nối 5 shots thành thước phim 10.0 giây liền mạch (24fps, 768x448)...")
    trimmed_shots = []
    for i, vp in enumerate(all_shot_videos):
        t_out = OUT_DIR / f"trimmed_0{i+1}.mp4"
        if i == 0:
            # Giữ nguyên toàn bộ frame của Shot 1
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        else:
            # Bỏ frame đầu tiên (frame 0 trùng với last_frame của shot trước)
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-vf", "select='gte(n\\,1)',setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        subprocess.run(cmd, check=True, capture_output=True)
        trimmed_shots.append(t_out)

    concat_txt = OUT_DIR / "concat_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.name}'" for p in trimmed_shots) + "\n")

    # Bản No-Voice chuẩn xác 10.0s
    final_no_voice = OUT_DIR / "dragon_10s_flight_no_voice.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-t", "10.0",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-an",
        str(final_no_voice)
    ], check=True, capture_output=True)

    print(f"🎉 Hoàn tất thước phim 10s (No Voice): {final_no_voice} ({final_no_voice.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
