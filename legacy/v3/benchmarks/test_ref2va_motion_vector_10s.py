#!/usr/bin/env python3
"""
Script kiểm thử & tạo thước phim 10 giây (no voice) về chuyển động bay của Rồng bằng:
  🔥 MiniMax H3 Ref2VA + MiniMaxH3AddGuide + Turbo LoRA (6 steps)
  🔥 Multi-Frame Motion Vector Conditioning:
     Trích xuất chuỗi 3 frame liên tiếp ở cuối mỗi video trước (t - 0.45s, t - 0.22s, t = 0.0s)
     làm [<Picture 1>, <Picture 2>, <Picture 3>] cho MiniMaxH3ReferenceToVideo,
     ĐỒNG THỜI neo cứng frame cuối (t = 0.0s) vào frame_idx=0 qua MiniMaxH3AddGuide!
     -> Giúp Qwen3-VL 32B và Ref2VA DiT nắm bắt chính xác vector vận tốc (hướng bay, tốc độ camera, pha đập cánh).
"""

import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = BASE_DIR / "benchmark_outputs" / "dragon_10s_ref2va_motion"
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


def extract_motion_tail_frames(video_path: Path, prefix: str) -> tuple[Path, Path, Path]:
    """
    Trích xuất chính xác 3 frame theo chỉ số frame ở cuối video trước:
      - f1 (frame N-9): ~0.38s trước khi kết thúc (<Picture 1>)
      - f2 (frame N-5): ~0.21s trước khi kết thúc (<Picture 2>)
      - f3 (frame N-1): frame cuối cùng (<Picture 3> + first_frame guide ở frame_idx=0)
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    v_stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    nb_frames = int(v_stream.get("nb_frames") or 39)

    indices = [
        ("t_minus_08", max(0, nb_frames - 9)),
        ("t_minus_04", max(0, nb_frames - 5)),
        ("t_last", max(0, nb_frames - 1)),
    ]
    extracted = []
    for tag, f_idx in indices:
        out_png = OUT_DIR / f"{prefix}_{tag}.png"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select='eq(n\\,{f_idx})'",
            "-frames:v", "1",
            "-update", "1",
            str(out_png),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if not out_png.exists() or out_png.stat().st_size == 0:
            raise RuntimeError(f"Failed to extract {tag} (frame {f_idx}/{nb_frames}) from {video_path}")
        extracted.append(out_png)
    return extracted[0], extracted[1], extracted[2]


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


def encode_b64_uri(png_path: Path) -> str:
    b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
    shot1_copy = OUT_DIR / "dragon_ref_shot_01.mp4"
    shutil.copy2(SHOT1_VIDEO, shot1_copy)
    print(f"🎬 [Shot 1/5 - Có sẵn] {shot1_copy.name} ({shot1_copy.stat().st_size / 1024:.1f} KB)")

    # Chuẩn bị ảnh gốc 768x448 làm điểm neo kết thúc (last_frame của Shot 5)
    master_768 = OUT_DIR / "master_dragon_768x448.png"
    prepare_768x448_png(MASTER_IMG, master_768)

    shots_plan = [
        {
            "idx": 2,
            "num_frames": 39,
            "prompt": (
                "Continuing the exact forward-left flight velocity vector and wingbeat rhythm shown sequentially from <Picture 1> to <Picture 2> to <Picture 3>: "
                "the colossal obsidian-black and crimson-scaled dragon stops breathing fire as dark smoke trails behind its jaws, "
                "banking smoothly along the same leftward flight path high above the jagged coastal cliffs and stormy ocean waves. "
                "Continuous aerial tracking camera preserving the exact camera pan speed and dragon anatomy."
            ),
            "use_end_anchor": False,
            "seed": 302602,
        },
        {
            "idx": 3,
            "num_frames": 39,
            "prompt": (
                "Continuing the exact flight trajectory, banking angle, and momentum from <Picture 1> through <Picture 2> to <Picture 3>: "
                "the giant obsidian-black dragon folds its leathery bat-wings slightly into a fast aerodynamic glide along the same motion vector, "
                "slicing through dramatic dark storm clouds with golden sunbeams breaking over the rugged coastline below. "
                "Seamless FPV aerial tracking shot maintaining continuous velocity."
            ),
            "use_end_anchor": False,
            "seed": 302603,
        },
        {
            "idx": 4,
            "num_frames": 39,
            "prompt": (
                "Continuing the exact aerodynamic glide vector and wing motion from <Picture 1> through <Picture 2> to <Picture 3>: "
                "the colossal black-and-crimson dragon spreads its vast bat-like leathery wings wide to catch the thermal updraft, "
                "soaring majestically in a continuous sweeping arc above the ancient stone medieval fortress on the sea cliffs. "
                "Smooth orbital tracking camera maintaining the motion flow."
            ),
            "use_end_anchor": False,
            "seed": 302604,
        },
        {
            "idx": 5,
            "num_frames": 39,
            "prompt": (
                "Continuing the exact approach trajectory and wingbeat momentum from <Picture 1> through <Picture 2> to <Picture 3>: "
                "the colossal obsidian-black dragon descends smoothly toward the jagged rocky cliff peak overlooking the coastal castle, "
                "flaring its massive leathery wings wide to decelerate and perch majestically on the dark crag."
            ),
            "use_end_anchor": True,
            "seed": 302605,
        },
    ]

    prev_video = shot1_copy
    all_shot_videos = [shot1_copy]

    for shot in shots_plan:
        idx = shot["idx"]
        out_video = OUT_DIR / f"dragon_ref_shot_0{idx}.mp4"
        f_minus_10, f_minus_5, f_last = extract_motion_tail_frames(prev_video, f"shot_0{idx-1}")
        print(
            f"   🎞️ [Motion Vector Input cho Shot {idx}] "
            f"<Picture 1>={f_minus_10.name}, <Picture 2>={f_minus_5.name}, <Picture 3>+Guide0={f_last.name}"
        )

        if out_video.exists() and out_video.stat().st_size > 100_000:
            print(f"⏩ [Shot {idx}/5] Đã tồn tại {out_video.name}, bỏ qua bước sinh lại.")
            all_shot_videos.append(out_video)
            prev_video = out_video
            continue

        payload = {
            "model": "Comfy-Org/MiniMax-H3-Ref2VA-Turbo",
            "mode": "ref2va",
            "turbo": True,
            "prompt": shot["prompt"],
            "reference_images": [
                encode_b64_uri(f_minus_10),
                encode_b64_uri(f_minus_5),
                encode_b64_uri(f_last),
            ],
            "first_frame": encode_b64_uri(f_last),
            "width": 768,
            "height": 448,
            "num_frames": shot["num_frames"],
            "fps": 24,
            "steps": 6,
            "guidance": 1.0,
            "seed": shot["seed"],
            "enable_audio": False,
        }
        if shot["use_end_anchor"]:
            payload["last_frame"] = encode_b64_uri(master_768)

        print(
            f"🚀 [Shot {idx}/5] Đang sinh bằng Ref2VA (3-Frame Motion Vector) + MiniMaxH3AddGuide + Turbo LoRA "
            f"(6 steps, {shot['num_frames']} frames, end_anchor={shot['use_end_anchor']})..."
        )
        t0 = time.time()
        res = post_video_generation(base_url, payload)
        dt = time.time() - t0
        vid_b64 = res["data"][0]["b64_json"]
        out_video.write_bytes(base64.b64decode(vid_b64))
        print(f"✅ [Shot {idx}/5] Hoàn thành trong {dt:.2f}s -> {out_video.name} ({out_video.stat().st_size / 1024:.1f} KB)")

        all_shot_videos.append(out_video)
        prev_video = out_video

    # Thêm Shot 6 (39 frames) để tổng thời lượng sau khi bỏ frame trùng đạt đúng 10.0s (6 shots * 1.625s = 9.75s -> 10.0s)
    shot6_video = OUT_DIR / "dragon_ref_shot_06.mp4"
    if not (shot6_video.exists() and shot6_video.stat().st_size > 100_000):
        f_minus_10, f_minus_5, f_last = extract_motion_tail_frames(prev_video, "shot_05")
        payload_s6 = {
            "model": "Comfy-Org/MiniMax-H3-Ref2VA-Turbo",
            "mode": "ref2va",
            "turbo": True,
            "prompt": (
                "Continuing the exact perched stance and wing movement from <Picture 1> through <Picture 2> to <Picture 3>: "
                "the colossal obsidian-black dragon grips the coastal cliff edge, folding its massive leathery wings slightly "
                "and roaring silently with glowing red eyes as storm clouds roll over the medieval castle behind it."
            ),
            "reference_images": [
                encode_b64_uri(f_minus_10),
                encode_b64_uri(f_minus_5),
                encode_b64_uri(f_last),
            ],
            "first_frame": encode_b64_uri(f_last),
            "width": 768,
            "height": 448,
            "num_frames": 39,
            "fps": 24,
            "steps": 6,
            "guidance": 1.0,
            "seed": 302606,
            "enable_audio": False,
        }
        print("🚀 [Shot 6/6] Đang sinh bằng Ref2VA (3-Frame Motion Vector) + MiniMaxH3AddGuide + Turbo LoRA (6 steps, 39 frames)...")
        t0 = time.time()
        res = post_video_generation(base_url, payload_s6)
        dt = time.time() - t0
        shot6_video.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
        print(f"✅ [Shot 6/6] Hoàn thành trong {dt:.2f}s -> {shot6_video.name} ({shot6_video.stat().st_size / 1024:.1f} KB)")
    all_shot_videos.append(shot6_video)

    # Ghép nối thành thước phim 10.0s liền mạch (bỏ frame 0 trùng ở các shot nối tiếp)
    print("\n🎞️ Đang ghép nối các shots Ref2VA Motion-Vector thành thước phim 10.0 giây liền mạch (24fps, 768x448)...")
    trimmed_shots = []
    for i, vp in enumerate(all_shot_videos):
        t_out = OUT_DIR / f"trimmed_ref_0{i+1}.mp4"
        if i == 0:
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-vf", "select='gte(n\\,1)',setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        subprocess.run(cmd, check=True, capture_output=True)
        trimmed_shots.append(t_out)

    concat_txt = OUT_DIR / "concat_ref_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.name}'" for p in trimmed_shots) + "\n")

    final_ref_video = OUT_DIR / "dragon_10s_ref2va_motion_vector.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-an",
        str(final_ref_video)
    ], check=True, capture_output=True)

    print(f"🎉 Hoàn tất thước phim Ref2VA Motion-Vector (No Voice): {final_ref_video} ({final_ref_video.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
