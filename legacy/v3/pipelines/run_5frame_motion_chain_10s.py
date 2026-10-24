#!/usr/bin/env python3
"""
🎬 Script sinh chuỗi 10 giây chuyển động bay của Rồng (No Voice)
Sử dụng chuẩn 3D Temporal Latent Clip Guide (5 frames: 17*0 + 5):
- Mỗi phân cảnh mới (Shot k+1) nhận đúng 5 frame cuối [N-5 : N] của Shot k làm `guide_frames`.
- ComfyUI nạp 5 frame qua chuỗi ImageBatch [5, H, W, 3] đưa vào MiniMaxH3AddGuide (frame_idx=0)
  để Video VAE mã hóa thành 1 block tensor không-thời gian 3D (latent_t = 2),
  khóa cứng cả vị trí x⃗(t) và vector vận tốc v⃗(t) xuyên suốt frame 0..4.
- Khi ghép video, ffmpeg loại bỏ đúng 5 frame ghim này (select='gte(n,5)')
  để điểm chuyển tiếp có gia tốc chuyển động mượt mà 100%.
"""

import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = BASE_DIR / "benchmark_outputs" / "dragon_10s_5frame_motion"
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
    raise RuntimeError("Chưa kết nối được Kaggle Server. Vui lòng kiểm tra tunnel URL.")


def extract_last_5_frames(video_path: Path, prefix: str) -> list[Path]:
    """Trích xuất chính xác 5 frame liên tiếp cuối cùng của video."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    v_stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    nb_frames = int(v_stream.get("nb_frames") or 39)

    frames = []
    # 5 frame cuối: nb_frames - 5, nb_frames - 4, ..., nb_frames - 1
    for offset in range(5, 0, -1):
        f_idx = max(0, nb_frames - offset)
        out_png = OUT_DIR / f"{prefix}_guide_f{5-offset}.png"
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
            raise RuntimeError(f"Lỗi trích xuất frame {f_idx} từ {video_path}")
        frames.append(out_png)
    return frames


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

    # Shot 1 có sẵn (0.0s - 1.625s, 39 frames)
    shot1_copy = OUT_DIR / "dragon_5f_shot_01.mp4"
    shutil.copy2(SHOT1_VIDEO, shot1_copy)
    print(f"🎬 [Shot 1/5 - Có sẵn] {shot1_copy.name} ({shot1_copy.stat().st_size / 1024:.1f} KB)")

    master_768 = OUT_DIR / "master_dragon_768x448.png"
    prepare_768x448_png(MASTER_IMG, master_768)

    # Kế hoạch 4 shots nối tiếp (mỗi shot 39 frames = 1.625s; sau khi trừ 5 frame ghim còn 34 frames = 1.417s)
    # Tổng thời lượng 5 shots: 39 + 34 + 34 + 34 + 34 = 175 frames -> ~7.3s (hoặc nếu num_frames=56: 39 + 51*4 = 243 frames -> đúng 10.1s!)
    # Sử dụng num_frames=56 để 5 shots đạt chính xác 10.0 giây!
    shots_plan = [
        {
            "idx": 2,
            "num_frames": 56,
            "prompt": (
                "Continuous cinematic aerial shot: The colossal obsidian-black dragon smoothly maintains its right-to-left "
                "horizontal flight trajectory and forward velocity momentum. It finishes breathing fire as dark smoke disperses behind its jaws, "
                "banking powerfully with deep leathery bat-wing beats across the stormy ocean and jagged sea cliffs. "
                "Camera tracks steadily beside the flying dragon maintaining constant velocity."
            ),
            "use_end_anchor": False,
            "seed": 502602,
        },
        {
            "idx": 3,
            "num_frames": 56,
            "prompt": (
                "Continuous cinematic aerial shot: The giant obsidian-black dragon continues flying forward from right to left, "
                "folding its massive leathery wings slightly into an aerodynamic glide along the exact same flight vector. "
                "Slicing smoothly through dark storm clouds as dramatic golden sunbeams break over the coastal mountains below. "
                "Smooth tracking shot with steady camera speed and continuous forward momentum."
            ),
            "use_end_anchor": False,
            "seed": 502603,
        },
        {
            "idx": 4,
            "num_frames": 56,
            "prompt": (
                "Continuous cinematic aerial shot: The colossal black-and-crimson dragon spreads its vast bat-like wings wide, "
                "soaring majestically from right to left across the screen above the ancient stone Dragonstone fortress on the rocky cliff. "
                "Camera glides smoothly along the flight path maintaining continuous horizontal momentum."
            ),
            "use_end_anchor": False,
            "seed": 502604,
        },
        {
            "idx": 5,
            "num_frames": 56,
            "prompt": (
                "Continuous cinematic aerial shot: The colossal obsidian-black dragon approaches the jagged coastal cliff peak from the right, "
                "flaring its massive bat-like leathery wings wide to brake in mid-air before extending its talons to grip the dark crag "
                "and land proudly in its perched stance overlooking the sea."
            ),
            "use_end_anchor": True,
            "seed": 502605,
        },
    ]

    prev_video = shot1_copy
    all_shot_videos = [shot1_copy]

    for shot in shots_plan:
        idx = shot["idx"]
        out_video = OUT_DIR / f"dragon_5f_shot_0{idx}.mp4"
        last_5_pngs = extract_last_5_frames(prev_video, f"shot_0{idx-1}")
        print(f"   🎞️ [5-Frame Guide Clip cho Shot {idx}]: {[p.name for p in last_5_pngs]}")

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
            "guide_frames": [encode_b64_uri(p) for p in last_5_pngs],
            "first_frame": encode_b64_uri(last_5_pngs[-1]),
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

        print(f"🚀 [Shot {idx}/5] Đang sinh bằng 5-Frame 3D Temporal Latent Clip Guide + Turbo LoRA (6 steps, {shot['num_frames']} frames)...")
        t0 = time.time()
        res = post_video_generation(base_url, payload)
        dt = time.time() - t0
        vid_b64 = res["data"][0]["b64_json"]
        out_video.write_bytes(base64.b64decode(vid_b64))
        print(f"✅ [Shot {idx}/5] Hoàn thành trong {dt:.2f}s -> {out_video.name} ({out_video.stat().st_size / 1024:.1f} KB)")

        all_shot_videos.append(out_video)
        prev_video = out_video

    # Ghép nối chuẩn: Shot 1 giữ nguyên 39 frames, các shot 2..5 bỏ đúng 5 frame ghim (select='gte(n,5)')
    print("\n🎞️ Đang ghép nối 5 shots (loại bỏ 5 frame ghim tại mỗi điểm nối)...")
    trimmed_shots = []
    for i, vp in enumerate(all_shot_videos):
        t_out = OUT_DIR / f"trimmed_5f_0{i+1}.mp4"
        if i == 0:
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        else:
            # Bỏ đúng 5 frame ghim của MiniMax H3 3D latent block
            cmd = [
                "ffmpeg", "-y", "-i", str(vp),
                "-vf", "select='gte(n\\,5)',setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-an", str(t_out)
            ]
        subprocess.run(cmd, check=True, capture_output=True)
        trimmed_shots.append(t_out)

    concat_txt = OUT_DIR / "concat_5f_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.name}'" for p in trimmed_shots) + "\n")

    final_video = OUT_DIR / "dragon_10s_smooth_flight_no_voice.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-t", "10.0",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-an",
        str(final_video)
    ], check=True, capture_output=True)

    print(f"🎉 Hoàn tất thước phim 10s mượt mà (5-Frame Motion Guide): {final_video} ({final_video.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
