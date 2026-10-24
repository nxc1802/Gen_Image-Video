#!/usr/bin/env python3
"""
🎬 Script Render Hồi 1 - Cảnh 1: Cuộc Gặp Gỡ Giữa Lumi & Pip (Disney/Pixar Style)
Sử dụng mô hình MiniMax H3 Ref2VA + Turbo LoRA (6 steps).
Nạp trực tiếp bộ ảnh LookDev đã được duyệt:
- Reference Image 1: 01_lumi_character_sheet.jpg (Khóa nhận diện khuôn mặt, trang phục Lumi)
- Reference Image 2: 03_pip_creature_sheet.jpg (Khóa nhận diện sinh vật ánh sáng Pip)
- First Frame: 04_lumi_and_pip_bonding.jpg (Start Keyframe tại xưởng cơ khí)
"""

import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
LOOKDEV_DIR = BASE_DIR / "benchmark_outputs" / "lookdev_lumi_disney"
OUT_DIR = BASE_DIR / "benchmark_outputs" / "lumi_scene1_workshop"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_LUMI = LOOKDEV_DIR / "01_lumi_character_sheet.jpg"
REF_PIP = LOOKDEV_DIR / "03_pip_creature_sheet.jpg"
KEYFRAME_BONDING = LOOKDEV_DIR / "04_lumi_and_pip_bonding.jpg"


def get_server_url() -> str:
    url_file = BASE_DIR / "current_url.txt"
    if url_file.exists():
        u = url_file.read_text().strip().rstrip("/")
        if u:
            try:
                with urllib.request.urlopen(f"{u}/health", timeout=8) as r:
                    if r.status == 200:
                        return u
            except Exception:
                pass
    raise RuntimeError("Chưa kết nối được Kaggle Server. Vui lòng kiểm tra Cloudflare URL.")


def encode_b64(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{b64}"


def crop_resize_768x448(src_path: Path, out_path: Path) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-vf", "scale=768:448:force_original_aspect_ratio=increase,crop=768:448",
        "-frames:v", "1",
        "-update", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def post_video_generation(base_url: str, payload: dict, timeout: int = 1200) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/videos/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        vid_b64 = res["data"][0]["b64_json"]
        return base64.b64decode(vid_b64)


def extract_last_5_frames(video_path: Path, prefix: str) -> list[Path]:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    v_stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    nb_frames = int(v_stream.get("nb_frames") or 56)

    frames = []
    for offset in range(5, 0, -1):
        f_idx = max(0, nb_frames - offset)
        out_png = OUT_DIR / f"{prefix}_f{5-offset}.png"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select='eq(n\\,{f_idx})'",
            "-frames:v", "1",
            "-update", "1",
            str(out_png),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frames.append(out_png)
    return frames


def main():
    base_url = get_server_url()
    print(f"✅ Kết nối Kaggle Server: {base_url}")

    # Chuẩn bị ảnh 768x448
    keyframe_768 = OUT_DIR / "keyframe_bonding_768x448.png"
    crop_resize_768x448(KEYFRAME_BONDING, keyframe_768)

    ref_lumi_768 = OUT_DIR / "ref_lumi_768x448.png"
    crop_resize_768x448(REF_LUMI, ref_lumi_768)

    ref_pip_768 = OUT_DIR / "ref_pip_768x448.png"
    crop_resize_768x448(REF_PIP, ref_pip_768)

    shots = [
        {
            "id": "shot_01_healing_breath",
            "prompt": (
                "Pixar 3D animated cinematic close-up: Featuring the exact character Lumi from <Picture 1> and magical spirit Pip from <Picture 2>. "
                "In the cozy warm clockwork workshop, 10-year-old Lumi gently cradles Pip in her cupped hands. "
                "Pip's glowing mint-teal petal wings flutter softly with tiny golden sparkles as the miniature brass gear splint holds. "
                "Pip looks up with large sparkling purple eyes, breathing a soft pulse of bioluminescent teal light that casts warm green-blue reflections "
                "across Lumi's freckled smiling face. Soft camera slow push-in, volumetric warm lantern lighting, emotive 3D animation."
            ),
            "num_frames": 56,
            "seed": 701001,
            "use_guide": False,
        },
        {
            "id": "shot_02_pip_takeoff",
            "prompt": (
                "Pixar 3D animated cinematic medium shot: Featuring Lumi from <Picture 1> and Pip from <Picture 2>. "
                "Pip joyfully lifts off from Lumi's cupped hands into the air, doing a graceful spiral loop with fluttering glowing wings. "
                "A trail of sparkling golden stardust swirls around Lumi's face. Lumi chuckles with delight, pushing her brass goggles up her forehead, "
                "watching her new companion fly with wonder and affection. Smooth tracking camera following Pip's ascent."
            ),
            "num_frames": 56,
            "seed": 701002,
            "use_guide": True,
        },
    ]

    all_videos = []
    prev_video = None

    for shot in shots:
        vid_path = OUT_DIR / f"{shot['id']}.mp4"
        if vid_path.exists() and vid_path.stat().st_size > 100_000:
            print(f"⏩ [Đã tồn tại] {vid_path.name}, bỏ qua.")
            all_videos.append(vid_path)
            prev_video = vid_path
            continue

        payload = {
            "model": "Comfy-Org/MiniMax-H3-Turbo",
            "mode": "ref2va",
            "turbo": True,
            "prompt": shot["prompt"],
            "reference_images": [encode_b64(ref_lumi_768), encode_b64(ref_pip_768)],
            "width": 768,
            "height": 448,
            "num_frames": shot["num_frames"],
            "fps": 24,
            "steps": 6,
            "guidance": 1.0,
            "seed": shot["seed"],
            "enable_audio": False,
        }

        if not shot["use_guide"]:
            payload["first_frame"] = encode_b64(keyframe_768)
        else:
            guide_frames = extract_last_5_frames(prev_video, shot["id"])
            payload["guide_frames"] = [encode_b64(f) for f in guide_frames]
            payload["first_frame"] = encode_b64(guide_frames[-1])

        print(f"🚀 Đang render {shot['id']} ({shot['num_frames']} frames, 6 steps Turbo LoRA, Ref2VA)...")
        t0 = time.time()
        vid_bytes = post_video_generation(base_url, payload)
        dt = time.time() - t0
        vid_path.write_bytes(vid_bytes)
        print(f"✅ Hoàn tất trong {dt:.1f}s -> {vid_path.name} ({len(vid_bytes) / 1024:.1f} KB)")

        all_videos.append(vid_path)
        prev_video = vid_path

    # Ghép nối các shots thành video Cảnh 1 hoàn chỉnh
    print("\n🎞️ Đang ghép nối Scene 1 (loại bỏ 5 frame ghim)...")
    trimmed = []
    for i, vp in enumerate(all_videos):
        t_out = OUT_DIR / f"trimmed_0{i+1}.mp4"
        if i == 0:
            cmd = ["ffmpeg", "-y", "-i", str(vp), "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-r", "24", "-an", str(t_out)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(vp), "-vf", "select='gte(n\\,5)',setpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-r", "24", "-an", str(t_out)]
        subprocess.run(cmd, check=True, capture_output=True)
        trimmed.append(t_out)

    concat_txt = OUT_DIR / "concat_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.name}'" for p in trimmed) + "\n")

    final_scene1 = OUT_DIR / "lumi_scene1_awakening_no_voice.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-an",
        str(final_scene1)
    ], check=True, capture_output=True)

    print(f"🎉 Hoàn tất video Cảnh 1: {final_scene1} ({final_scene1.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
