#!/usr/bin/env python3
"""
🎬 Script Render Hồi 2 - Cảnh 4: "Bước Chân Vào Xứ Sở Dạ Quang" (Disney/Pixar Style)
Sử dụng mô hình MiniMax H3 Ref2VA + Turbo LoRA (6 steps) + 5-Frame Spatiotemporal Latent Guide:
- Reference 1: 01_lumi_character_sheet.jpg (Khóa nhận diện Lumi)
- Reference 2: 03_pip_creature_sheet.jpg (Khóa nhận diện Pip)
- Reference 3: 06_bioluminescent_grove.jpg (Khóa bối cảnh Rừng Nấm Dạ Quang)
- First Frame: 08_lumi_entering_grove.png (Ảnh Start Keyframe nạp vào Shot 1)
- Ghép nối 3 shots liên hoàn với 5-Frame Guide đảm bảo chuyển động chạy và bay liên tục 100% không giật.
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
OUT_DIR = BASE_DIR / "benchmark_outputs" / "lumi_scene4_grove"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_LUMI = LOOKDEV_DIR / "01_lumi_character_sheet.jpg"
REF_PIP = LOOKDEV_DIR / "03_pip_creature_sheet.jpg"
REF_GROVE = LOOKDEV_DIR / "06_bioluminescent_grove.jpg"
KEYFRAME_ENTERING = LOOKDEV_DIR / "08_lumi_entering_grove.png"


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


def post_video_generation(base_url: str, payload: dict, timeout: int = 1200, max_retries: int = 3) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(1, max_retries + 1):
        try:
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
        except Exception as e:
            print(f"⚠️ Thử lần {attempt}/{max_retries} gặp lỗi: {e}")
            if attempt < max_retries:
                time.sleep(5)
            else:
                raise


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
    if not KEYFRAME_ENTERING.exists():
        # Fallback to 06_bioluminescent_grove if keyframe not yet saved
        keyframe_src = REF_GROVE
    else:
        keyframe_src = KEYFRAME_ENTERING

    keyframe_768 = OUT_DIR / "keyframe_entering_768x448.png"
    crop_resize_768x448(keyframe_src, keyframe_768)

    ref_lumi_768 = OUT_DIR / "ref_lumi_768x448.png"
    crop_resize_768x448(REF_LUMI, ref_lumi_768)

    ref_pip_768 = OUT_DIR / "ref_pip_768x448.png"
    crop_resize_768x448(REF_PIP, ref_pip_768)

    ref_grove_768 = OUT_DIR / "ref_grove_768x448.png"
    crop_resize_768x448(REF_GROVE, ref_grove_768)

    shots = [
        {
            "id": "shot_01_grove_entrance",
            "prompt": (
                "Pixar 3D animated cinematic wide establishing shot: Featuring Lumi from <Picture 1>, Pip from <Picture 2>, and The Bioluminescent Grove from <Picture 3>. "
                "10-year-old girl Lumi walks through massive arching ancient mossy roots into the enchanting glowing mushroom forest. "
                "She slows her steps, eyes wide with awe and wonder. Pip flutters gracefully ahead of her like a tiny glowing teal lantern, "
                "casting ripples of light over the sparkling moss beneath her boots. Slow forward tracking camera, volumetric moonlight god rays, breathtaking magical atmosphere."
            ),
            "num_frames": 56,
            "seed": 802001,
            "use_guide": False,
        },
        {
            "id": "shot_02_mushroom_tracking",
            "prompt": (
                "Pixar 3D animated dynamic tracking shot: Featuring Lumi from <Picture 1> and Pip from <Picture 2> in <Picture 3>. "
                "Continuous motion: Pip swoops joyfully between towering three-story-high glowing cyan and violet mushroom stems, "
                "leaving a sparkling swirling ribbon of golden and teal stardust. Lumi jogs eagerly along the glowing forest path, "
                "following Pip with a bright adventurous smile, her wavy pigtails bouncing gently. Smooth cinematic tracking camera."
            ),
            "num_frames": 56,
            "seed": 802002,
            "use_guide": True,
        },
        {
            "id": "shot_03_spore_wonder",
            "prompt": (
                "Pixar 3D animated emotional close-medium shot: Featuring Lumi from <Picture 1> and Pip from <Picture 2>. "
                "Continuous motion: Pip spirals down gently and hovers right before Lumi's sparkling amber eyes with a joyful chirp. "
                "Lumi slows down and extends her gloved hand as a glowing floating light spore lands softly on her fingertip, blooming into sparkling ripples of light. "
                "Lumi chuckles with warmth, her face illuminated by the magical bioluminescent glow, pure Disney Pixar wonder."
            ),
            "num_frames": 56,
            "seed": 802003,
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
            "reference_images": [encode_b64(ref_lumi_768), encode_b64(ref_pip_768), encode_b64(ref_grove_768)],
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

    # Ghép nối các shots thành video Cảnh 4 hoàn chỉnh
    print("\n🎞️ Đang ghép nối Scene 4 (loại bỏ 5 frame ghim tại các điểm nối)...")
    trimmed = []
    for i, vp in enumerate(all_videos):
        t_out = OUT_DIR / f"trimmed_s4_0{i+1}.mp4"
        if i == 0:
            cmd = ["ffmpeg", "-y", "-i", str(vp), "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-r", "24", "-an", str(t_out)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(vp), "-vf", "select='gte(n\\,5)',setpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-r", "24", "-an", str(t_out)]
        subprocess.run(cmd, check=True, capture_output=True)
        trimmed.append(t_out)

    concat_txt = OUT_DIR / "concat_s4_list.txt"
    concat_txt.write_text("\n".join(f"file '{p.name}'" for p in trimmed) + "\n")

    final_scene4 = OUT_DIR / "lumi_scene4_bioluminescent_grove_no_voice.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-an",
        str(final_scene4)
    ], check=True, capture_output=True)

    print(f"🎉 Hoàn tất video Cảnh 4: {final_scene4} ({final_scene4.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
