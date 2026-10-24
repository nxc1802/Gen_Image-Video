#!/usr/bin/env python3
"""
🎬 Script Render Hồi 1 - Cảnh 3: "Khởi Hành Trong Ánh Hoàng Hôn" (Disney/Pixar Style)
Sử dụng mô hình MiniMax H3 Ref2VA + Turbo LoRA (6 steps) + 5-Frame Spatiotemporal Latent Guide:
- Reference 1: 01_lumi_character_sheet.jpg (Khóa nhận diện Lumi)
- Reference 2: 03_pip_creature_sheet.jpg (Khóa nhận diện Pip)
- Reference 3: 05_cog_haven_sunset.jpg (Khóa bối cảnh Làng Cog Haven lúc hoàng hôn)
- First Frame: 05_cog_haven_sunset.jpg (Ảnh Master Plate Làng Bánh Răng)
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
OUT_DIR = BASE_DIR / "benchmark_outputs" / "lumi_scene3_departure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_LUMI = LOOKDEV_DIR / "01_lumi_character_sheet.jpg"
REF_PIP = LOOKDEV_DIR / "03_pip_creature_sheet.jpg"
REF_SUNSET = LOOKDEV_DIR / "05_cog_haven_sunset.jpg"


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

    kframe_768 = OUT_DIR / "keyframe_sunset_768x448.jpg"
    crop_resize_768x448(REF_SUNSET, kframe_768)

    ref_lumi_b64 = encode_b64(REF_LUMI)
    ref_pip_b64 = encode_b64(REF_PIP)
    ref_sunset_b64 = encode_b64(REF_SUNSET)
    kframe_b64 = encode_b64(kframe_768)

    # =========================================================================
    # SHOT 3.1: Lumi nhìn về hoàng hôn thung lũng Cog Haven
    # =========================================================================
    shot3_1_path = OUT_DIR / "shot3_1_overlooking_sunset.mp4"
    if not shot3_1_path.exists():
        print("\n🚀 [1/3] Đang render Shot 3.1: Lumi trên cầu dây văng nhìn hoàng hôn làng Cog Haven...")
        prompt_3_1 = (
            "Disney Pixar 3D animation style. A 10-year-old girl named Lumi with messy hazel-brown pigtails, "
            "brass tinker goggles on her forehead, and caramel leather overalls stands at the wooden suspension bridge of Cog Haven village. "
            "She gazes across the breathtaking fairy tale valley bathed in warm golden-orange sunset light, with spinning waterwheels and clockwork cottages. "
            "Floating gently beside her shoulder is Pip, the tiny glowing mint-teal teardrop spirit with lotus ears, leaving faint sparkling dust in the golden hour air. "
            "Slow cinematic camera zoom-in, warm nostalgic lighting, emotional storytelling, masterpiece."
        )
        payload_3_1 = {
            "prompt": prompt_3_1,
            "negative_prompt": "blurry, low quality, deformed, flickering, stuttering, photorealistic, ugly, distorted faces",
            "ref_images": [ref_lumi_b64, ref_pip_b64, ref_sunset_b64],
            "guides": [{"frame_idx": 0, "image": kframe_b64}],
            "width": 768,
            "height": 448,
            "num_frames": 56,
            "steps": 6,
            "guidance_scale": 1.0,
            "seed": 42031,
        }
        t0 = time.time()
        video_bytes = post_video_generation(base_url, payload_3_1)
        shot3_1_path.write_bytes(video_bytes)
        print(f"✅ Hoàn thành Shot 3.1 ({len(video_bytes):,} bytes, {time.time()-t0:.1f}s)")
    else:
        print(f"ℹ️ Shot 3.1 đã tồn tại: {shot3_1_path}")

    # =========================================================================
    # TRÍCH XUẤT 5 FRAMES CUỐI CỦA SHOT 3.1 LÀM SPATIOTEMPORAL GUIDE
    # =========================================================================
    print("🎯 Trích xuất 5 frame cuối Shot 3.1 làm Spatiotemporal Latent Guide...")
    guide_frames_3_1 = extract_last_5_frames(shot3_1_path, "shot3_1_guide")
    print(f"✅ Đã trích xuất {len(guide_frames_3_1)} frame guide: {[f.name for f in guide_frames_3_1]}")

    # =========================================================================
    # SHOT 3.2: Lumi quay lưng bước đi về phía con đường rừng
    # =========================================================================
    shot3_2_path = OUT_DIR / "shot3_2_stepping_into_journey.mp4"
    if not shot3_2_path.exists():
        print("\n🚀 [2/3] Đang render Shot 3.2: Lumi bước đi quyết đoán về phía rừng già...")
        prompt_3_2 = (
            "Disney Pixar 3D animation style. Seamless motion continuation. "
            "A 10-year-old girl Lumi turns with a determined, courageous smile and begins walking purposefully forward along the mossy stone path leading away from the village. "
            "Pip, the glowing mint-teal spirit, happily zips ahead in a playful arc, leading the way. "
            "The warm sunset rays give way to the deep enchanting twilight of the forest edge, golden dust and floating fireflies, smooth tracking camera movement, masterpiece."
        )
        guides_3_2 = [{"frame_idx": i, "image": encode_b64(f)} for i, f in enumerate(guide_frames_3_1)]

        payload_3_2 = {
            "prompt": prompt_3_2,
            "negative_prompt": "blurry, low quality, deformed, flickering, jump cut, stuttering, photorealistic, ugly",
            "ref_images": [ref_lumi_b64, ref_pip_b64, ref_sunset_b64],
            "guides": guides_3_2,
            "width": 768,
            "height": 448,
            "num_frames": 56,
            "steps": 6,
            "guidance_scale": 1.0,
            "seed": 42032,
        }
        t0 = time.time()
        video_bytes = post_video_generation(base_url, payload_3_2)
        shot3_2_path.write_bytes(video_bytes)
        print(f"✅ Hoàn thành Shot 3.2 ({len(video_bytes):,} bytes, {time.time()-t0:.1f}s)")
    else:
        print(f"ℹ️ Shot 3.2 đã tồn tại: {shot3_2_path}")

    # =========================================================================
    # [3/3] CẮT 5 FRAME TRÙNG VÀ GHÉP SEAMLESS NỐI LIỀN
    # =========================================================================
    print("\n🎞️ [3/3] Đang cắt 5 frame trùng và ghép nối liền thành Video Cảnh 3 hoàn chỉnh...")
    shot3_2_trimmed = OUT_DIR / "shot3_2_trimmed.mp4"
    cmd_trim = [
        "ffmpeg", "-y",
        "-i", str(shot3_2_path),
        "-vf", "select='gte(n\\,5)',setpts=PTS-STARTPTS",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(shot3_2_trimmed),
    ]
    subprocess.run(cmd_trim, check=True, capture_output=True)

    final_video = OUT_DIR / "lumi_scene3_sunset_departure_no_voice.mp4"
    concat_list = OUT_DIR / "concat_list.txt"
    concat_list.write_text(f"file '{shot3_1_path.name}'\nfile '{shot3_2_trimmed.name}'\n")

    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(final_video),
    ]
    subprocess.run(cmd_concat, check=True, capture_output=True, cwd=OUT_DIR)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH RENDER HỒI 1 - CẢNH 3: KHỞI HÀNH TRONG ÁNH HOÀNG HÔN!")
    print(f"👉 File Video Cảnh 3 : {final_video}")
    print("=" * 70)


if __name__ == "__main__":
    main()
