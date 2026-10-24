#!/usr/bin/env python3
"""
🎬 Script Render Hồi 2 - Cảnh 5: "Cầu Rễ Cây & Thử Thách Vực Pha Lê" (Disney/Pixar Style)
Sử dụng mô hình MiniMax H3 Ref2VA + Turbo LoRA (6 steps) + 5-Frame Spatiotemporal Latent Guide:
- Reference 1: 01_lumi_character_sheet.jpg (Khóa nhận diện Lumi)
- Reference 2: 03_pip_creature_sheet.jpg (Khóa nhận diện Pip)
- Reference 3: 07_crystal_chasm_bridge.jpg (Khóa bối cảnh Vực Pha Lê & Cầu Rễ Cây)
- First Frame: 09_lumi_crystal_bridge_crossing.png (Ảnh Start Keyframe nạp vào Shot 1)
- Ghép nối 2 shots liên hoàn với 5-Frame Spatiotemporal Guide đảm bảo chuyển động bước đi và gió thổi liên tục không giật.
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
OUT_DIR = BASE_DIR / "benchmark_outputs" / "lumi_scene5_chasm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_LUMI = LOOKDEV_DIR / "01_lumi_character_sheet.jpg"
REF_PIP = LOOKDEV_DIR / "03_pip_creature_sheet.jpg"
REF_CHASM = LOOKDEV_DIR / "07_crystal_chasm_bridge.jpg"
KEYFRAME_CROSSING = LOOKDEV_DIR / "09_lumi_crystal_bridge_crossing.png"


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
    if not KEYFRAME_CROSSING.exists():
        keyframe_src = REF_CHASM
    else:
        keyframe_src = KEYFRAME_CROSSING

    kframe_768 = OUT_DIR / "keyframe_crossing_768x448.jpg"
    crop_resize_768x448(keyframe_src, kframe_768)

    ref_lumi_b64 = encode_b64(REF_LUMI)
    ref_pip_b64 = encode_b64(REF_PIP)
    ref_chasm_b64 = encode_b64(REF_CHASM)
    kframe_b64 = encode_b64(kframe_768)

    # =========================================================================
    # SHOT 5.1: Bước chân qua cầu rễ cây & Ánh sáng dẫn đường của Pip
    # =========================================================================
    shot5_1_path = OUT_DIR / "shot5_1_crossing_bridge.mp4"
    if not shot5_1_path.exists():
        print("\n🚀 [1/3] Đang render Shot 5.1: Lumi bước lên cầu rễ cây, Pip bay soi đường...")
        prompt_5_1 = (
            "Disney Pixar 3D animation style. A 10-year-old girl named Lumi with messy wavy brown pigtails, "
            "brass tinker goggles on her forehead, and caramel leather overalls carefully walking across a colossal ancient mossy tree root bridge "
            "spanning over a dizzying deep canyon. Beside and ahead of her floats Pip, an adorable tiny glowing mint-teal teardrop spirit with lotus-petal ears, "
            "illuminating the mossy wooden root path with a soft radiant glow and sparkling light spores. "
            "Far below in the purple mist, gigantic glowing quartz crystal pillars pulse with cyan and amethyst light. "
            "Gentle forward camera movement, cinematic volumetric lighting, breathtaking fantasy adventure scale, masterpiece."
        )
        payload_5_1 = {
            "prompt": prompt_5_1,
            "negative_prompt": "blurry, low quality, deformed, flickering, stuttering, photorealistic, ugly, distorted faces",
            "ref_images": [ref_lumi_b64, ref_pip_b64, ref_chasm_b64],
            "guides": [
                {
                    "frame_idx": 0,
                    "image": kframe_b64
                }
            ],
            "width": 768,
            "height": 448,
            "num_frames": 56,
            "steps": 6,
            "guidance_scale": 1.0,
            "seed": 42051,
        }
        t0 = time.time()
        video_bytes = post_video_generation(base_url, payload_5_1)
        shot5_1_path.write_bytes(video_bytes)
        print(f"✅ Hoàn thành Shot 5.1 ({len(video_bytes):,} bytes, {time.time()-t0:.1f}s)")
    else:
        print(f"ℹ️ Shot 5.1 đã tồn tại: {shot5_1_path}")

    # =========================================================================
    # TRÍCH XUẤT 5 FRAMES CUỐI CỦA SHOT 5.1 LÀM SPATIOTEMPORAL GUIDE
    # =========================================================================
    print("🎯 Trích xuất 5 frame cuối Shot 5.1 làm Spatiotemporal Latent Guide...")
    guide_frames_5_1 = extract_last_5_frames(shot5_1_path, "shot5_1_guide")
    print(f"✅ Đã trích xuất {len(guide_frames_5_1)} frame guide: {[f.name for f in guide_frames_5_1]}")

    # =========================================================================
    # SHOT 5.2: Gió thần bí qua vực sâu & Cánh sen dạ quang tiếp sức
    # =========================================================================
    shot5_2_path = OUT_DIR / "shot5_2_protective_gust.mp4"
    if not shot5_2_path.exists():
        print("\n🚀 [2/3] Đang render Shot 5.2: Cơn gió thần bí qua vực sâu & Dải sáng đom đóm nâng đỡ...")
        prompt_5_2 = (
            "Disney Pixar 3D animation style. Seamless motion continuation. "
            "A 10-year-old girl Lumi on the ancient mossy root bridge pauses and balances courageously as a sudden gust of mystical wind sweeps through the canyon, "
            "her brown pigtails gently fluttering. Pip, the tiny glowing mint-teal light spirit with lotus-petal ears, flares its radiant wings brightly, "
            "summoning a magical swirling ribbon of glowing golden and cyan fireflies around Lumi, illuminating her path and giving her warm courage "
            "to step confidently toward the other side of the crystal canyon. "
            "Colossal glowing purple quartz crystals in the mist below, rich volumetric god rays, magical fairy tale wonder, masterpiece."
        )
        guides_5_2 = []
        for i, frame_p in enumerate(guide_frames_5_1):
            guides_5_2.append({
                "frame_idx": i,
                "image": encode_b64(frame_p)
            })

        payload_5_2 = {
            "prompt": prompt_5_2,
            "negative_prompt": "blurry, low quality, deformed, flickering, jump cut, stuttering, photorealistic, ugly",
            "ref_images": [ref_lumi_b64, ref_pip_b64, ref_chasm_b64],
            "guides": guides_5_2,
            "width": 768,
            "height": 448,
            "num_frames": 56,
            "steps": 6,
            "guidance_scale": 1.0,
            "seed": 42052,
        }
        t0 = time.time()
        video_bytes = post_video_generation(base_url, payload_5_2)
        shot5_2_path.write_bytes(video_bytes)
        print(f"✅ Hoàn thành Shot 5.2 ({len(video_bytes):,} bytes, {time.time()-t0:.1f}s)")
    else:
        print(f"ℹ️ Shot 5.2 đã tồn tại: {shot5_2_path}")

    # =========================================================================
    # [3/3] HÒA ÂM HÌNH ẢNH: CẮT 5 FRAME TRÙNG VÀ GHÉP SEAMLESS NỐI LIỀN
    # =========================================================================
    print("\n🎞️ [3/3] Đang cắt 5 frame trùng và ghép nối liền thành Video Cảnh 5 hoàn chỉnh...")
    shot5_2_trimmed = OUT_DIR / "shot5_2_trimmed.mp4"
    cmd_trim = [
        "ffmpeg", "-y",
        "-i", str(shot5_2_path),
        "-vf", "select='gte(n\\,5)',setpts=PTS-STARTPTS",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(shot5_2_trimmed),
    ]
    subprocess.run(cmd_trim, check=True, capture_output=True)

    final_video = OUT_DIR / "lumi_scene5_crystal_chasm_no_voice.mp4"
    concat_list = OUT_DIR / "concat_list.txt"
    concat_list.write_text(f"file '{shot5_1_path.name}'\nfile '{shot5_2_trimmed.name}'\n")

    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(final_video),
    ]
    subprocess.run(cmd_concat, check=True, capture_output=True, cwd=OUT_DIR)

    # Trích xuất keyframes để kiểm tra độ mượt
    frames_dir = OUT_DIR / "inspection_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f_idx, name in [(10, "f010"), (45, "f045"), (55, "f055_seam1"), (56, "f056_seam2"), (75, "f075"), (100, "f100")]:
        cmd_f = [
            "ffmpeg", "-y",
            "-i", str(final_video),
            "-vf", f"select='eq(n\\,{f_idx})'",
            "-frames:v", "1",
            "-update", "1",
            str(frames_dir / f"{name}.png"),
        ]
        subprocess.run(cmd_f, check=True, capture_output=True)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH RENDER HỒI 2 - CẢNH 5: THỬ THÁCH VỰC PHA LÊ!")
    print(f"👉 File Video Cảnh 5 : {final_video}")
    print("=" * 70)


if __name__ == "__main__":
    main()
