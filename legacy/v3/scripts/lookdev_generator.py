#!/usr/bin/env python3
"""
🎨 Script Tự Động Sinh Bộ Ảnh Tham Chiếu LookDev (Disney/Pixar Style)
Sử dụng mô hình Qwen-Image-2.1 qua endpoint /v1/images/generations của Kaggle Server.
Sinh toàn bộ:
1. Character Sheet: Cô bé thợ máy Lumi (Turnaround, Biểu cảm, Chi tiết kính bảo hộ)
2. Creature Sheet: Linh thú Pip (Giọt sáng dạ quang ngọc bích)
3. Environment Plates: Làng Cog Haven, Rừng Nấm Dạ Quang, Vực Pha Lê, Trái Tim Rừng Cổ Thụ
4. Keyframe Concept Plates cho từng Hồi
"""

import base64
import json
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
OUT_DIR = BASE_DIR / "benchmark_outputs" / "lookdev_lumi_disney"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def generate_image(base_url: str, prompt: str, width: int, height: int, seed: int, steps: int = 25) -> bytes:
    payload = {
        "model": "qwen_image_21",
        "prompt": prompt,
        "negative_prompt": "photorealistic human, 2D flat drawing, low quality, distorted anatomy, bad hands, extra fingers, blurry, watermark",
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": 3.5,
        "seed": seed,
        "response_format": "b64_json",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/images/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        b64_img = res["data"][0]["b64_json"]
        return base64.b64decode(b64_img)


LOOKDEV_ASSETS = [
    # --- 1. NHÂN VẬT CHÍNH: LUMI ---
    {
        "id": "01_lumi_character_sheet",
        "title": "Lumi - Character Concept Sheet (Front & 3/4 Turnaround)",
        "category": "character",
        "width": 1024,
        "height": 1024,
        "seed": 2026101,
        "prompt": (
            "Pixar and Disney 3D animation style character concept model sheet of a cute 10-year-old girl named Lumi: "
            "warm amber sparkling curious eyes, rosy freckled cheeks, messy voluminous wavy hazel-brown hair tied in high pigtails. "
            "Wearing brass clockwork tinker goggles with glowing cyan crystal lens resting on her forehead. "
            "Earthy caramel-brown leather overall tinker apron over a cream puffy-sleeve linen shirt, small leather tool pouch with tiny gears. "
            "Front view and three-quarter view turnaround, warm cinematic lighting, subsurface scattering skin, clean soft background."
        ),
    },
    {
        "id": "02_lumi_close_expressions",
        "title": "Lumi - Emotional Expressions & Goggles Detail",
        "category": "character",
        "width": 1024,
        "height": 1024,
        "seed": 2026102,
        "prompt": (
            "Disney Pixar 3D animated feature close-up portrait of Lumi: "
            "pulling down her brass tinker goggles over her eyes, determined brave expression, "
            "huge sparkling expressive eyes reflecting tiny dancing light spores, warm smile, "
            "soft hazel pigtails fluttering gently in wind, cinematic volumetric golden rim lighting, 8k render, Octane render quality."
        ),
    },

    # --- 2. BẠN ĐỒNG HÀNH: LINH THÚ PIP ---
    {
        "id": "03_pip_creature_sheet",
        "title": "Pip - Bioluminescent Light Sporeling Creature Sheet",
        "category": "creature",
        "width": 1024,
        "height": 1024,
        "seed": 2026103,
        "prompt": (
            "Pixar 3D animation style creature design sheet of Pip, an adorable tiny magical light spirit: "
            "apple-sized teardrop-shaped body made of translucent soft glowing mint-teal bioluminescent energy, "
            "two soft floppy glowing lotus-petal ears, big innocent glossy purple eyes with tiny star reflections. "
            "Floating gracefully in mid-air leaving a sparkling dust trail of glowing golden and cyan pollen spores, "
            "cute smiling expression, soft volumetric glow, ultra high detail 3D render."
        ),
    },
    {
        "id": "04_lumi_and_pip_bonding",
        "title": "Lumi & Pip - First Meeting on Workbench",
        "category": "keyframe",
        "width": 1280,
        "height": 720,
        "seed": 2026104,
        "prompt": (
            "Disney Pixar 3D animated cinematic shot: 10-year-old girl Lumi gently cradling the tiny glowing teal spirit Pip "
            "in her cupped hands on a rustic wooden tinker workbench. Lumi has applied a tiny brass watch-gear splint to Pip's fragile glowing petal-wing. "
            "Pip looks up at Lumi with grateful glowing eyes, floating gently above her palms with warm sparkles. "
            "Cluttered clockwork workshop in background with glowing crystal vials and copper pulleys, warm cozy lantern light, emotional storytelling."
        ),
    },

    # --- 3. BỐI CẢNH MASTER PLATES ---
    {
        "id": "05_cog_haven_sunset",
        "title": "Environment Plate - Cog Haven Village at Golden Hour",
        "category": "environment",
        "width": 1280,
        "height": 720,
        "seed": 2026105,
        "prompt": (
            "Disney Pixar 3D animated panoramic establishing landscape shot of Cog Haven Village: "
            "quaint fairy-tale wooden cottages built into towering ancient mossy trees, interconnected by rope suspension bridges "
            "and copper gear pulleys, waterwheels and clockwork windmills spinning peacefully. "
            "Golden hour sunset casting deep warm orange and pink light across the valley, chimneys puffing cozy smoke, lush green nature."
        ),
    },
    {
        "id": "06_bioluminescent_grove",
        "title": "Environment Plate - The Bioluminescent Grove",
        "category": "environment",
        "width": 1280,
        "height": 720,
        "seed": 2026106,
        "prompt": (
            "Pixar 3D animated feature cinematic landscape: Deep enchanted Lumina forest at night, "
            "colossal three-story-high translucent glowing mushrooms radiating vivid cyan, violet, and amber bioluminescence. "
            "Luminescent moss carpet glowing where footsteps touch the ground, floating glowing spore particles drifting like fireflies in mist, "
            "majestic magical atmosphere, deep atmospheric depth, Avatar meets Tangled, volumetric god rays."
        ),
    },
    {
        "id": "07_crystal_chasm_bridge",
        "title": "Environment Plate - The Crystal Chasm & Ancient Root Bridge",
        "category": "environment",
        "width": 1280,
        "height": 720,
        "seed": 2026107,
        "prompt": (
            "Disney Pixar 3D animation wide cinematic vista of The Crystal Chasm: "
            "a colossal ancient tree-root bridge spanning across a dizzying dark canyon filled with towering glowing quartz crystal pillars. "
            "Pulsing rhythmically with teal and amethyst light from below, drifting purple fog, epic scale, breathtaking fantasy adventure vista."
        ),
    },
    {
        "id": "08_heartwood_sanctum_restored",
        "title": "Environment Plate - The Heartwood Sanctum Restored",
        "category": "environment",
        "width": 1280,
        "height": 720,
        "seed": 2026108,
        "prompt": (
            "Pixar 3D animated climactic moment: Inside the colossal hollow Heartwood of the Ancient World Tree Oakhaven, "
            "the giant blooming light flower at the center unleashes a blinding shockwave of glorious warm emerald and golden aurora light. "
            "All ancient carvings and wooden spirals glowing with pure life energy, millions of glowing flower petals raining down, pure cinematic wonder."
        ),
    },
]


def main():
    base_url = get_server_url()
    print(f"✅ Kết nối Kaggle Server: {base_url}")
    print(f"🎨 Bắt đầu sinh bộ LookDev ({len(LOOKDEV_ASSETS)} tài nguyên) phong cách 3D Disney/Pixar...")

    for i, item in enumerate(LOOKDEV_ASSETS, 1):
        out_png = OUT_DIR / f"{item['id']}.png"
        if out_png.exists() and out_png.stat().st_size > 50_000:
            print(f"⏩ [{i}/{len(LOOKDEV_ASSETS)}] Đã tồn tại {out_png.name}, bỏ qua.")
            continue

        print(f"\n🖼️ [{i}/{len(LOOKDEV_ASSETS)}] Đang sinh '{item['title']}' ({item['width']}x{item['height']}, seed={item['seed']})...")
        t0 = time.time()
        try:
            img_bytes = generate_image(
                base_url,
                prompt=item["prompt"],
                width=item["width"],
                height=item["height"],
                seed=item["seed"],
            )
            out_png.write_bytes(img_bytes)
            dt = time.time() - t0
            print(f"✅ Hoàn thành trong {dt:.1f}s -> {out_png.name} ({len(img_bytes) / 1024:.1f} KB)")
        except Exception as e:
            print(f"❌ Lỗi sinh {item['id']}: {e}")

    print(f"\n🎉 Hoàn tất sinh toàn bộ tài nguyên LookDev tại: {OUT_DIR}")


if __name__ == "__main__":
    main()
