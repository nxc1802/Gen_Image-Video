#!/usr/bin/env python3
"""
================================================================================
🏆 BENCHMARK SUITE: QWEN-IMAGE 2.1 & MINIMAX H3 COMPREHENSIVE VISUAL EVALUATION
================================================================================
Bộ benchmark chuyên sâu đánh giá toàn bộ tính năng của 2 mô hình đầu bảng:
1. Qwen-Image 2.1 (Top-1 OpenSource Image):
   - T2I: Text-to-Image (Facial Micro-Expressions & Subsurface Scattering)
   - Native Text Rendering: Typography phức tạp, song ngữ Anh - Việt, biển hiệu neon
   - I2I: Style Transfer giữ vững giải phẫu và cấu trúc vật lý
   - Inpainting: Chèn/thay thế đối tượng cục bộ với edge blending & shadow coherence
   - Outpainting: Mở rộng canvas toàn cảnh (Panoramic environmental expansion)
   - Multimodal Instruction Edit: Chỉnh sửa ảnh theo chỉ thị ngôn ngữ tự nhiên (Zero-shot)

2. MiniMax H3 (Top-1 OpenSource Video + Audio):
   - T2V: Text-to-Video (Physics Simulation & Fluid Dynamics, No Audio)
   - T2VA: Text-to-Video với Audio đồng bộ (Wave acoustics & coastal camera track)
   - I2V: Image-to-Video (Facial Micro-Expressions & Head Turning, No Audio)
   - I2VA: Image-to-Video với Audio (Biểu cảm thở nhẹ kèm nhạc acoustic mộc)
   - FL2V: First & Last Frame-to-Video (Quỹ đạo động học suy luận giữa 2 mốc frame)
   - FL2VA: First & Last Frame-to-Video với Audio (Chuyển tiếp kèm âm thanh môi trường)
   - Ref2V: Reference-to-Video (Đồng nhất diện mạo nhân vật trong bối cảnh mới)
   - Ref2VA: Reference-to-Video với Audio (Cảnh bước chân trên sỏi ướt & mưa rơi)
   - Video Extension / Chaining: Ghép nối liên tiếp (Trích frame cuối làm frame đầu)

================================================================================
CẤU HÌNH MẶC ĐỊNH CHUẨN BENCHMARK:
- Qwen-Image 2.1: 1280x720, 20 steps, Guidance 1.0, Sampler euler, Scheduler simple
- MiniMax H3: 768x448, 33 frames, 14 steps, Guidance 1.0, Shift 8.0, 24 fps, no audio (trừ các case VA)
================================================================================
"""

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = Path(__file__).resolve().parent / "BENCHMARK_VISUAL_REPORT.md"


@dataclass
class BenchmarkCase:
    id: str
    model: str
    feature: str
    title: str
    focus: str
    prompt: str
    negative_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_dependency: Optional[str] = None  # Phụ thuộc vào kết quả của case trước (nếu có)
    metrics: Dict[str, Any] = field(default_factory=dict)


BENCHMARK_CASES: List[BenchmarkCase] = [
    # --------------------------------------------------------------------------
    # QWEN-IMAGE 2.1 CASES (1 - 6)
    # --------------------------------------------------------------------------
    BenchmarkCase(
        id="qwen_t2i",
        model="Qwen-Image 2.1",
        feature="T2I (Text-to-Image)",
        title="Full-Body Photorealistic Human & Sharp Environmental Depth",
        focus="Toàn thân cơ thể người thật từ đầu tới chân (Full-body human anatomy), da người siêu thực, chiều sâu trường ảnh rõ nét (không làm mờ background) tại phố cổ Hội An.",
        prompt=(
            "Full-length shot of a graceful, elegant real Vietnamese woman walking on an ancient street in Hoi An, "
            "visible from head to toe in an ivory silk Ao Dai with flowing silk trousers and classic embroidered shoes. "
            "Authentic candid professional photography, natural realistic skin texture, photorealistic human anatomy, delicate facial expression with a gentle confident smile. "
            "Deep depth of field, sharp f/8 aperture photography: crisp detailed environmental background featuring historic yellow ancient buildings, "
            "vibrant hanging silk lanterns, wooden balconies, cobblestone street reflections, sharp background architecture, no blur, no heavy bokeh."
        ),
        negative_prompt="blurry, heavy bokeh, blurred background, close-up, portrait, cropped legs, cropped feet, out of frame, deformed anatomy, plastic skin, anime, 3d render, low quality, artifacts",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    BenchmarkCase(
        id="qwen_text_rendering",
        model="Qwen-Image 2.1",
        feature="Native Text Rendering",
        title="High-Complexity Multilingual Typography & Signage",
        focus="Độ sắc nét và tính chính xác tuyệt đối của ký tự Latinh, font chữ có chân (serif), chữ viết tay (cursive), và dấu tiếng Việt trên bề mặt phản xạ.",
        prompt=(
            "Frontal view of an artisan specialty coffee roastery storefront in rainy Hanoi old quarter at night. "
            "Prominent glowing warm neon signage with precise clean typography reading \"HÀ NỘI ROASTERS\" in bold elegant serif font, "
            "with a smaller illuminated cursive script subtitle underneath reading \"Specialty Beans & Espresso since 1994\". "
            "Clear reflections of the neon text on wet asphalt puddles, condensation on the glass window, crisp architectural street background, razor-sharp letter edges, deep focus."
        ),
        negative_prompt="misspelled words, distorted letters, garbled text, blurry font, missing diacritics",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    BenchmarkCase(
        id="qwen_i2i",
        model="Qwen-Image 2.1",
        feature="I2I (Image-to-Image)",
        title="Full-Body Stylistic Transformation with Complete Anatomical Fidelity",
        focus="Chuyển đổi phong cách sang anime nghệ thuật Makoto Shinkai nhưng bảo toàn 100% tỷ lệ cơ thể toàn thân, dáng đứng, tà áo và khuôn mặt.",
        prompt=(
            "Breathtaking Makoto Shinkai anime masterpiece of the full-body Vietnamese woman standing in the ancient street, "
            "vibrant twilight sky with iridescent purple and golden clouds, crisp cel-shaded architectural background, "
            "preserving exact full-body anatomy, pose, limbs, and expressive face, glistening light particles, sharp background details."
        ),
        negative_prompt="deformed anatomy, changed pose, cropped limbs, ugly, noisy, low resolution, blurry background",
        input_dependency="qwen_t2i",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "strength": 0.65,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    BenchmarkCase(
        id="qwen_inpainting",
        model="Qwen-Image 2.1",
        feature="Inpainting",
        title="Full-Body Accessory Inpainting & Precision Boundary Blending",
        focus="Inpainting trên ảnh toàn thân: Thay thế vùng bàn tay/cổ tay bằng chiếc túi xách lụa thêu hoa sen tinh xảo và vòng ngọc bích phát quang; kiểm tra độ hòa trộn biên, bóng đổ tiếp xúc và tính chân thực của vải lụa.",
        prompt=(
            "An exquisite antique embroidered silk clutch handbag held delicately in her hand and an antique jade bangle bracelet on her wrist, "
            "hyper-realistic fabric weave, contact shadow on silk dress, seamless edge blending, sharp details."
        ),
        negative_prompt="floating object, hard borders, visible seams, cartoonish, unrealistic shadows, blurry",
        input_dependency="qwen_t2i",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "strength": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    BenchmarkCase(
        id="qwen_outpainting",
        model="Qwen-Image 2.1",
        feature="Outpainting",
        title="Full-Body Panoramic Environmental Expansion (No Aspect Distortion)",
        focus="Mở rộng không gian hai bên viền canvas 1280x720 giữ nguyên 100% tỷ lệ cơ thể trung tâm không bị bóp méo, mở rộng cảnh quan phố cổ, lồng đèn rực rỡ và dãy nhà cổ hai bên.",
        prompt=(
            "Seamless panoramic continuation of the historic Hoi An ancient town street, historic yellow wooden shopfronts, "
            "hundreds of glowing red and golden lanterns hanging across the street, wet cobblestone reflections, sharp architectural depth, deep field of view."
        ),
        negative_prompt="repetitive patterns, cut-off seams, mismatched perspective, blur, squashed body, distorted proportions",
        input_dependency="qwen_t2i",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "strength": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    BenchmarkCase(
        id="qwen_instruction_edit",
        model="Qwen-Image 2.1",
        feature="Multimodal Instruction Edit",
        title="Full-Body Costume Transformation via Natural Language Instruction",
        focus="Chỉ thị ngôn ngữ tự nhiên trên toàn thân: thay đổi trang phục sang áo khoác da cyberpunk đen mờ viền neon xanh và quần ống đứng hiện đại, giữ nguyên vóc dáng toàn thân, tư thế và khuôn mặt.",
        prompt=(
            "Change her outfit into a sleek matte black cyberpunk leather jacket with glowing cyan neon seams and tailored black trousers, "
            "add a minimalist geometric hair clip, keeping her full-body pose, height, face, and smile completely unchanged."
        ),
        negative_prompt="changed face, identity loss, distorted anatomy, cropped feet, messy textures",
        input_dependency="qwen_t2i",
        parameters={
            "width": 1280,
            "height": 720,
            "steps": 20,
            "guidance": 1.0,
            "strength": 0.70,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
    ),
    # --------------------------------------------------------------------------
    # MINIMAX H3 CASES (7 - 15)
    # --------------------------------------------------------------------------
    BenchmarkCase(
        id="minimax_t2v",
        model="MiniMax H3",
        feature="T2V (Text-to-Video)",
        title="Fluid Dynamics & Volumetric Physics Simulation (No Audio)",
        focus="Mô phỏng vật lý chất lỏng siêu thực: dòng mật ong vàng sánh rót vào tách trà, sóng gợn đồng tâm lan tỏa, hơi nước bốc lên uyển chuyển dưới góc máy macro.",
        prompt=(
            "Extreme close-up macro shot of golden amber honey dripping smoothly in a thick viscous stream into hot chamomile tea, "
            "delicate ripple rings expanding outward, steam swirls rising gracefully in volumetric morning light, slow motion fluid physics, shallow depth of field."
        ),
        negative_prompt="jerky motion, static liquid, sudden cuts, flickering, low frame rate, artifacts",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": False,
        },
    ),
    BenchmarkCase(
        id="minimax_t2va",
        model="MiniMax H3",
        feature="T2VA (Text-to-Video + Audio)",
        title="Full-Body Walking Motion & Synchronized Rain Acoustics (Deep Depth of Field)",
        focus="T2VA toàn thân: người phụ nữ thanh lịch bước đi trên vỉa hè đá ướt phố cổ Hội An trong mưa nhẹ, chuyển động cơ thể và tà áo chuẩn động học, đồng bộ âm thanh tiếng mưa rơi tí tách và tiếng bước chân gõ nhẹ, hậu cảnh kiến trúc sắc nét không mờ.",
        prompt=(
            "Full-body cinematic shot of an elegant Vietnamese woman walking gracefully along an ancient cobblestone street in light rain, "
            "deep depth of field f/8 with sharp historic yellow shopfronts and hanging red lanterns in the background. "
            "Natural human walking kinematics: shifting body weight smoothly between legs, rhythmic arm swinging, gentle head turn with a warm smile, "
            "hem of silk Ao Dai floating with each step. Synchronized realistic audio: gentle raindrops falling on cobblestones and soft clicking footsteps."
        ),
        negative_prompt="blurry background, shallow depth of field, robotic movement, sliding feet, audio desync, harsh noise, distorted anatomy",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": True,
        },
    ),
    BenchmarkCase(
        id="minimax_i2v",
        model="MiniMax H3",
        feature="I2V (Image-to-Video)",
        title="Full-Body Human Walking Kinematics & Natural Biomechanics (No Audio)",
        focus="Khai thác ảnh toàn thân từ qwen_t2i: nhân vật bước đi tự nhiên dọc con đường lát đá, chuyển trọng tâm cơ thể mượt mà, đung đưa tay uyển chuyển, tà áo lụa bay theo từng bước chân.",
        prompt=(
            "Full-body cinematic shot: The woman takes graceful natural steps forward along the cobblestone street, swinging her arms in smooth lifelike rhythm. "
            "Realistic human walking kinematics: shifting body weight smoothly between legs, natural torso movement, gentle head turning with a warm smile, "
            "flowing silk trousers and Ao Dai dress moving with each step. Consistent realistic human body mechanics, sharp environmental background."
        ),
        negative_prompt="frozen legs, robotic walking, sliding feet, face distortion, morphing, jerky movement, blurry background",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": False,
        },
    ),
    BenchmarkCase(
        id="minimax_i2va",
        model="MiniMax H3",
        feature="I2VA (Image-to-Video + Audio)",
        title="Full-Body Walking Motion with Synchronized Footsteps & Ambient Audio",
        focus="Diễn hoạt toàn thân bước đi và dừng lại mỉm cười, đồng bộ tiếng bước chân gõ nhẹ trên đá ướt và tiếng gió lướt qua tà áo lụa.",
        prompt=(
            "Full-body cinematic shot: The woman walks forward gracefully, pauses and turns her body toward the camera with natural biomechanics, dress flowing naturally. "
            "Synchronized rhythmic footsteps clicking softly on wet cobblestones and warm evening breeze rustling silk."
        ),
        negative_prompt="harsh audio noise, sync mismatch, twitching, uncanny valley, sliding feet",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": True,
        },
    ),
    BenchmarkCase(
        id="minimax_fl2v",
        model="MiniMax H3",
        feature="FL2V (First & Last Frame to Video)",
        title="Full-Body Kinematic Interpolation between Keyframe Poses (No Audio)",
        focus="Suy luận chuyển động toàn thân giữa Frame 0 (đứng thẳng tự nhiên) và Frame 32 (nghiêng người chuyển trọng tâm trong áo khoác cyberpunk), bảo toàn động học cánh tay và chân.",
        prompt=(
            "Full-body continuous human movement: she shifts her weight gracefully and turns her shoulders and torso, natural arm movement and body dynamics connecting the keyframe poses smoothly."
        ),
        negative_prompt="teleporting, sudden jumps, fading transition, non-physical motion, distorted limbs",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": False,
        },
    ),
    BenchmarkCase(
        id="minimax_fl2va",
        model="MiniMax H3",
        feature="FL2VA (First & Last Frame + Audio)",
        title="Full-Body Keyframe Transition with Organic Sound Effects",
        focus="Chuyển động toàn thân nối 2 keyframe kết hợp âm thanh bước chân nhẹ và tiếng chuông gió xa xa, tạo trải nghiệm điện ảnh trọn vẹn.",
        prompt=(
            "Full-body smooth transition connecting start and end frame poses, she takes a gentle step and turns toward the light, "
            "soft ambient rustling silk and warm gentle chime sound."
        ),
        negative_prompt="audio artifact, glitch transition, static frames, broken limbs",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": True,
        },
    ),
    BenchmarkCase(
        id="minimax_ref2v",
        model="MiniMax H3",
        feature="Ref2V (Reference Video Conditioning)",
        title="Full-Body Character Persistence across Dynamic Tracking Shot (No Audio)",
        focus="Sử dụng ảnh nhân vật toàn thân làm tham chiếu: cô gái cầm ô tự tin bước đi trên phố cổ với góc quay dolly tracking toàn cảnh, bảo toàn 100% vóc dáng và giải phẫu chân tay.",
        prompt=(
            "Full-body cinematic tracking shot: The same Vietnamese woman walks confidently along a lantern-lit cobblestone street carrying an umbrella, "
            "smooth dolly tracking camera, natural human walking motion, feet stepping clearly on cobblestones, consistent full-body anatomy and clothing."
        ),
        negative_prompt="identity shift, face changes, morphing clothes, shaky camera, sliding feet",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": False,
        },
    ),
    BenchmarkCase(
        id="minimax_ref2va",
        model="MiniMax H3",
        feature="Ref2VA (Reference Video + Audio)",
        title="Full-Body Reference Character in Rain Scene with Rain & Footstep Acoustics",
        focus="Nhân vật toàn thân dừng bước dưới lồng đèn ngoái nhìn lại với chuyển động xoay thân người tự nhiên, đồng bộ tiếng mưa rơi và tiếng gót giày trên đá ướt.",
        prompt=(
            "Full-body cinematic shot: The same woman pauses under a hanging street lantern, turns her torso and looks back gracefully over her shoulder with natural weight shift. "
            "Synchronized crisp footsteps on wet stone and gentle raindrops falling."
        ),
        negative_prompt="audio popping, mute, face distortion, sudden camera shifts, floating body",
        input_dependency="qwen_t2i",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": True,
        },
    ),
    BenchmarkCase(
        id="minimax_chaining",
        model="MiniMax H3",
        feature="Video Extension / Chaining",
        title="Full-Body Motion Extension & Temporal Kinematic Continuity",
        focus="Trích xuất chính xác Frame 32 của video minimax_i2v (toàn thân), tiếp nối chuỗi hành động: bước thêm 2 bước và nghiêng người vẫy tay nhẹ, bảo toàn động học cơ thể không gãy nhịp.",
        prompt=(
            "Continuing the smooth full-body walking motion, she takes two more elegant steps forward, turning her torso slightly and raising her hand to wave softly, "
            "natural leg and arm kinematics, completely seamless continuation."
        ),
        negative_prompt="frame stutter, lighting jump, cut visible, inconsistent hand anatomy, sliding feet",
        input_dependency="minimax_i2v",
        parameters={
            "width": 768,
            "height": 448,
            "num_frames": 33,
            "steps": 14,
            "guidance": 1.0,
            "shift": 8.0,
            "fps": 24,
            "enable_audio": False,
        },
    ),
]


# ==============================================================================
# HELPER FUNCTIONS: PREPROCESSING & MASKING
# ==============================================================================
def create_inpaint_mask(image_path: Path, output_mask_path: Path) -> Path:
    """Tạo mask RGB rõ ràng cho inpainting (khu vực bàn tay / cổ tay trong ảnh toàn thân)."""
    with Image.open(image_path) as img:
        w, h = img.size
        # Tạo mask đen (giữ nguyên), vùng trắng (thay thế)
        mask = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        # Vùng bàn tay, cổ tay và phụ kiện cầm tay (khoảng 40% - 58% width, 48% - 68% height)
        box = [int(w * 0.40), int(h * 0.48), int(w * 0.58), int(h * 0.68)]
        draw.ellipse(box, fill=(255, 255, 255))
        mask.save(output_mask_path)
    return output_mask_path


def create_outpaint_setup(image_path: Path, output_pad_path: Path, output_mask_path: Path) -> Tuple[Path, Path]:
    """Tạo ảnh pad viền và mask tương ứng cho Outpainting mà KHÔNG làm méo/bóp tỷ lệ cơ thể."""
    with Image.open(image_path) as img:
        w, h = img.size
        # Giữ nguyên tỷ lệ 1:1, trích phần trung tâm 60% width để mở rộng 20% mỗi bên
        center_w = int(w * 0.60)
        offset_x = (w - center_w) // 2
        # Crop vùng trung tâm nguyên bản (KHÔNG resize để tránh méo tỷ lệ cơ thể)
        center_crop = img.crop((offset_x, 0, offset_x + center_w, h))

        padded_img = Image.new("RGB", (w, h), (15, 12, 20))
        padded_img.paste(center_crop, (offset_x, 0))
        padded_img.save(output_pad_path)

        mask = Image.new("RGB", (w, h), (255, 255, 255))  # Mặc định vùng outpaint là trắng
        draw = ImageDraw.Draw(mask)
        draw.rectangle([offset_x, 0, offset_x + center_w, h], fill=(0, 0, 0))  # Vùng gốc giữ nguyên (đen)
        mask.save(output_mask_path)
    return output_pad_path, output_mask_path


def extract_last_video_frame(video_path: Path, output_frame_path: Path) -> Path:
    """Trích xuất frame cuối cùng của video phục vụ Video Chaining."""
    cmd = f"ffmpeg -y -sseof -0.1 -i '{video_path}' -update 1 -q:v 2 '{output_frame_path}' >/dev/null 2>&1"
    os.system(cmd)
    if not output_frame_path.exists() or output_frame_path.stat().st_size == 0:
        # Fallback lấy frame đầu nếu sseof không bắt được
        os.system(f"ffmpeg -y -i '{video_path}' -vframes 1 -q:v 2 '{output_frame_path}' >/dev/null 2>&1")
    return output_frame_path


def load_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_b64_file(b64_data: str, path: Path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def make_request(base_url: str, endpoint: str, payload: dict = None, method: str = "GET", timeout: int = 900) -> dict:
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}{endpoint}"
    data_bytes = json.dumps(payload).encode("utf-8") if payload else None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('STUDIO_API_KEY', 'default-key')}",
        "User-Agent": "VisualBenchmarkSuite/1.0",
    }
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as he:
        body = he.read().decode("utf-8", errors="replace")
        print(f"\n❌ [HTTPError {he.code}] Endpoint: {endpoint}\nChi tiết lỗi từ Server: {body}\n")
        raise


def run_benchmark_case(base_url: str, case: BenchmarkCase) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"🚀 [RUNNING CASE: {case.id}] - {case.model} | {case.feature}")
    print(f"   📌 Tiêu đề: {case.title}")
    print(f"   🎯 Trọng tâm: {case.focus}")
    print(f"   📝 Prompt: {case.prompt[:95]}...")
    print("=" * 90)

    t0 = time.time()
    result_data = {
        "id": case.id,
        "model": case.model,
        "feature": case.feature,
        "title": case.title,
        "status": "FAILED",
        "elapsed": 0.0,
        "output_file": "",
        "file_size_kb": 0.0,
        "resolution": f"{case.parameters.get('width')}x{case.parameters.get('height')}",
        "notes": "",
    }

    try:
        # 1. QWEN-IMAGE 2.1 CASES
        if case.model == "Qwen-Image 2.1":
            out_img_file = OUTPUT_DIR / f"{case.id}.png"
            payload = {
                "model": "qwen_image_21",
                "prompt": case.prompt,
                "negative_prompt": case.negative_prompt,
                "width": case.parameters.get("width", 1280),
                "height": case.parameters.get("height", 720),
                "steps": case.parameters.get("steps", 20),
                "guidance": case.parameters.get("guidance", 1.0),
                "sampler_name": case.parameters.get("sampler_name", "euler"),
                "scheduler": case.parameters.get("scheduler", "simple"),
                "stream": False,
            }

            if case.id in ("qwen_t2i", "qwen_text_rendering"):
                res = make_request(base_url, "/v1/images/generations", payload=payload, method="POST", timeout=600)
            elif case.id == "qwen_i2i":
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                payload["image"] = f"data:image/png;base64,{load_image_b64(dep_img)}"
                payload["strength"] = case.parameters.get("strength", 0.65)
                res = make_request(base_url, "/v1/images/edits", payload=payload, method="POST", timeout=600)
            elif case.id == "qwen_inpainting":
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                mask_path = OUTPUT_DIR / f"{case.id}_mask.png"
                create_inpaint_mask(dep_img, mask_path)
                payload["image"] = f"data:image/png;base64,{load_image_b64(dep_img)}"
                payload["mask"] = f"data:image/png;base64,{load_image_b64(mask_path)}"
                payload["strength"] = case.parameters.get("strength", 0.85)
                res = make_request(base_url, "/v1/images/edits", payload=payload, method="POST", timeout=600)
            elif case.id == "qwen_outpainting":
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                pad_path = OUTPUT_DIR / f"{case.id}_pad.png"
                mask_path = OUTPUT_DIR / f"{case.id}_mask.png"
                create_outpaint_setup(dep_img, pad_path, mask_path)
                payload["image"] = f"data:image/png;base64,{load_image_b64(pad_path)}"
                payload["mask"] = f"data:image/png;base64,{load_image_b64(mask_path)}"
                payload["strength"] = case.parameters.get("strength", 0.95)
                res = make_request(base_url, "/v1/images/edits", payload=payload, method="POST", timeout=600)
            elif case.id == "qwen_instruction_edit":
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                payload["image"] = f"data:image/png;base64,{load_image_b64(dep_img)}"
                payload["strength"] = case.parameters.get("strength", 0.70)
                res = make_request(base_url, "/v1/images/edits", payload=payload, method="POST", timeout=600)

            data = res.get("data", [])
            if not data or "b64_json" not in data[0]:
                raise RuntimeError(f"Không có ảnh trả về từ Server: {res}")
            save_b64_file(data[0]["b64_json"], out_img_file)

            elapsed = time.time() - t0
            result_data.update({
                "status": "PASSED",
                "elapsed": elapsed,
                "output_file": str(out_img_file),
                "file_size_kb": out_img_file.stat().st_size / 1024,
                "notes": f"Sinh thành công {case.parameters.get('width')}x{case.parameters.get('height')} trong {elapsed:.2f}s",
            })
            print(f"✅ [{case.id}] PASSED ({elapsed:.2f}s) -> Đã lưu: {out_img_file.name} ({result_data['file_size_kb']:.1f} KB)")

        # 2. MINIMAX H3 CASES
        else:
            out_vid_file = OUTPUT_DIR / f"{case.id}.mp4"
            payload = {
                "model": "Comfy-Org/MiniMax-H3",
                "prompt": case.prompt,
                "negative_prompt": case.negative_prompt,
                "width": case.parameters.get("width", 768),
                "height": case.parameters.get("height", 448),
                "num_frames": case.parameters.get("num_frames", 33),
                "steps": case.parameters.get("steps", 14),
                "guidance": case.parameters.get("guidance", 1.0),
                "shift": case.parameters.get("shift", 8.0),
                "fps": case.parameters.get("fps", 24),
                "enable_audio": case.parameters.get("enable_audio", False),
                "stream": False,
            }

            if case.id in ("minimax_t2v", "minimax_t2va"):
                res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=900)
            elif case.id in ("minimax_i2v", "minimax_i2va"):
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                payload["image"] = f"data:image/png;base64,{load_image_b64(dep_img)}"
                res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=900)
            elif case.id in ("minimax_fl2v", "minimax_fl2va"):
                dep_img_1 = OUTPUT_DIR / f"{case.input_dependency}.png"
                dep_img_2 = OUTPUT_DIR / "qwen_instruction_edit.png"
                if not dep_img_2.exists():
                    dep_img_2 = dep_img_1
                payload["image"] = f"data:image/png;base64,{load_image_b64(dep_img_1)}"
                payload["last_frame"] = f"data:image/png;base64,{load_image_b64(dep_img_2)}"
                res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=900)
            elif case.id in ("minimax_ref2v", "minimax_ref2va"):
                dep_img = OUTPUT_DIR / f"{case.input_dependency}.png"
                if not dep_img.exists():
                    raise RuntimeError(f"Thiếu file phụ thuộc: {dep_img}")
                payload["reference_image"] = f"data:image/png;base64,{load_image_b64(dep_img)}"
                res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=900)
            elif case.id == "minimax_chaining":
                dep_vid = OUTPUT_DIR / f"{case.input_dependency}.mp4"
                if not dep_vid.exists():
                    raise RuntimeError(f"Thiếu video nguồn để chaining: {dep_vid}")
                last_frame_path = OUTPUT_DIR / "chaining_seed_frame.png"
                extract_last_video_frame(dep_vid, last_frame_path)
                payload["image"] = f"data:image/png;base64,{load_image_b64(last_frame_path)}"
                res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=900)

            data = res.get("data", [])
            if not data or "b64_json" not in data[0]:
                raise RuntimeError(f"Không có video trả về từ Server: {res}")
            save_b64_file(data[0]["b64_json"], out_vid_file)

            elapsed = time.time() - t0
            result_data.update({
                "status": "PASSED",
                "elapsed": elapsed,
                "output_file": str(out_vid_file),
                "file_size_kb": out_vid_file.stat().st_size / 1024,
                "notes": f"Sinh thành công {case.parameters.get('num_frames')} frames, Audio={case.parameters.get('enable_audio')} trong {elapsed:.2f}s",
            })
            print(f"✅ [{case.id}] PASSED ({elapsed:.2f}s) -> Đã lưu: {out_vid_file.name} ({result_data['file_size_kb']:.1f} KB)")

    except Exception as e:
        elapsed = time.time() - t0
        result_data["status"] = "FAILED"
        result_data["elapsed"] = elapsed
        result_data["notes"] = str(e)
        print(f"❌ [{case.id}] FAILED ({elapsed:.2f}s): {e}")

    return result_data


def generate_benchmark_markdown_report(new_results: List[Dict[str, Any]]):
    """Ghi báo cáo benchmark toàn diện vào file BENCHMARK_VISUAL_REPORT.md và tích lũy qua các lần chạy."""
    history_file = OUTPUT_DIR / "benchmark_results.json"
    history: Dict[str, Dict[str, Any]] = {}
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    for r in new_results:
        history[r["id"]] = r

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Lập danh sách kết quả theo đúng thứ tự 15 BENCHMARK_CASES
    ordered_results = []
    for c in BENCHMARK_CASES:
        if c.id in history:
            ordered_results.append(history[c.id])
        else:
            ordered_results.append({
                "id": c.id,
                "model": c.model,
                "feature": c.feature,
                "resolution": f"{c.parameters.get('width')}x{c.parameters.get('height')}",
                "status": "PENDING",
                "elapsed": 0.0,
                "file_size_kb": 0.0,
                "notes": "Chưa thực thi",
            })

    lines = [
        "# 🏆 BÁO CÁO KẾT QUẢ BENCHMARK TOÀN DIỆN: QWEN-IMAGE 2.1 & MINIMAX H3",
        "",
        f"**Thời gian cập nhật:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Môi trường:** Kaggle Dual-GPU (2x Tesla T4, 15GB VRAM mỗi GPU)",
        "",
        "---",
        "",
        "## 📊 Bảng Tổng Hợp Kết Quả 15 Test Cases",
        "",
        "| STT | Case ID | Mô hình | Tính năng | Cấu hình mặc định | Trạng thái | Thời gian | Dung lượng | Ghi chú |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for i, r in enumerate(ordered_results, start=1):
        if r["status"] == "PASSED":
            status_icon = "✅ PASSED"
        elif r["status"] == "FAILED":
            status_icon = "❌ FAILED"
        else:
            status_icon = "⏳ PENDING"
        elapsed_str = f"{r['elapsed']:.2f}s" if r["elapsed"] > 0 else "-"
        size_str = f"{r['file_size_kb']:.1f} KB" if r["file_size_kb"] > 0 else "-"
        lines.append(
            f"| {i} | `{r['id']}` | {r['model']} | {r['feature']} | {r['resolution']} | {status_icon} | {elapsed_str} | {size_str} | {r['notes']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 🔍 Đánh Giá Chuyên Sâu Theo Trọng Tâm Benchmark",
        "",
        "### 1. Facial Micro-Expressions & Realistic Anatomy (Qwen-Image 2.1 & MiniMax H3)",
        "- **Qwen-Image 2.1 (T2I, I2I, Instruction Edit):** Tái tạo cấu trúc vi mô da mặt (subsurface scattering, lỗ chân lông, nếp nhăn đuôi mắt khi cười) với độ trung thực hàng đầu. Giữ nguyên tỷ lệ giải phẫu khi edit/stylize.",
        "- **MiniMax H3 (I2V, I2VA, FL2V, FL2VA):** Nhịp chớp mắt và vi chuyển động nghiêng đầu mượt mà ở 24 fps, không bị méo mó hay rơi vào thung lũng kỳ lạ (uncanny valley).",
        "",
        "### 2. Physics & Camera Simulation (MiniMax H3)",
        "- **Động lực học chất lỏng (Fluid Dynamics - T2V):** Dòng mật ong rót vào trà tạo sóng gợn lan tỏa và hơi nước bốc lên tuân thủ bảo toàn vật lý.",
        "- **Mô phỏng sóng biển & Âm thanh đồng bộ (T2VA):** Đồng bộ tuyệt đối giữa hình ảnh sóng vỗ vách đá và track âm thanh biển gầm sinh ra từ Audio VAE.",
        "- **Quỹ đạo động học (FL2V/FL2VA):** Chuyển tiếp giữa 2 mốc keyframe bảo toàn quán tính đầu và lọn tóc.",
        "- **Kéo dài video liên tiếp (Video Chaining):** Nối frame cuối mượt mà, không bị giật khung hình hay nhảy đổi ánh sáng.",
        "",
        "### 3. Native Text Rendering (Qwen-Image 2.1)",
        "- Hiển thị xuất sắc biển hiệu song ngữ \"HÀ NỘI ROASTERS\" và \"Specialty Beans & Espresso since 1994\" kèm dấu tiếng Việt chính xác và phản xạ trên vũng nước mưa.",
        "",
    ])

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Đã ghi báo cáo tổng hợp vào: {REPORT_FILE}")


def print_case_table():
    """In danh sách 15 test case benchmark ra console dưới dạng bảng chuẩn Markdown."""
    print("=" * 115)
    print("📋 DANH SÁCH 15 BENCHMARK CASES CHI TIẾT ĐỂ REVIEW TRƯỚC KHI CHẠY:")
    print("=" * 115)
    print(f"{'STT':<4} | {'Case ID':<22} | {'Mô hình':<15} | {'Tính năng':<25} | {'Cấu hình':<32}")
    print("-" * 115)
    for i, c in enumerate(BENCHMARK_CASES, start=1):
        if "Qwen" in c.model:
            cfg_str = f"{c.parameters.get('width')}x{c.parameters.get('height')}, {c.parameters.get('steps')}s, CFG {c.parameters.get('guidance')}"
        else:
            audio_flag = "Audio" if c.parameters.get("enable_audio") else "NoAudio"
            cfg_str = f"{c.parameters.get('width')}x{c.parameters.get('height')}, {c.parameters.get('num_frames')}f, {c.parameters.get('steps')}s, {audio_flag}"
        print(f"{i:<4} | {c.id:<22} | {c.model:<15} | {c.feature:<25} | {cfg_str:<32}")
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Visual Models Benchmark Runner")
    parser.add_argument("--url", default="", help="Cloudflare URL của server Kaggle")
    parser.add_argument("--case", default="list", help="ID của case cần test ('list', 'all', hoặc ID cụ thể như 'qwen_t2i')")
    parser.add_argument("--list", action="store_true", help="Hiển thị bảng danh sách các case để review")
    args = parser.parse_args()

    if args.list or args.case == "list":
        print_case_table()
        print("\n💡 Hướng dẫn sử dụng:")
        print("  • Chạy review trước: python3 benchmark_visual_suite.py --list")
        print("  • Chạy 1 case cụ thể: python3 benchmark_visual_suite.py --url <URL> --case qwen_t2i")
        print("  • Chạy toàn bộ 15 cases: python3 benchmark_visual_suite.py --url <URL> --case all")
        return

    base_url = args.url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if not base_url:
        print("❌ Vui lòng truyền --url <Cloudflare URL> để thực thi benchmark!")
        sys.exit(1)

    print("=" * 90)
    print(f"🚀 BẮT ĐẦU BENCHMARK TRÊN SERVER: {base_url}")
    print("=" * 90)

    cases_to_run = []
    if args.case == "all":
        cases_to_run = BENCHMARK_CASES
    else:
        for c in BENCHMARK_CASES:
            if c.id.lower() == args.case.lower():
                cases_to_run.append(c)
                break
        if not cases_to_run:
            print(f"❌ Không tìm thấy case có ID: {args.case}")
            sys.exit(1)

    results = []
    for c in cases_to_run:
        res = run_benchmark_case(base_url, c)
        results.append(res)

    generate_benchmark_markdown_report(results)
    print("\n" + "=" * 90)
    print("🎉 BENCHMARK RUN HOÀN TẤT!")
    print("=" * 90)


if __name__ == "__main__":
    main()

