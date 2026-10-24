#!/usr/bin/env python3
"""
🧪 Kaggle Visual Studio Automated Multi-Model Test Suite
================================================================================
Kiểm thử tuần tự toàn bộ các mô hình Visual Studio:
1. Qwen-Image 2.1 (T2I)
2. Wan2.2-TI2V-5B (I2V / T2V)
3. Wan2.2-I2V-A14B (MoE I2V)
4. Wan2.2-T2V-A14B (MoE T2V)
5. MiniMax H3 (T2VA / FL2VA native audio)
Tối ưu hóa tốc độ tối đa cho mỗi case kiểm thử (nhỏ resolution, ít steps).
"""

import argparse
import base64
import json
import os
import sys
import time
import socket
import urllib.request
import urllib.error
from pathlib import Path

# Đảm bảo phân giải tên miền Cloudflare Quick Tunnel không bị nghẽn cache DNS địa phương
_orig_getaddrinfo = socket.getaddrinfo
def _cf_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror:
        if "trycloudflare.com" in str(host):
            return _orig_getaddrinfo("104.16.230.132", port, *args, **kwargs)
        raise
socket.getaddrinfo = _cf_getaddrinfo

OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file():
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent / "kaggle" / "all-in-one" / ".env",
    ]
    for env_p in env_paths:
        if env_p.exists():
            with open(env_p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ or not os.environ[k]:
                            os.environ[k] = v


load_env_file()


def get_public_url(user: str = "") -> str:
    """Tự động lấy URL từ ntfy.sh nếu không truyền vào và kiểm tra /health."""
    topics = [
        f"studio-ai-url-{user}" if user else "",
        "studio-ai-url-cuongnguyen1802",
        "studio-ai-url-nguynxuncngde180528",
    ]
    for topic in [t for t in topics if t]:
        url_req = f"https://ntfy.sh/{topic}/raw?poll=1&since=30m"
        try:
            with urllib.request.urlopen(urllib.request.Request(url_req), timeout=4) as resp:
                content = resp.read().decode("utf-8").strip()
                if content:
                    for line in reversed(content.splitlines()):
                        if "trycloudflare.com" in line:
                            cand = line.strip().split("/v1")[0].rstrip("/")
                            try:
                                with urllib.request.urlopen(f"{cand}/health", timeout=3) as h_resp:
                                    if h_resp.status == 200:
                                        return cand
                            except Exception:
                                pass
        except Exception:
            pass
    return os.environ.get("API_BASE_URL", "")


def make_request(base_url: str, endpoint: str, payload: dict = None, method: str = "GET", timeout: int = 600) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    data_bytes = json.dumps(payload).encode("utf-8") if payload else None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('STUDIO_API_KEY', 'default-key')}",
        "User-Agent": "VisualTestSuite/1.0",
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


def test_health(base_url: str) -> bool:
    print(f"\n🩺 [1/6] Kiểm tra Health Check: {base_url}/health...")
    try:
        t0 = time.time()
        res = make_request(base_url, "/health", timeout=10)
        elapsed = time.time() - t0
        print(f"✅ Server Health: OK ({elapsed:.2f}s) - Status: {res.get('status')}, Mode: {res.get('studio_mode')}")

        catalog = res.get("catalog", {})
        print("   📋 Phân bổ GPU mô hình (Hardware Mapping):")
        for m_name in ["image", "minimax", "video"]:
            m_info = catalog.get(m_name, {})
            dev = m_info.get("device") or f"gpu_{m_info.get('gpu_device', 'N/A')}"
            print(f"      • {m_name.upper():<8}: {m_info.get('model_id', 'N/A')} -> Thiết bị: {dev} (Strategy: {m_info.get('device_strategy', 'single')})")

        comfy_diag = res.get("comfyui_diagnostics", {})
        print("   🎮 Trạng thái ComfyUI Engine(s):")
        if isinstance(comfy_diag, dict):
            for k, v in comfy_diag.items():
                if isinstance(v, dict):
                    print(f"      • {k.upper()}: running={v.get('is_running')}, device=cuda:{v.get('cuda_device')}, url={v.get('base_url')}, models_linked={v.get('linked_models_count')}")
                elif k == "is_running":
                    print(f"      • COMFYUI: running={comfy_diag.get('is_running')}, device=cuda:{comfy_diag.get('cuda_device')}, url={comfy_diag.get('base_url')}")
                    break

        vram = res.get("vram_status", {})
        gpus = vram.get("gpus", {})
        print("   ⚡ Tài nguyên VRAM & RAM:")
        for g_id, g_info in gpus.items():
            print(f"      • {g_id.upper()}: {g_info.get('name')} | Used: {g_info.get('hardware_used_gb', 0)}GB / Free: {g_info.get('hardware_free_gb', 0)}GB / Total: {g_info.get('total_gb', 0)}GB")
        ram = vram.get("ram", {})
        if ram:
            print(f"      • CPU RAM: Used: {ram.get('used_gb', 0)}GB ({ram.get('percent', 0)}%) / Free: {ram.get('available_gb', 0)}GB / Total: {ram.get('total_gb', 0)}GB")
        return True
    except Exception as e:
        print(f"❌ Server không phản hồi /health: {e}")
        return False


def test_qwen_image(base_url: str) -> str:
    print(f"\n🖼️ [2/6] Test 1: Qwen-Image 2.1 (T2I Fast Preview: 512x512, 4 steps)...")
    payload = {
        "model": "qwen_image_21",
        "prompt": "A modern minimalist glass coffee cup on a light wooden table, soft morning daylight, studio photo",
        "size": "512x512",
        "steps": 4,
        "guidance": 1.0,
        "stream": False,
    }
    t0 = time.time()
    res = make_request(base_url, "/v1/images/generations", payload=payload, method="POST", timeout=600)
    elapsed = time.time() - t0

    data = res.get("data", [])
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Không có dữ liệu ảnh trả về: {res}")

    b64_img = data[0]["b64_json"]
    out_file = OUTPUT_DIR / "test_qwen_image.png"
    with open(out_file, "wb") as f:
        f.write(base64.b64decode(b64_img))

    print(f"✅ Qwen-Image 2.1 thành công ({elapsed:.2f}s) -> Đã lưu: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return b64_img


def test_wan_ti2v_5b(base_url: str, conditioning_b64: str) -> str:
    print(f"\n🎬 [3/6] Test 2: Wan2.2-TI2V-5B (I2V: 416x240, 9 frames, 4 steps)...")
    payload = {
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "prompt": "Gentle steam rising from the hot coffee, smooth cinematic slow motion",
        "image": f"data:image/png;base64,{conditioning_b64}" if not conditioning_b64.startswith("data:") else conditioning_b64,
        "width": 416,
        "height": 240,
        "num_frames": 9,
        "steps": 4,
        "guidance": 1.0,
        "fps": 16,
        "stream": False,
    }
    t0 = time.time()
    res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=600)
    elapsed = time.time() - t0

    data = res.get("data", [])
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Không có video trả về: {res}")

    b64_vid = data[0]["b64_json"]
    out_file = OUTPUT_DIR / "test_wan_ti2v_5b.mp4"
    with open(out_file, "wb") as f:
        f.write(base64.b64decode(b64_vid))

    print(f"✅ Wan2.2-TI2V-5B thành công ({elapsed:.2f}s) -> Đã lưu: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return b64_vid


def test_wan_i2v_a14b(base_url: str, conditioning_b64: str) -> str:
    print(f"\n🎬 [4/6] Test 3: Wan2.2-I2V-A14B (MoE I2V: 416x240, 9 frames, 2+2 steps)...")
    payload = {
        "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "prompt": "Camera softly zooming into the glass, ambient lighting",
        "image": f"data:image/png;base64,{conditioning_b64}" if not conditioning_b64.startswith("data:") else conditioning_b64,
        "width": 416,
        "height": 240,
        "num_frames": 9,
        "high_steps": 2,
        "low_steps": 2,
        "guidance": 1.0,
        "fps": 16,
        "stream": False,
    }
    t0 = time.time()
    res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=600)
    elapsed = time.time() - t0

    data = res.get("data", [])
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Không có video trả về: {res}")

    b64_vid = data[0]["b64_json"]
    out_file = OUTPUT_DIR / "test_wan_i2v_a14b.mp4"
    with open(out_file, "wb") as f:
        f.write(base64.b64decode(b64_vid))

    print(f"✅ Wan2.2-I2V-A14B thành công ({elapsed:.2f}s) -> Đã lưu: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return b64_vid


def test_wan_t2v_a14b(base_url: str) -> str:
    print(f"\n🎬 [5/6] Test 4: Wan2.2-T2V-A14B (MoE T2V: 416x240, 9 frames, 4 steps)...")
    payload = {
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "prompt": "Golden sunrise above clouds, majestic aerial view, 4k cinematic film grain",
        "width": 416,
        "height": 240,
        "num_frames": 9,
        "steps": 4,
        "guidance": 1.0,
        "fps": 16,
        "stream": False,
    }
    t0 = time.time()
    res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=600)
    elapsed = time.time() - t0

    data = res.get("data", [])
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Không có video trả về: {res}")

    b64_vid = data[0]["b64_json"]
    out_file = OUTPUT_DIR / "test_wan_t2v_a14b.mp4"
    with open(out_file, "wb") as f:
        f.write(base64.b64decode(b64_vid))

    print(f"✅ Wan2.2-T2V-A14B thành công ({elapsed:.2f}s) -> Đã lưu: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return b64_vid


def test_minimax_h3(base_url: str) -> str:
    print(f"\n🎬 [6/6] Test 5: MiniMax H3 (T2VA + Native Stereo Audio: 416x256, 5 frames, 4 steps)...")
    payload = {
        "model": "Comfy-Org/MiniMax-H3",
        "prompt": "Glowing neon crystalline artifact floating in dark void, soft ambient synthwave drone sound",
        "width": 416,
        "height": 256,
        "num_frames": 5,
        "steps": 4,
        "guidance": 1.0,
        "fps": 24,
        "stream": False,
    }
    t0 = time.time()
    res = make_request(base_url, "/v1/videos/generations", payload=payload, method="POST", timeout=600)
    elapsed = time.time() - t0

    data = res.get("data", [])
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Không có video trả về: {res}")

    b64_vid = data[0]["b64_json"]
    out_file = OUTPUT_DIR / "test_minimax_h3.mp4"
    with open(out_file, "wb") as f:
        f.write(base64.b64decode(b64_vid))

    print(f"✅ MiniMax H3 thành công ({elapsed:.2f}s) -> Đã lưu: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return b64_vid


def test_anime_pipeline(base_url: str, quality: str = "default", reuse_image: bool = True) -> tuple[str, str]:
    """
    Use case: Prompt -> Qwen-Image 2.1 (GPU 0) -> Image -> MiniMax H3 (GPU 1) -> Video.
    1. Sinh ảnh nhân vật anime nữ tuyệt đẹp bằng Qwen-Image 2.1 trên GPU 0.
    2. Chuyển ảnh đó sang MiniMax H3 (I2V / FL2VA) trên GPU 1 để tạo video animation chuyển động tự nhiên.
    """
    is_default = (quality.lower() == "default")
    print("\n" + "=" * 76)
    print(f"🌸 BẮT ĐẦU PIPELINE END-TO-END DUAL-GPU: ANIME GIRL IMAGE -> VIDEO [Chế độ: {'MẶC ĐỊNH (DEFAULT QUALITY)' if is_default else 'TỐI ƯU TỐC ĐỘ (FAST)'}]")
    print("   • GPU 0 : Qwen-Image 2.1 (Tạo ảnh anime chất lượng cao)")
    print("   • GPU 1 : MiniMax H3 (Tạo animation I2V + Stereo Audio tự nhiên)")
    print("=" * 76)

    out_img_file = OUTPUT_DIR / "test_anime_girl_qwen.png"
    elapsed_img = 0.0

    # Kiểm tra xem có thể tái sử dụng ảnh anime 1024x1024 đã sinh ở Bước 1 trước đó không
    if reuse_image and out_img_file.exists() and out_img_file.stat().st_size > 100 * 1024:
        print(f"\n[1/2] 🎨 Bước 1: Tái sử dụng ảnh Anime Girl có sẵn ({out_img_file.stat().st_size / 1024:.1f} KB): {out_img_file.name}")
        with open(out_img_file, "rb") as f:
            b64_anime_img = base64.b64encode(f.read()).decode("utf-8")
        print(f"✅ [GPU 0] Tái sử dụng ảnh Anime Girl thành công!")
    else:
        # BƯỚC 1: Sinh ảnh Anime Girl với Qwen-Image 2.1 trên GPU 0
        anime_image_prompt = (
            "masterpiece, highest quality, breathtaking anime girl with long flowing iridescent silver hair, "
            "delicate expressive violet eyes, gentle sweet smile, wearing an elegant pastel blue kimono dress "
            "with intricate cherry blossom embroidery, soft morning sunlight, cherry blossom petals floating around, "
            "8k anime art, highly detailed face and eyes, stunning composition, Makoto Shinkai style"
        )
        img_size = "1024x1024" if is_default else "512x512"
        img_steps = 20 if is_default else 4
        print(f"\n[1/2] 🎨 Bước 1: Qwen-Image 2.1 trên GPU 0 sinh ảnh Anime Girl ({img_size}, {img_steps} steps)...")
        print(f"      Prompt: {anime_image_prompt[:90]}...")
        payload_img = {
            "model": "qwen_image_21",
            "prompt": anime_image_prompt,
            "size": img_size,
            "steps": img_steps,
            "guidance": 1.0,
            "stream": False,
        }
        t0 = time.time()
        res_img = make_request(base_url, "/v1/images/generations", payload=payload_img, method="POST", timeout=600)
        elapsed_img = time.time() - t0

        data_img = res_img.get("data", [])
        if not data_img or "b64_json" not in data_img[0]:
            raise RuntimeError(f"Không có dữ liệu ảnh trả về: {res_img}")

        b64_anime_img = data_img[0]["b64_json"]
        with open(out_img_file, "wb") as f:
            f.write(base64.b64decode(b64_anime_img))

        print(f"✅ [GPU 0] Qwen-Image 2.1 thành công ({elapsed_img:.2f}s) -> Đã lưu: {out_img_file} ({out_img_file.stat().st_size / 1024:.1f} KB)")

    # BƯỚC 2: Sinh Video Animation với MiniMax H3 trên GPU 1 (I2V / FL2VA)
    anime_video_prompt = (
        "The gorgeous anime girl smiles warmly and blinks her violet eyes, her long silver hair and silk kimono sleeves "
        "fluttering softly in a gentle spring breeze, cherry blossom petals drifting around her, smooth natural head turning and hair movement, "
        "cinematic lighting, serene and graceful animation, soft gentle ambient acoustic melody"
    )
    vid_frames = 17 if is_default else 5
    vid_steps = 14 if is_default else 4
    vid_width = 416
    # Lưu ý: MiniMax H3 VAE downscale 16x * DiT patch 2x = bắt buộc width & height chia hết cho 32!
    vid_height = 256
    print(f"\n[2/2] 🎬 Bước 2: MiniMax H3 trên GPU 1 sinh Video Animation (I2V + Audio: {vid_width}x{vid_height}, {vid_frames} frames, {vid_steps} steps)...")
    print(f"      Prompt: {anime_video_prompt[:90]}...")
    payload_vid = {
        "model": "Comfy-Org/MiniMax-H3",
        "prompt": anime_video_prompt,
        "image": f"data:image/png;base64,{b64_anime_img}",
        "width": vid_width,
        "height": vid_height,
        "num_frames": vid_frames,
        "steps": vid_steps,
        "guidance": 1.0,
        "fps": 24,
        "stream": False,
    }
    t1 = time.time()
    res_vid = make_request(base_url, "/v1/videos/generations", payload=payload_vid, method="POST", timeout=900)
    elapsed_vid = time.time() - t1

    data_vid = res_vid.get("data", [])
    if not data_vid or "b64_json" not in data_vid[0]:
        raise RuntimeError(f"Không có video trả về: {res_vid}")

    b64_vid = data_vid[0]["b64_json"]
    out_vid_file = OUTPUT_DIR / "test_anime_girl_minimax_i2v.mp4"
    with open(out_vid_file, "wb") as f:
        f.write(base64.b64decode(b64_vid))

    print(f"✅ [GPU 1] MiniMax H3 I2V thành công ({elapsed_vid:.2f}s) -> Đã lưu: {out_vid_file} ({out_vid_file.stat().st_size / 1024:.1f} KB)")
    print("\n" + "=" * 76)
    print(f"🎉 PIPELINE DUAL-GPU HOÀN TẤT THÀNH CÔNG TRONG {elapsed_img + elapsed_vid:.2f}s!")
    print(f"   🖼️ Ảnh Anime Girl (GPU 0): {out_img_file}")
    print(f"   🎬 Video Anime Girl (GPU 1): {out_vid_file}")
    print("=" * 76)
    return str(out_img_file), str(out_vid_file)


def shutdown_server(base_url: str) -> bool:
    print(f"\n🛑 [Shutdown] Gửi lệnh dừng server vĩnh viễn qua API (/v1/admin/shutdown?confirm=1)...")
    try:
        res = make_request(base_url, "/v1/admin/shutdown?confirm=1", method="POST", timeout=10)
        print(f"✅ Đã gửi lệnh shutdown: {res.get('status')} - {res.get('message')}")
        return True
    except Exception as e:
        print(f"⚠️ Shutdown request: {e}")
        return False


def wait_for_session_complete(slug: str = "kaggle-all-in-one-studio", timeout: int = 180) -> bool:
    print(f"⏳ Đang đợi Kaggle session dừng hoàn toàn (tối đa {timeout}s)...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                st = api.kernels_status(f"cuongnguyen1802/{slug}")
                status = str(st.status)
                print(f"[{time.strftime('%H:%M:%S')}] Trạng thái kernel ({slug}): {status}")
                if "RUNNING" not in status and "ACKNOWLEDGED" not in status:
                    print("✅ Session đã dừng hoàn tất!")
                    return True
            except Exception:
                pass
            time.sleep(5)
    except Exception as e:
        print(f"Lỗi kiểm tra session: {e}")
    return False


def get_conditioning_image_b64(base_url: str = "") -> str:
    for candidate in [
        OUTPUT_DIR / "test_anime_girl_qwen.png",
        OUTPUT_DIR / "test_qwen_image.png",
        OUTPUT_DIR / "1_qwen_image_2_1_output.png",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"📷 Sử dụng ảnh conditioning có sẵn: {candidate.name}")
            with open(candidate, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    if base_url:
        return test_qwen_image(base_url)
    raise RuntimeError("Không tìm thấy ảnh mẫu để làm conditioning cho I2V!")


def main():
    parser = argparse.ArgumentParser(description="Visual Models Test Suite")
    parser.add_argument("--url", default="", help="Base URL of Kaggle Studio (e.g. https://xxx.trycloudflare.com)")
    parser.add_argument("--model", choices=["all", "qwen", "wan_ti2v", "wan_i2v", "wan_t2v", "minimax", "anime"], default="anime")
    parser.add_argument("--quality", choices=["default", "fast"], default="default", help="Chất lượng kiểm thử: default (mặc định) | fast (tối ưu tốc độ)")
    parser.add_argument("--force-image", action="store_true", help="Bắt buộc sinh lại ảnh Bước 1 kể cả khi đã có sẵn")
    parser.add_argument("--shutdown-after-test", action="store_true", help="Dừng hẳn server sau khi test xong")
    args = parser.parse_args()

    base_url = args.url or get_public_url()
    if not base_url:
        print("❌ Không tìm thấy URL Cloudflare hợp lệ! Hãy truyền --url hoặc đảm bảo kernel đang chạy.")
        sys.exit(1)

    print("=" * 76)
    print("🚀 BẮT ĐẦU KIỂM THỬ VISUAL STUDIO TRÊN KAGGLE")
    print(f"🌐 Target URL: {base_url}")
    print("=" * 76)

    if not test_health(base_url):
        sys.exit(1)

    if args.model == "anime":
        test_anime_pipeline(base_url, quality=args.quality, reuse_image=(not args.force_image))
    else:
        img_b64 = None
        if args.model in ("all", "qwen"):
            img_b64 = test_qwen_image(base_url)

        if args.model in ("all", "wan_ti2v"):
            if not img_b64:
                img_b64 = get_conditioning_image_b64(base_url if args.model == "all" else "")
            test_wan_ti2v_5b(base_url, img_b64)

        if args.model in ("all", "wan_i2v"):
            if not img_b64:
                img_b64 = get_conditioning_image_b64(base_url if args.model == "all" else "")
            test_wan_i2v_a14b(base_url, img_b64)

        if args.model in ("all", "wan_t2v"):
            test_wan_t2v_a14b(base_url)

        if args.model in ("all", "minimax"):
            test_minimax_h3(base_url)

    print("\n" + "=" * 76)
    print("🎉 KIỂM THỬ MODEL ĐÃ HOÀN TẤT THÀNH CÔNG VỚI 0 LỖI OOM!")
    print("=" * 76)

    if args.shutdown_after_test:
        shutdown_server(base_url)
        wait_for_session_complete()


if __name__ == "__main__":
    main()
