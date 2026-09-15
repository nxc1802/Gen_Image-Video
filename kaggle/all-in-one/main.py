"""
🚀 Kaggle All-in-One Studio: Main Entrypoint
Điều phối khởi động hệ sinh thái AI với cơ chế Adaptive Dynamic Balancing:
1. Warmup các mô hình có preload: true (gpu hoặc cpu) theo khai báo trong models.yaml.
2. Ghim các mô hình always_active (Whisper, Kokoro, VLM nhỏ) an toàn trong VRAM.
3. Kích hoạt Cloudflare Quick Tunnel (xuất Public HTTPS URL).
4. Khởi chạy FastAPI Server chuẩn OpenAI REST API.
"""

import argparse
import logging
import sys
import threading
import time
import uvicorn

from config import HOST, PORT, ENABLE_CLOUDFLARE, MODELS_CONFIG
from core.memory_manager import get_memory_manager
from core.model_registry import get_model_registry
from server.app import create_app
from tunnel.cloudflare import start_cloudflare_tunnel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StudioMain")


def main():
    parser = argparse.ArgumentParser(description="🎨 Kaggle All-in-One AI Studio Entrypoint")
    parser.add_argument("--host", default=HOST, help="Host để bind server")
    parser.add_argument("--port", type=int, default=PORT, help="Port cho FastAPI")
    parser.add_argument("--no-tunnel", action="store_true", help="Tắt Cloudflare Quick Tunnel")
    parser.add_argument("--skip-warmup", action="store_true", help="Bỏ qua bước pre-load weights")

    args = parser.parse_args()

    vlm_cfg = MODELS_CONFIG.get("vlm", {})
    img_cfg = MODELS_CONFIG.get("image", {})
    vid_cfg = MODELS_CONFIG.get("video", {})
    stt_cfg = MODELS_CONFIG.get("stt", {})
    tts_cfg = MODELS_CONFIG.get("tts", {})

    print("\n" + "=" * 78)
    print("🎨 KHỞI ĐỘNG KAGGLE ALL-IN-ONE AI STUDIO (ADAPTIVE DYNAMIC BALANCING)")
    print(f"   • VLM       : {vlm_cfg.get('id', 'N/A')} [{vlm_cfg.get('lifecycle', 'always_active')}]")
    print(f"   • Audio STT : {stt_cfg.get('id', 'N/A')} [{stt_cfg.get('lifecycle', 'always_active')}]")
    print(f"   • Audio TTS : {tts_cfg.get('id', 'N/A')} [{tts_cfg.get('lifecycle', 'always_active')}]")
    print(f"   • GenImage  : {img_cfg.get('id', 'N/A')} [{img_cfg.get('lifecycle', 'dynamic_switch')}]")
    print(f"   • GenVideo  : {vid_cfg.get('id', 'N/A')} [{vid_cfg.get('lifecycle', 'dynamic_switch')}]")
    print("=" * 78 + "\n")

    mem = get_memory_manager()
    registry = get_model_registry()
    logger.info(f"📊 Trạng thái VRAM ban đầu: {mem.report_vram()}")

    # 1. Warmup các mô hình preload nếu không bị bỏ qua
    if not args.skip_warmup:
        mem.warmup_models(registry)

    # 2. Khởi động Cloudflare Tunnel trong luồng riêng để không chặn server
    if ENABLE_CLOUDFLARE and not args.no_tunnel:
        def tunnel_thread():
            time.sleep(2.0)  # Đợi Uvicorn bind port
            start_cloudflare_tunnel(port=args.port)

        t = threading.Thread(target=tunnel_thread, daemon=True)
        t.start()

    # 3. Tạo FastAPI App
    app = create_app()

    # 4. Chạy Uvicorn Server
    logger.info(f"🌐 Server bắt đầu lắng nghe tại http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
