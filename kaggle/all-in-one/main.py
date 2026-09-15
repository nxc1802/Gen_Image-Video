"""
🚀 Kaggle All-in-One Studio: Main Entrypoint
Điều phối khởi động hệ sinh thái AI:
1. Pre-warm Audio Core (Whisper Full FP16 + Kokoro Full FP16 trên GPU 0).
2. Khởi động Supabase Queue Worker daemon.
3. Kích hoạt Cloudflare Quick Tunnel (xuất Public URL).
4. Khởi chạy FastAPI Server chuẩn OpenAI.
"""

import argparse
import logging
import sys
import threading
import time
import uvicorn

from config import HOST, PORT, ENABLE_CLOUDFLARE, ENABLE_SUPABASE
from core.memory_manager import get_memory_manager
from server.app import create_app
from tunnel.cloudflare import start_cloudflare_tunnel
from tunnel.supabase_queue import start_supabase_worker

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
    parser.add_argument("--no-supabase", action="store_true", help="Tắt Supabase Queue worker")
    parser.add_argument("--skip-warmup", action="store_true", help="Bỏ qua bước pre-load weights")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("🎨 KHỞI ĐỘNG KAGGLE ALL-IN-ONE AI STUDIO (2x TESLA T4)")
    print("   • VLM       : Qwen 26B (4-bit Dual-GPU)")
    print("   • Audio STT : Whisper-large-v3-turbo (BẢN FULL FP16 trên GPU 0)")
    print("   • Audio TTS : Kokoro-82M (BẢN FULL FP16 trên GPU 0)")
    print("   • GenImage  : FLUX.1-schnell (4-bit NF4 trên GPU 1)")
    print("   • GenVideo  : Wan2.1-1.3B (Text-to-Video trên GPU 1)")
    print("=" * 75 + "\n")

    mem = get_memory_manager()
    logger.info(f"📊 Trạng thái VRAM ban đầu: {mem.report_vram()}")

    # 1. Khởi động Supabase Queue Worker (nếu bật)
    if ENABLE_SUPABASE and not args.no_supabase:
        start_supabase_worker()

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
