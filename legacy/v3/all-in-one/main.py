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
import os
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
    parser = argparse.ArgumentParser(description="🚀 Kaggle Dedicated AI Studio Entrypoint")
    parser.add_argument(
        "--mode",
        default=os.environ.get("STUDIO_MODE", "speech"),
        choices=["speech", "vlm", "image_video", "visual", "embedding"],
        help="Dedicated Studio mode: speech | vlm | image_video | embedding",
    )
    parser.add_argument("--host", default=HOST, help="Host để bind server")
    parser.add_argument("--port", type=int, default=PORT, help="Port cho FastAPI")
    parser.add_argument("--no-tunnel", action="store_true", help="Tắt Cloudflare Quick Tunnel")
    parser.add_argument("--skip-warmup", action="store_true", help="Bỏ qua bước pre-load weights")

    args = parser.parse_args()
    mode = str(args.mode).strip().lower()
    if mode in ("visual",):
        mode = "image_video"
    if mode not in ("speech", "vlm", "image_video", "embedding"):
        mode = "speech"
    os.environ["STUDIO_MODE"] = mode

    vlm_cfg = MODELS_CONFIG.get("vlm", {})
    img_cfg = MODELS_CONFIG.get("image", {})
    vid_cfg = MODELS_CONFIG.get("video", {})
    stt_cfg = MODELS_CONFIG.get("stt", {})
    tts_cfg = MODELS_CONFIG.get("tts", {})

    print("\n" + "=" * 78)
    if mode == "speech":
        print("🎙️ KHỞI ĐỘNG KAGGLE SPEECH AI STUDIO (DEDICATED SPEECH MODE)")
        print(f"   • Audio STT : {stt_cfg.get('id', 'N/A')} [auto-preload: GPU VRAM]")
        print(f"   • Audio TTS : {tts_cfg.get('id', 'N/A')} [auto-preload: GPU VRAM]")
        print(f"   • XTTS-v2   : {tts_cfg.get('xtts_id', 'thivux/XTTS-v2-vietnamse')} [auto-preload: GPU VRAM]")
    elif mode == "vlm":
        print("👁️ KHỞI ĐỘNG KAGGLE VLM AI STUDIO (DEDICATED VLM MODE)")
        print(f"   • VLM       : {vlm_cfg.get('id', 'N/A')} [auto-preload: GPU VRAM]")
    elif mode == "image_video":
        mm_cfg = MODELS_CONFIG.get("minimax", {})
        print("🖼️ KHỞI ĐỘNG KAGGLE IMAGE & VIDEO AI STUDIO (MULTI-GPU DEDICATED VISUAL MODE)")
        print(f"   • GPU 0 (cuda:0, port 8188) : Qwen-Image 2.1 [{img_cfg.get('id', 'N/A')}]")
        print(f"   • GPU 1 (cuda:1, port 8189) : MiniMax H3 [{mm_cfg.get('id', 'N/A')}] & Wan [{vid_cfg.get('id', 'N/A')}]")
    elif mode == "embedding":
        print("🧬 KHỞI ĐỘNG KAGGLE EMBEDDING & RERANK AI STUDIO (DEDICATED EMBEDDING MODE)")
        print("   • GPU 0 (cuda:0) - 3 Model Dòng Text : Qwen3-0.6B, Qwen3-4B, Qwen3-8B (Embed & Rerank)")
        print("   • GPU 1 (cuda:1) - 2 Model Dòng VL   : Qwen3-VL-2B, Qwen3-VL-8B (Embed & Rerank)")
        print("   • Framework                         : Official llama-server / llama_cpp (GGUF Q4_K_M)")
    print("=" * 78 + "\n")

    mem = get_memory_manager()
    registry = get_model_registry()
    logger.info(f"📊 Trạng thái VRAM ban đầu: {mem.report_vram()}")

    # 1. Khởi động Cloudflare Tunnel trong luồng riêng ngay lập tức
    if ENABLE_CLOUDFLARE and not args.no_tunnel:
        def tunnel_thread():
            time.sleep(1.0)  # Đợi Uvicorn bắt đầu lắng nghe
            start_cloudflare_tunnel(port=args.port)

        t = threading.Thread(target=tunnel_thread, daemon=True)
        t.start()

    # 2. Khởi động Audio Service & Auto-preload nếu chạy Speech Mode
    if mode == "speech":
        try:
            from audio.audio_manager import get_audio_service_manager
            audio_mgr = get_audio_service_manager()
            def audio_preload_thread():
                time.sleep(1.0)
                audio_mgr.preload_all_models()
            threading.Thread(target=audio_preload_thread, daemon=True).start()
        except Exception as e:
            logger.warning(f"Audio preload thread warning: {e}")

    # 3. Khởi động Embedding Service & Auto-preload nếu chạy Embedding Mode
    if mode == "embedding":
        try:
            from embedding.embedding_service import get_embedding_service_manager
            emb_mgr = get_embedding_service_manager()
            def emb_preload_thread():
                time.sleep(1.0)
                emb_mgr.preload_all_models()
            threading.Thread(target=emb_preload_thread, daemon=True).start()
        except Exception as e:
            logger.warning(f"Embedding preload thread warning: {e}")

    # 4. Khởi động ComfyUI Service(s) nếu chạy Image/Video Mode
    if mode in ("image_video", "visual"):
        try:
            import torch
            from core.comfyui_manager import get_comfyui_manager
            def comfy_start_thread():
                time.sleep(1.0)
                # Khởi động ComfyUI trên GPU 0 (Qwen-Image)
                mgr0 = get_comfyui_manager(cuda_device=0)
                mgr0.start()
                # Khởi động ComfyUI trên GPU 1 (MiniMax H3 / Video) nếu có >= 2 GPUs
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    mgr1 = get_comfyui_manager(cuda_device=1)
                    mgr1.start()
            threading.Thread(target=comfy_start_thread, daemon=True).start()
        except Exception as e:
            logger.warning(f"ComfyUI startup thread warning: {e}")

    # 5. Warmup các mô hình preload khác (không chặn server)
    if not args.skip_warmup and mode in ("vlm", "image_video"):
        def warmup_thread():
            time.sleep(1.5)
            mem.warmup_models(registry)

        w_th = threading.Thread(target=warmup_thread, daemon=True)
        w_th.start()

    # 3. Tạo FastAPI App
    app = create_app()

    # 4. Chạy Uvicorn Server
    logger.info(f"🌐 Server bắt đầu lắng nghe tại http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
