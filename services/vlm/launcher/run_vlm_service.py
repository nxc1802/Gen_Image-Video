"""
services/vlm/launcher/run_vlm_service.py
CLI runner for the Vision-Language Model daemon.
"""
import argparse
import uvicorn
from services.vlm.src.config import load_vlm_config


def main():
    parser = argparse.ArgumentParser(description="Run Factory v4 Vision-Language Model Daemon")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on (default: 8081)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading")
    args = parser.parse_args()

    config = load_vlm_config()
    port = args.port or config.port
    host = args.host or config.host

    print(f"🚀 Starting Factory v4 VLM Daemon on {host}:{port} (Domain: {config.static_domain})")
    uvicorn.run("services.vlm.src.app:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
