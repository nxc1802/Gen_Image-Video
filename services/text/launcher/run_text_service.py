"""
services/text/launcher/run_text_service.py
CLI runner for the Text & VLM generation daemon.
"""
import argparse
import sys
import uvicorn
from services.text.src.config import load_text_config


def main():
    parser = argparse.ArgumentParser(description="Run Factory v4 Text Generation Daemon")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    args = parser.parse_args()

    config = load_text_config()
    port = args.port or config.port
    host = args.host or config.host

    print(f"🚀 Starting Factory v4 Text Daemon on {host}:{port} (Domain: {config.static_domain})")
    uvicorn.run("services.text.src.app:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
