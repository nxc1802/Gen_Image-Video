"""
services/decision/launcher/run_decision_service.py
CLI runner for the Decision Routing Engine.
"""
import argparse
import uvicorn
from services.decision.src.config import load_decision_config


def main():
    parser = argparse.ArgumentParser(description="Run Factory v4 Decision Routing Engine")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on (default: 8082)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading")
    args = parser.parse_args()

    config = load_decision_config()
    port = args.port or config.port
    host = args.host or config.host

    print(f"🚀 Starting Factory v4 Decision Engine on {host}:{port} (Domain: {config.static_domain})")
    uvicorn.run("services.decision.src.app:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
