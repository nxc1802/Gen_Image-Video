"""
Native vLLM launcher for Qwen3 embeddings.
"""
import argparse
import os
import shutil

from services.embedding.src.config import load_embedding_config


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' is not installed or not on PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Factory v4 Embedding subsystem with native vLLM")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()

    config = load_embedding_config()
    host = args.host or config.host
    port = args.port or config.port
    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required before exposing the embedding service.")

    _require_binary("vllm")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.hardware.gpu_device_index)

    cmd = [
        "vllm", "serve", config.model.embedding_name,
        "--host", host,
        "--port", str(port),
        "--served-model-name", config.model.embedding_alias,
        "--runner", "pooling",
        "--max-model-len", str(config.model.max_context_tokens),
        "--api-key", api_key,
        "--allowed-origins", os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000"),
        "--allowed-methods", "*",
        "--allowed-headers", "*",
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
