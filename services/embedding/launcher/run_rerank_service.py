"""
Native vLLM launcher for the Qwen3 reranker.
"""
import argparse
import json
import os
import shutil

from services.embedding.src.config import load_embedding_config


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' is not installed or not on PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Factory v4 Rerank subsystem with native vLLM")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RERANK_SERVICE_PORT", "8084")))
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    args = parser.parse_args()

    config = load_embedding_config()
    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required before exposing the rerank service.")

    _require_binary("vllm")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.hardware.gpu_device_index)

    overrides = {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    cmd = [
        "vllm", "serve", config.model.rerank_name,
        "--host", args.host,
        "--port", str(args.port),
        "--served-model-name", config.model.rerank_alias,
        "--runner", "pooling",
        "--max-model-len", str(config.model.max_context_tokens),
        "--hf-overrides", json.dumps(overrides, separators=(",", ":")),
        "--api-key", api_key,
        "--allowed-origins", os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000"),
        "--allowed-methods", "*",
        "--allowed-headers", "*",
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
