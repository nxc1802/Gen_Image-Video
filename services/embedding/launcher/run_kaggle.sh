#!/usr/bin/env bash
# run_kaggle.sh - Launch Embedding & Rerank Service on GPU 1 in Kaggle/Colab dual T4
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PORT=8083
export FACTORY_SERVICE_ID=embed-service

echo "=== Initializing Factory v4 Embedding Daemon on GPU 1 (Port 8083) ==="
python3 -m services.embedding.launcher.run_embedding_service --port 8083 --host 0.0.0.0
