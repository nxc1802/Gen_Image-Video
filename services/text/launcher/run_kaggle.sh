#!/usr/bin/env bash
# run_kaggle.sh - Launch Text & VLM Service on GPU 0 in Kaggle/Colab dual T4
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PORT=8080
export FACTORY_SERVICE_ID=text-service

echo "=== Initializing Factory v4 Text Daemon on GPU 0 (Port 8080) ==="
python3 -m services.text.launcher.run_text_service --port 8080 --host 0.0.0.0
