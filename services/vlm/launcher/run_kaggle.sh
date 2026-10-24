#!/usr/bin/env bash
# run_kaggle.sh - Launch VLM Service on GPU 1 in Kaggle/Colab dual T4
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PORT=8081
export FACTORY_SERVICE_ID=vision-service

echo "=== Initializing Factory v4 VLM Daemon on GPU 1 (Port 8081) ==="
python3 -m services.vlm.launcher.run_vlm_service --port 8081 --host 0.0.0.0
