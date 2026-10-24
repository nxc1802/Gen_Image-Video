#!/usr/bin/env bash
# run_kaggle.sh - Launch Decision Service on GPU 0 in Kaggle/Colab dual T4
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PORT=8082
export FACTORY_SERVICE_ID=decision-service

echo "=== Initializing Factory v4 Decision Engine on GPU 0 (Port 8082) ==="
python3 -m services.decision.launcher.run_decision_service --port 8082 --host 0.0.0.0
