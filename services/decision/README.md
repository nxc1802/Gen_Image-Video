# Factory v4 Decision Routing Engine (`services/decision/`)

## Overview
The Decision Engine Subsystem provides fast, calibrated decision-making, hypothesis validation, and multi-model pipeline routing powered by Qwen2.5-3B. It operates on port 8082 behind the static domain `decision.ai.example.com`.

## Tasks Supported on `POST /v1/decisions`
- `predicate`: Binary True/False decision with confidence probability and exact token logprobs.
- `choice`: Multi-class routing selecting the optimal candidate choice with full Softmax probability distribution.
- `score`: Continuous scalar quality/priority evaluation with Expected Calibration Error (ECE) and Brier reliability scores.

## Hardware & Deployment
- Target GPU: GPU 0 (cuda:0) co-located with `text-service` on Dual Tesla T4
- VRAM Target: ~4.5 GB (Qwen2.5-3B bf16)
- Launch command: `python -m services.decision.decision_service --port 8082`
