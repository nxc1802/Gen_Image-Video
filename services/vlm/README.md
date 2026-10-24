# Factory v4 Vision-Language Model Daemon (`services/vlm/`)

## Overview
The VLM Subsystem provides decoupled vision-language understanding powered by Qwen2-VL. It operates on port 8081 behind the static domain `vision.ai.example.com`.

## Capabilities
- Visual QA & Dialogue: Handles text prompts paired with image/video URLs or base64 data URLs.
- Visual Grounding: Detects objects and emits coordinate bounding boxes for UI grounding.
- Document QA: High-resolution OCR and layout-aware document comprehension.

## Hardware & Deployment
- Target GPU: GPU 1 (cuda:1) on Dual Tesla T4
- VRAM Target: ~9.5 GB (Qwen2-VL-7B AWQ)
- Launch command: `python -m services.vlm.launcher.run_vlm_service --port 8081`
