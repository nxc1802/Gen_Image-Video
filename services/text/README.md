# Factory v4 Text & VLM Generation Daemon (`services/text/`)

## Overview
The Text Subsystem provides decoupled, high-performance text and multimodal inference powered by Qwen2.5 and vLLM continuous batching. It runs independently on port 8080 behind the static domain `text.ai.example.com`.

## Endpoints
- `POST /v1/chat/completions`: OpenAI-compatible chat completions with SSE streaming (`stream=true`), structured JSON output (`response_format={"type": "json_object"}`), and tool calling.
- `POST /v1/responses`: OpenAI Responses API for multi-step reasoning, tracking reasoning tokens and model reflection.
- `GET /health`: Subsystem health status and readiness probe.

## Hardware & Deployment
- Target GPU: GPU 0 (cuda:0) on Dual Tesla T4
- VRAM Footprint: ~7.5 GB (Qwen2.5-7B AWQ)
- Launch command: `python -m services.text.launcher.run_text_service --port 8080`
