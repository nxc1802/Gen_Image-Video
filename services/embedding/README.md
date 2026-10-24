# Factory v4 Vector Embedding & Rerank Daemon (`services/embedding/`)

## Overview
The Embedding & Rerank Subsystem provides dense semantic embeddings and cross-encoder document reranking powered by Qwen-Embedding and Qwen-Rerank. It operates on port 8083 behind the static domain `embed.ai.example.com`.

## Integrity Guardrails
- **Zero Mock / Fake Vectors**: If an embedding model is offline or uninitialized, the service strictly fails fast with HTTP 503 `MODEL_UNAVAILABLE`. Deterministic pseudo-hash or emulated vectors are strictly prohibited.
- **Zero Mock Rerank Scores**: If a rerank model is unavailable, the service fails fast with HTTP 503 `MODEL_UNAVAILABLE`.
- **Normalization**: Dense float vectors are strictly L2 normalized ($\|v\|_2 = 1.0$), with values bounded in $[-1.0, 1.0]$.
- **Rerank Sorting**: Rerank results are strictly sorted in descending order of `relevance_score`.

## Endpoints
- `POST /v1/embeddings` & `POST /v2/embed`: Dense vector representations (1024-dim normalized).
- `POST /v1/rerank` & `POST /rerank` & `POST /v2/rerank`: Cross-encoder document relevance scoring.
- `GET /health`: Health and readiness probe.

## Hardware & Deployment
- Target GPU: GPU 1 (cuda:1) co-located with `vision-service` on Dual Tesla T4
- VRAM Footprint: ~1.5 GB
- Launch command: `python -m services.embedding.launcher.run_embedding_service --port 8083`
