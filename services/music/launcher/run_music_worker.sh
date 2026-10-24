#!/usr/bin/env bash
set -euo pipefail

WORKER_ID="${WORKER_ID:-worker-music-0}"
CONTROL_API="${CONTROL_API:-https://jobs.ai.example.com}"

echo "Starting Factory Music Worker [${WORKER_ID}] against ${CONTROL_API}..."
exec python -m services.music.worker --worker-id "${WORKER_ID}" --control-api "${CONTROL_API}"
