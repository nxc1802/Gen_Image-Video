#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8002}"
HOST="${HOST:-0.0.0.0}"
WORKER_ID="${WORKER_ID:-worker-enhance-0}"
CONTROL_API="${CONTROL_API:-https://jobs.ai.example.com}"

echo "Starting Media Enhancement Daemon on ${HOST}:${PORT}..."
python -m services.enhancement.server --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!

echo "Starting Media Enhancement Worker [${WORKER_ID}] against ${CONTROL_API}..."
python -m services.enhancement.worker --worker-id "${WORKER_ID}" --control-api "${CONTROL_API}" &
WORKER_PID=$!

trap "kill ${SERVER_PID} ${WORKER_PID} 2>/dev/null || true" SIGINT SIGTERM EXIT
wait
