#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8288}"
HOST="${HOST:-0.0.0.0}"

echo "Starting Speech Subsystem Daemon on ${HOST}:${PORT}..."
exec python -m services.speech.server --host "${HOST}" --port "${PORT}"
