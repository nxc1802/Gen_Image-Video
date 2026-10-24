#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8008}"
HOST="${HOST:-127.0.0.1}"

echo "Starting Native ACE-Step 1.5 Server on ${HOST}:${PORT}..."
# In live GPU cluster: acestep-api --port 8008 --model ACE-Studio/ACE-Step-1.5
exec python -m services.music.server --host "${HOST}" --port "${PORT}"
