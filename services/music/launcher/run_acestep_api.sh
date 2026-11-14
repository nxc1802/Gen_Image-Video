#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8008}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SEC="${HEALTH_TIMEOUT:-60}"
# Explicit identity only; raw repository/variant defaults are retired.
: "${ACESTEP_MODEL_REF:?Set an explicit canonical ACESTEP_MODEL_REF}"
MODEL="$(python - <<'PYCODE'
import os
from model_registry.bindings import model_binding
model = model_binding(os.environ['ACESTEP_MODEL_REF'])
if model.apiVersion == 'factory.ai/v2':
    raise SystemExit('Legacy ACE launcher cannot bind the full pinned bundle; canonical backend lifecycle is required. No checkpoint fallback.')
print(model.weights.repository)
PYCODE
)"

# Launch native acestep-api executable if available, otherwise launch via python module
if command -v acestep-api >/dev/null 2>&1; then
    acestep-api --host "${HOST}" --port "${PORT}" --model "${MODEL}" &
    SERVER_PID=$!
elif [ -n "${ACESTEP_DIR:-}" ] && [ -d "${ACESTEP_DIR:-}" ]; then
    (cd "${ACESTEP_DIR}" && uv run acestep-api --server-name "${HOST}" --port "${PORT}" --model "${MODEL}") &
    SERVER_PID=$!
else
    # Fallback to python module execution for native acestep entrypoint
    python -m acestep.api --host "${HOST}" --port "${PORT}" --model "${MODEL}" &
    SERVER_PID=$!
fi

# Function to perform clean shutdown
cleanup() {
    echo "Stopping native ACE-Step server (PID: ${SERVER_PID})..."
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Perform launcher health check on native backend
echo "Waiting for native ACE-Step backend health check on http://${HOST}:${PORT}/health..."
START_TIME=$(date +%s)
HEALTHY=0

while [ $(( $(date +%s) - START_TIME )) -lt "${TIMEOUT_SEC}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Error: Native ACE-Step process died prematurely." >&2
        exit 1
    fi
    if curl -s -f -o /dev/null "http://${HOST}:${PORT}/health" 2>/dev/null; then
        echo "Native ACE-Step process health check passed; inference remains uncertified."
        HEALTHY=1
        break
    fi
    sleep 1
done

if [ "${HEALTHY}" -ne 1 ]; then
    echo "Error: Native ACE-Step health check timed out after ${TIMEOUT_SEC}s." >&2
    exit 1
fi

# Wait for server process
wait "${SERVER_PID}"
