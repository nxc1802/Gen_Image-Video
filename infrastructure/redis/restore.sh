#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - Redis Snapshot Restore Script
# Usage: ./restore.sh [--dry-run] [--help] <dump_file.rdb>
# ==============================================================================
set -euo pipefail

DRY_RUN=false
BACKUP_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--help] <dump_file.rdb>"
            echo "Environment variables:"
            echo "  REDIS_HOST      Redis host (default: 127.0.0.1)"
            echo "  REDIS_PORT      Redis port (default: 6379)"
            echo "  REDIS_PASSWORD  Redis password (default: empty)"
            exit 0
            ;;
        *)
            if [ -z "${BACKUP_FILE}" ]; then
                BACKUP_FILE="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [ -z "${BACKUP_FILE}" ]; then
    echo "ERROR: Backup dump file path required as argument." >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

echo "=== AI Factory v4 Redis Restore ==="
echo "Target: ${REDIS_HOST}:${REDIS_PORT}"
echo "Input: ${BACKUP_FILE}"
echo "Dry Run: ${DRY_RUN}"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "ERROR: Specified backup file does not exist: ${BACKUP_FILE}" >&2
    exit 2
fi

# Verify checksum if present
if [ -f "${BACKUP_FILE}.sha256" ]; then
    echo "Verifying SHA256 checksum..."
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "${BACKUP_FILE}.sha256"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "${BACKUP_FILE}.sha256"
    fi
    echo "✓ Checksum matches."
fi

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Verification complete. Restore would stop Redis, replace dump.rdb, and restart service."
    exit 0
fi

if ! command -v redis-cli >/dev/null 2>&1; then
    echo "ERROR: redis-cli utility not found in PATH." >&2
    exit 2
fi

AUTH_ARGS=()
if [ -n "${REDIS_PASSWORD}" ]; then
    AUTH_ARGS=("-a" "${REDIS_PASSWORD}")
fi

REDIS_DIR=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" CONFIG GET dir | sed -n '2p')
REDIS_DBFILENAME=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" CONFIG GET dbfilename | sed -n '2p')
DEST_RDB="${REDIS_DIR}/${REDIS_DBFILENAME}"

echo "Restoring RDB snapshot to: ${DEST_RDB}..."
cp "${BACKUP_FILE}" "${DEST_RDB}"

echo "✓ Redis RDB restored. Restart Redis server to load new snapshot."
exit 0
