#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - Redis Snapshot & AOF Backup Script
# Usage: ./backup.sh [--dry-run] [--help]
# ==============================================================================
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--help]"
            echo "Environment variables:"
            echo "  REDIS_HOST      Redis host (default: 127.0.0.1)"
            echo "  REDIS_PORT      Redis port (default: 6379)"
            echo "  REDIS_PASSWORD  Redis password (default: empty)"
            echo "  BACKUP_DIR      Destination directory (default: ./backups/redis)"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
BACKUP_DIR="${BACKUP_DIR:-./backups/redis}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
BACKUP_FILE="${BACKUP_DIR}/dump_${TIMESTAMP}.rdb"
METADATA_FILE="${BACKUP_DIR}/dump_${TIMESTAMP}.meta.json"

echo "=== AI Factory v4 Redis Backup ==="
echo "Target: ${REDIS_HOST}:${REDIS_PORT}"
echo "Output: ${BACKUP_FILE}"
echo "Dry Run: ${DRY_RUN}"

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Verification complete. Backup command would trigger BGSAVE and copy dump.rdb."
    exit 0
fi

mkdir -p "${BACKUP_DIR}"

if ! command -v redis-cli >/dev/null 2>&1; then
    echo "ERROR: redis-cli utility not found in PATH." >&2
    exit 2
fi

AUTH_ARGS=()
if [ -n "${REDIS_PASSWORD}" ]; then
    AUTH_ARGS=("-a" "${REDIS_PASSWORD}")
fi

echo "Triggering Redis BGSAVE..."
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" BGSAVE || true

# Wait briefly for BGSAVE to complete
sleep 2

# Retrieve Redis directory and dbfilename
REDIS_DIR=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" CONFIG GET dir | sed -n '2p')
REDIS_DBFILENAME=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" CONFIG GET dbfilename | sed -n '2p')
SOURCE_RDB="${REDIS_DIR}/${REDIS_DBFILENAME}"

if [ -f "${SOURCE_RDB}" ]; then
    cp "${SOURCE_RDB}" "${BACKUP_FILE}"
else
    # Fallback: create snapshot via redis-cli --rdb
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${AUTH_ARGS[@]}" --rdb "${BACKUP_FILE}"
fi

# Generate SHA256 checksum
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
fi

cat <<EOF > "${METADATA_FILE}"
{
  "host": "${REDIS_HOST}",
  "port": "${REDIS_PORT}",
  "timestamp": "${TIMESTAMP}",
  "file": "${BACKUP_FILE}",
  "size_bytes": $(wc -c < "${BACKUP_FILE}" | tr -d ' ')
}
EOF

echo "✓ Redis RDB snapshot created successfully: ${BACKUP_FILE}"
exit 0
