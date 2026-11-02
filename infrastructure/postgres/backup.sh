#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - PostgreSQL Automated Backup Script
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
            echo "  PGHOST        PostgreSQL host (default: localhost)"
            echo "  PGPORT        PostgreSQL port (default: 5432)"
            echo "  PGUSER        PostgreSQL user (default: postgres)"
            echo "  PGDATABASE    PostgreSQL database (default: factory_v4)"
            echo "  BACKUP_DIR    Destination directory (default: ./backups/postgres)"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGDATABASE="${PGDATABASE:-factory_v4}"
BACKUP_DIR="${BACKUP_DIR:-./backups/postgres}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
BACKUP_FILE="${BACKUP_DIR}/${PGDATABASE}_${TIMESTAMP}.dump"
METADATA_FILE="${BACKUP_DIR}/${PGDATABASE}_${TIMESTAMP}.meta.json"

echo "=== AI Factory v4 PostgreSQL Backup ==="
echo "Target: ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
echo "Output: ${BACKUP_FILE}"
echo "Dry Run: ${DRY_RUN}"

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Verification complete. Backup command would be:"
    echo "  pg_dump -h ${PGHOST} -p ${PGPORT} -U ${PGUSER} -Fc -d ${PGDATABASE} -f ${BACKUP_FILE}"
    exit 0
fi

mkdir -p "${BACKUP_DIR}"

if ! command -v pg_dump >/dev/null 2>&1; then
    echo "ERROR: pg_dump utility not found in PATH." >&2
    exit 2
fi

echo "Exporting database schema and data..."
pg_dump \
    -h "${PGHOST}" \
    -p "${PGPORT}" \
    -U "${PGUSER}" \
    -Fc \
    -d "${PGDATABASE}" \
    -f "${BACKUP_FILE}"

# Generate SHA256 checksum
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
fi

cat <<EOF > "${METADATA_FILE}"
{
  "database": "${PGDATABASE}",
  "timestamp": "${TIMESTAMP}",
  "host": "${PGHOST}",
  "port": "${PGPORT}",
  "user": "${PGUSER}",
  "file": "${BACKUP_FILE}",
  "size_bytes": $(wc -c < "${BACKUP_FILE}" | tr -d ' ')
}
EOF

echo "✓ PostgreSQL backup created successfully: ${BACKUP_FILE}"
exit 0
