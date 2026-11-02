#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - PostgreSQL Automated Restore Script
# Usage: ./restore.sh [--dry-run] [--help] <backup_file.dump>
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
            echo "Usage: $0 [--dry-run] [--help] <backup_file.dump>"
            echo "Environment variables:"
            echo "  PGHOST        PostgreSQL host (default: localhost)"
            echo "  PGPORT        PostgreSQL port (default: 5432)"
            echo "  PGUSER        PostgreSQL user (default: postgres)"
            echo "  PGDATABASE    PostgreSQL database (default: factory_v4)"
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

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGDATABASE="${PGDATABASE:-factory_v4}"

echo "=== AI Factory v4 PostgreSQL Restore ==="
echo "Target: ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
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
    echo "[DRY-RUN] Verification complete. Restore command would be:"
    echo "  pg_restore -h ${PGHOST} -p ${PGPORT} -U ${PGUSER} -d ${PGDATABASE} --clean --if-exists ${BACKUP_FILE}"
    exit 0
fi

if ! command -v pg_restore >/dev/null 2>&1; then
    echo "ERROR: pg_restore utility not found in PATH." >&2
    exit 2
fi

echo "Restoring database from dump..."
pg_restore \
    -h "${PGHOST}" \
    -p "${PGPORT}" \
    -U "${PGUSER}" \
    -d "${PGDATABASE}" \
    --clean \
    --if-exists \
    --no-owner \
    "${BACKUP_FILE}"

echo "✓ PostgreSQL restore completed successfully for database: ${PGDATABASE}"
exit 0
