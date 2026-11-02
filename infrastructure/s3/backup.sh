#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - S3 / MinIO Object Storage Backup Script
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
            echo "  S3_ENDPOINT_URL   Endpoint URL for MinIO/S3 (e.g. http://localhost:9000)"
            echo "  S3_BUCKET_NAME    Target bucket name (default: factory-artifacts)"
            echo "  BACKUP_DIR        Destination directory (default: ./backups/s3)"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://localhost:9000}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-factory-artifacts}"
BACKUP_DIR="${BACKUP_DIR:-./backups/s3}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TARGET_DIR="${BACKUP_DIR}/${S3_BUCKET_NAME}_${TIMESTAMP}"
METADATA_FILE="${BACKUP_DIR}/${S3_BUCKET_NAME}_${TIMESTAMP}.meta.json"

echo "=== AI Factory v4 S3 / MinIO Backup ==="
echo "Bucket: ${S3_BUCKET_NAME} (${S3_ENDPOINT_URL})"
echo "Output Directory: ${TARGET_DIR}"
echo "Dry Run: ${DRY_RUN}"

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Verification complete. Backup would synchronize s3://${S3_BUCKET_NAME} to ${TARGET_DIR}."
    exit 0
fi

mkdir -p "${TARGET_DIR}"

if command -v aws >/dev/null 2>&1; then
    ENDPOINT_FLAG=()
    if [ -n "${S3_ENDPOINT_URL}" ]; then
        ENDPOINT_FLAG=("--endpoint-url" "${S3_ENDPOINT_URL}")
    fi
    echo "Synchronizing bucket with AWS CLI..."
    aws s3 sync "s3://${S3_BUCKET_NAME}" "${TARGET_DIR}" "${ENDPOINT_FLAG[@]}"
elif command -v mc >/dev/null 2>&1; then
    echo "Synchronizing bucket with MinIO Client (mc)..."
    mc mirror "factory/${S3_BUCKET_NAME}" "${TARGET_DIR}"
else
    echo "Using python boto3 synchronization fallback..."
    python3 - <<PY
import os, sys
from pathlib import Path
try:
    import boto3
    from botocore.client import Config
except ImportError:
    print("ERROR: Neither aws, mc, nor boto3 is available.", file=sys.stderr)
    sys.exit(2)

endpoint = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
bucket_name = os.environ.get("S3_BUCKET_NAME", "factory-artifacts")
target_dir = Path("${TARGET_DIR}")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    config=Config(signature_version="s3v4"),
)

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket_name):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        dest = target_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket_name, key, str(dest))
PY
fi

cat <<EOF > "${METADATA_FILE}"
{
  "bucket": "${S3_BUCKET_NAME}",
  "endpoint": "${S3_ENDPOINT_URL}",
  "timestamp": "${TIMESTAMP}",
  "backup_dir": "${TARGET_DIR}"
}
EOF

echo "✓ S3 backup synchronized successfully to: ${TARGET_DIR}"
exit 0
