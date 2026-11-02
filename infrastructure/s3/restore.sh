#!/usr/bin/env bash
# ==============================================================================
# AI Factory v4 - S3 / MinIO Object Storage Restore Script
# Usage: ./restore.sh [--dry-run] [--help] <backup_directory>
# ==============================================================================
set -euo pipefail

DRY_RUN=false
SOURCE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--help] <backup_directory>"
            echo "Environment variables:"
            echo "  S3_ENDPOINT_URL   Endpoint URL for MinIO/S3 (e.g. http://localhost:9000)"
            echo "  S3_BUCKET_NAME    Target bucket name (default: factory-artifacts)"
            exit 0
            ;;
        *)
            if [ -z "${SOURCE_DIR}" ]; then
                SOURCE_DIR="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [ -z "${SOURCE_DIR}" ]; then
    echo "ERROR: Backup directory path required as argument." >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://localhost:9000}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-factory-artifacts}"

echo "=== AI Factory v4 S3 / MinIO Restore ==="
echo "Bucket: ${S3_BUCKET_NAME} (${S3_ENDPOINT_URL})"
echo "Source Directory: ${SOURCE_DIR}"
echo "Dry Run: ${DRY_RUN}"

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "ERROR: Specified source backup directory does not exist: ${SOURCE_DIR}" >&2
    exit 2
fi

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Verification complete. Restore would synchronize ${SOURCE_DIR} to s3://${S3_BUCKET_NAME}."
    exit 0
fi

if command -v aws >/dev/null 2>&1; then
    ENDPOINT_FLAG=()
    if [ -n "${S3_ENDPOINT_URL}" ]; then
        ENDPOINT_FLAG=("--endpoint-url" "${S3_ENDPOINT_URL}")
    fi
    echo "Restoring bucket with AWS CLI..."
    aws s3 sync "${SOURCE_DIR}" "s3://${S3_BUCKET_NAME}" "${ENDPOINT_FLAG[@]}"
elif command -v mc >/dev/null 2>&1; then
    echo "Restoring bucket with MinIO Client (mc)..."
    mc mirror "${SOURCE_DIR}" "factory/${S3_BUCKET_NAME}"
else
    echo "Using python boto3 upload fallback..."
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
source_dir = Path("${SOURCE_DIR}")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    config=Config(signature_version="s3v4"),
)

for root, _, files in os.walk(source_dir):
    for f in files:
        full_path = Path(root) / f
        rel_key = str(full_path.relative_to(source_dir))
        s3.upload_file(str(full_path), bucket_name, rel_key)
PY
fi

echo "✓ S3 restore completed successfully to bucket: ${S3_BUCKET_NAME}"
exit 0
