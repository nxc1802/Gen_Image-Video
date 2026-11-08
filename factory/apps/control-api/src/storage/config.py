"""
factory/apps/control-api/src/storage/config.py
Storage configuration and production credential validation.
"""
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    storage_backend: str = Field(default="local", description="'s3' or 'local'")
    s3_endpoint_url: Optional[str] = None
    s3_public_endpoint_url: Optional[str] = None
    s3_bucket_name: str = "factory-artifacts"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"
    s3_download_url_expiration_seconds: int = Field(default=86400, ge=60, le=604800)
    s3_upload_url_expiration_seconds: int = Field(default=3600, ge=60, le=86400)
    local_storage_dir: str = Field(
        default_factory=lambda: (
            os.environ.get("SCRATCH_ARTIFACTS_DIR")
            or os.environ.get("FACTORY_ARTIFACT_DIR")
            or os.environ.get("LOCAL_STORAGE_DIR")
            or "/tmp/factory_lite/artifacts"
        )
    )
    local_storage_secret_key: str = ""
    local_base_url: str = Field(
        default_factory=lambda: (
            os.environ.get("CONTROL_PLANE_URL")
            or os.environ.get("FACTORY_BASE_URL")
            or os.environ.get("LOCAL_BASE_URL")
            or "http://localhost:8000"
        )
    )
    environment: str = "development"


def load_storage_config() -> StorageConfig:
    env = os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower().strip()
    backend = os.environ.get("STORAGE_BACKEND", "local").lower().strip() or "local"
    if backend not in {"local", "ephemeral_disk", "ephemeral", "s3"}:
        raise RuntimeError(f"Unsupported STORAGE_BACKEND '{backend}'. Supported backends: 'local', 'ephemeral_disk', 's3'.")


    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip() or None
    s3_public_endpoint = os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip() or None
    s3_bucket = os.environ.get("S3_BUCKET_NAME", "factory-artifacts").strip()
    access_key = os.environ.get("S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "")).strip()
    secret_key = os.environ.get("S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip()
    s3_region = os.environ.get("S3_REGION", os.environ.get("AWS_REGION", "us-east-1")).strip()

    # Fail-fast startup guard in production mode when S3 is configured
    if env == "production" and backend == "s3":
        if not s3_bucket:
            raise RuntimeError("FATAL: S3_BUCKET_NAME must be configured when STORAGE_BACKEND=s3 in production mode.")
        if not access_key or not secret_key:
            raise RuntimeError("FATAL: S3_ACCESS_KEY and S3_SECRET_KEY (or AWS credentials) must be configured when STORAGE_BACKEND=s3 in production mode.")
        if access_key == "minioadmin" or secret_key == "minioadmin":
            raise RuntimeError("FATAL: Default MinIO credentials ('minioadmin') are forbidden in production mode.")

    local_secret = os.environ.get("LOCAL_STORAGE_SECRET_KEY", "").strip()
    if not local_secret:
        local_secret = os.environ.get("FACTORY_API_KEY", "").strip()
    if not local_secret:
        local_secret = "factory-lite-storage-secret"

    return StorageConfig(
        storage_backend=backend,
        s3_endpoint_url=s3_endpoint,
        s3_public_endpoint_url=s3_public_endpoint,
        s3_bucket_name=s3_bucket,
        s3_access_key=access_key,
        s3_secret_key=secret_key,
        s3_region=s3_region,
        s3_download_url_expiration_seconds=int(os.environ.get("S3_DOWNLOAD_URL_EXPIRATION_SECONDS", "86400")),
        s3_upload_url_expiration_seconds=int(os.environ.get("S3_UPLOAD_URL_EXPIRATION_SECONDS", "3600")),
        local_storage_dir=(
            os.environ.get("SCRATCH_ARTIFACTS_DIR", "").strip()
            or os.environ.get("FACTORY_ARTIFACT_DIR", "").strip()
            or os.environ.get("LOCAL_STORAGE_DIR", "").strip()
            or "/tmp/factory_lite/artifacts"
        ),
        local_storage_secret_key=local_secret,
        local_base_url=(
            os.environ.get("CONTROL_PLANE_URL", "").strip()
            or os.environ.get("FACTORY_BASE_URL", "").strip()
            or os.environ.get("LOCAL_BASE_URL", "").strip()
            or "http://localhost:8000"
        ),
        environment=env,
    )
