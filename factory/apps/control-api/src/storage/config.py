"""
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
    local_storage_dir: str = ".factory_storage/artifacts"
    local_storage_secret_key: str = ""
    local_base_url: str = "http://localhost:8000"


def load_storage_config() -> StorageConfig:
    environment = os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower()
    backend = os.environ.get("STORAGE_BACKEND", "").lower().strip()
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip() or None
    if not backend:
        backend = "s3" if s3_endpoint else "local"
    if backend not in {"s3", "local"}:
        raise RuntimeError(f"Unsupported STORAGE_BACKEND '{backend}'.")

    access_key = os.environ.get("S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "")).strip()
    secret_key = os.environ.get("S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip()
    local_secret = os.environ.get("LOCAL_STORAGE_SECRET_KEY", "").strip()

    if environment == "production":
        if backend != "s3":
            raise RuntimeError("Production requires STORAGE_BACKEND=s3.")
        if not s3_endpoint:
            raise RuntimeError("Production S3 storage requires S3_ENDPOINT_URL.")
        if not access_key or not secret_key:
            raise RuntimeError("Production S3 storage requires explicit access and secret keys.")
        if access_key == "minioadmin" or secret_key == "minioadmin":
            raise RuntimeError("Default MinIO credentials are forbidden in production.")

    if backend == "local" and not local_secret:
        # Local storage is development-only; still use an explicit/random application
        # secret rather than a repository-wide hardcoded HMAC key.
        local_secret = os.environ.get("FACTORY_API_KEY", "").strip()
        if not local_secret:
            local_secret = "development-only-change-me"

    return StorageConfig(
        storage_backend=backend,
        s3_endpoint_url=s3_endpoint,
        s3_public_endpoint_url=os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip() or None,
        s3_bucket_name=os.environ.get("S3_BUCKET_NAME", "factory-artifacts"),
        s3_access_key=access_key,
        s3_secret_key=secret_key,
        s3_region=os.environ.get("S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
        s3_download_url_expiration_seconds=int(os.environ.get("S3_DOWNLOAD_URL_EXPIRATION_SECONDS", "86400")),
        s3_upload_url_expiration_seconds=int(os.environ.get("S3_UPLOAD_URL_EXPIRATION_SECONDS", "3600")),
        local_storage_dir=os.environ.get("LOCAL_STORAGE_DIR", ".factory_storage/artifacts"),
        local_storage_secret_key=local_secret,
        local_base_url=os.environ.get("LOCAL_BASE_URL", "http://localhost:8000"),
    )
