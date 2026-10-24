"""
factory/apps/control-api/src/storage/config.py
Storage configuration models and environment loader.
"""
import os
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    storage_backend: str = Field(default="local", description="'s3' or 'local'")
    s3_endpoint_url: Optional[str] = Field(default=None, description="Internal S3/MinIO endpoint, e.g. http://minio:9000")
    s3_public_endpoint_url: Optional[str] = Field(default=None, description="External S3/MinIO endpoint, e.g. https://s3.ai.example.com")
    s3_bucket_name: str = Field(default="factory-artifacts")
    s3_access_key: str = Field(default="minioadmin")
    s3_secret_key: str = Field(default="minioadmin")
    s3_region: str = Field(default="us-east-1")
    s3_download_url_expiration_seconds: int = Field(default=86400, ge=60, le=604800)  # 24 hours
    s3_upload_url_expiration_seconds: int = Field(default=3600, ge=60, le=86400)      # 1 hour
    local_storage_dir: str = Field(default=".factory_storage/artifacts")
    local_storage_secret_key: str = Field(default="factory-storage-hmac-secret-key-32b")
    local_base_url: str = Field(default="http://localhost:8000")


def load_storage_config() -> StorageConfig:
    backend = os.environ.get("STORAGE_BACKEND", "").lower().strip()
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip() or None
    
    # If not explicitly specified, auto-select 's3' if S3_ENDPOINT_URL is set, else 'local'
    if not backend:
        backend = "s3" if s3_endpoint else "local"

    return StorageConfig(
        storage_backend=backend,
        s3_endpoint_url=s3_endpoint,
        s3_public_endpoint_url=os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip() or None,
        s3_bucket_name=os.environ.get("S3_BUCKET_NAME", "factory-artifacts"),
        s3_access_key=os.environ.get("S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")),
        s3_secret_key=os.environ.get("S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")),
        s3_region=os.environ.get("S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
        s3_download_url_expiration_seconds=int(os.environ.get("S3_DOWNLOAD_URL_EXPIRATION_SECONDS", "86400")),
        s3_upload_url_expiration_seconds=int(os.environ.get("S3_UPLOAD_URL_EXPIRATION_SECONDS", "3600")),
        local_storage_dir=os.environ.get("LOCAL_STORAGE_DIR", ".factory_storage/artifacts"),
        local_storage_secret_key=os.environ.get("LOCAL_STORAGE_SECRET_KEY", "factory-storage-hmac-secret-key-32b"),
        local_base_url=os.environ.get("LOCAL_BASE_URL", "http://localhost:8000"),
    )
