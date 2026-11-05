"""
factory/apps/control-api/src/storage/s3.py
Production S3 and MinIO storage provider implementation.
"""
import asyncio
from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    import boto3
    import botocore.client
    from botocore.exceptions import ClientError

from factory.apps.control_api.src.storage.config import StorageConfig
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.storage.provider import (
    PresignedUpload,
    StorageObjectMetadata,
    StorageProvider,
)


class S3StorageProvider(StorageProvider):
    """
    S3 & MinIO storage provider.
    Supports path-style addressing, SigV4, public endpoint remapping, and presigned URLs.
    Fails fast in production mode if bucket is inaccessible or credentials are invalid.
    """

    def __init__(self, config: StorageConfig):
        import boto3
        import botocore.client

        self.config = config
        self.bucket = config.s3_bucket_name
        self.client_config = botocore.client.Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.s3_access_key or None,
            aws_secret_access_key=config.s3_secret_key or None,
            region_name=config.s3_region,
            config=self.client_config,
        )
        presign_endpoint = config.s3_public_endpoint_url or config.s3_endpoint_url
        self.presign_client = boto3.client(
            "s3",
            endpoint_url=presign_endpoint,
            aws_access_key_id=config.s3_access_key or None,
            aws_secret_access_key=config.s3_secret_key or None,
            region_name=config.s3_region,
            config=self.client_config,
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        env = getattr(self.config, "environment", None) or os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower()
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as head_err:
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except Exception as create_err:
                if env == "production":
                    raise RuntimeError(
                        f"FATAL: S3 bucket '{self.bucket}' is inaccessible or could not be created in production mode: {head_err}. Creation error: {create_err}"
                    ) from create_err
                # In development, ignore if offline/mock or permissions disallow creation

    def generate_presigned_upload_url(
        self,
        bucket: str,
        key: str,
        mime_type: str,
        max_size_bytes: int,
        expires_in_seconds: int = 3600,
    ) -> PresignedUpload:
        target_bucket = bucket or self.bucket
        params = {
            "Bucket": target_bucket,
            "Key": key,
            "ContentType": mime_type,
        }
        url = self.presign_client.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=expires_in_seconds,
        )
        return PresignedUpload(
            upload_url=url,
            method="PUT",
            headers={"Content-Type": mime_type},
            storage_path=key,
            expires_in_seconds=expires_in_seconds,
        )

    def generate_presigned_download_url(
        self,
        bucket: str,
        key: str,
        expires_in_seconds: int = 86400,
        filename: Optional[str] = None,
    ) -> str:
        target_bucket = bucket or self.bucket
        params: Dict[str, Any] = {
            "Bucket": target_bucket,
            "Key": key,
        }
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

        return self.presign_client.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=expires_in_seconds,
        )

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        mime_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObjectMetadata:
        target_bucket = bucket or self.bucket
        sha256 = compute_bytes_sha256(data)
        meta = dict(metadata or {})
        meta["sha256"] = sha256

        def _put():
            return self.client.put_object(
                Bucket=target_bucket,
                Key=key,
                Body=data,
                ContentType=mime_type,
                Metadata=meta,
            )

        resp = await asyncio.to_thread(_put)
        etag = resp.get("ETag", "").strip('"')

        return StorageObjectMetadata(
            bucket=target_bucket,
            key=key,
            size_bytes=len(data),
            content_type=mime_type,
            sha256=sha256,
            etag=etag,
            last_modified=datetime.now(timezone.utc),
            custom_metadata=meta,
        )

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        target_bucket = bucket or self.bucket

        def _get():
            resp = self.client.get_object(Bucket=target_bucket, Key=key)
            return resp["Body"].read()

        return await asyncio.to_thread(_get)

    async def head_object(self, bucket: str, key: str) -> Optional[StorageObjectMetadata]:
        from botocore.exceptions import ClientError

        target_bucket = bucket or self.bucket

        def _head():
            try:
                return self.client.head_object(Bucket=target_bucket, Key=key)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    return None
                raise

        resp = await asyncio.to_thread(_head)
        if resp is None:
            return None

        sha256 = resp.get("Metadata", {}).get("sha256", "")
        content_type = resp.get("ContentType", "application/octet-stream")
        size_bytes = resp.get("ContentLength", 0)
        etag = resp.get("ETag", "").strip('"')
        last_modified = resp.get("LastModified", datetime.now(timezone.utc))

        return StorageObjectMetadata(
            bucket=target_bucket,
            key=key,
            size_bytes=size_bytes,
            content_type=content_type,
            sha256=sha256,
            etag=etag,
            last_modified=last_modified,
            custom_metadata=resp.get("Metadata", {}),
        )

    async def delete_object(self, bucket: str, key: str) -> bool:
        target_bucket = bucket or self.bucket

        def _delete():
            self.client.delete_object(Bucket=target_bucket, Key=key)

        await asyncio.to_thread(_delete)
        return True


def __getattr__(name: str):
    if name == "boto3":
        import boto3
        return boto3
    if name == "botocore":
        import botocore
        return botocore
    if name == "ClientError":
        from botocore.exceptions import ClientError
        return ClientError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

