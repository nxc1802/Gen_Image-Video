"""
factory/apps/control-api/src/storage/local.py
Zero-dependency local filesystem storage provider with HMAC-SHA256 URL signing.
"""
import asyncio
import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from factory_contracts.errors import InvalidRequestError, NotFoundError
from factory.apps.control_api.src.storage.config import StorageConfig
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.storage.provider import (
    PresignedUpload,
    StorageObjectMetadata,
    StorageProvider,
)


class LocalStorageProvider(StorageProvider):
    """
    Zero-dependency local filesystem storage engine.
    Uses HMAC-SHA256 URL signatures to replicate presigned S3 URLs in dev/test.
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_dir = Path(config.local_storage_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.secret_key = config.local_storage_secret_key.encode("utf-8")
        self.base_url = config.local_base_url.rstrip("/")

    def _get_path(self, bucket: str, key: str) -> Path:
        """Resolves target path while strictly preventing directory traversal."""
        # 1. Explicitly reject any relative path traversal indicators in bucket or key
        if ".." in bucket or ".." in key:
            raise InvalidRequestError("Directory traversal detected in storage path.")

        clean_bucket = bucket.strip("/")
        clean_key = key.lstrip("/")
        target = (self.base_dir / clean_bucket / clean_key).resolve()

        # 2. Strict boundary check using is_relative_to and ensuring target is not the base_dir itself
        if not target.is_relative_to(self.base_dir) or target == self.base_dir:
            raise InvalidRequestError("Directory traversal detected in storage path.")

        return target

    def _sign(self, bucket: str, key: str, expires_at: int) -> str:
        msg = f"{bucket}:{key}:{expires_at}".encode("utf-8")
        return hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

    def verify_signature(self, bucket: str, key: str, expires_at: int, signature: str) -> bool:
        if time.time() > expires_at:
            return False
        expected = self._sign(bucket, key, expires_at)
        return hmac.compare_digest(expected, signature)

    def generate_presigned_upload_url(
        self,
        bucket: str,
        key: str,
        mime_type: str,
        max_size_bytes: int,
        expires_in_seconds: int = 3600,
    ) -> PresignedUpload:
        target_bucket = bucket or self.config.s3_bucket_name
        expires_at = int(time.time()) + expires_in_seconds
        sig = self._sign(target_bucket, key, expires_at)
        query = urlencode({
            "bucket": target_bucket,
            "key": key,
            "expires": expires_at,
            "sig": sig,
            "mime_type": mime_type,
        })
        upload_url = f"{self.base_url}/v1/storage/upload?{query}"
        return PresignedUpload(
            upload_url=upload_url,
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
        target_bucket = bucket or self.config.s3_bucket_name
        expires_at = int(time.time()) + expires_in_seconds
        sig = self._sign(target_bucket, key, expires_at)
        params = {
            "bucket": target_bucket,
            "key": key,
            "expires": expires_at,
            "sig": sig,
        }
        if filename:
            params["filename"] = filename
        query = urlencode(params)
        return f"{self.base_url}/v1/storage/download?{query}"

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        mime_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObjectMetadata:
        target_bucket = bucket or self.config.s3_bucket_name
        path = self._get_path(target_bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        sha256 = compute_bytes_sha256(data)

        def _write():
            path.write_bytes(data)

        await asyncio.to_thread(_write)

        return StorageObjectMetadata(
            bucket=target_bucket,
            key=key,
            size_bytes=len(data),
            content_type=mime_type,
            sha256=sha256,
            etag=sha256[:32],
            last_modified=datetime.now(timezone.utc),
            custom_metadata=metadata or {},
        )

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        target_bucket = bucket or self.config.s3_bucket_name
        path = self._get_path(target_bucket, key)
        if not path.exists():
            raise NotFoundError(f"Object {key} not found in bucket {target_bucket}")

        def _read():
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    async def head_object(self, bucket: str, key: str) -> Optional[StorageObjectMetadata]:
        target_bucket = bucket or self.config.s3_bucket_name
        path = self._get_path(target_bucket, key)
        if not path.exists():
            return None

        stat = path.stat()
        sha256 = compute_bytes_sha256(path.read_bytes())
        return StorageObjectMetadata(
            bucket=target_bucket,
            key=key,
            size_bytes=stat.st_size,
            content_type="application/octet-stream",
            sha256=sha256,
            etag=sha256[:32],
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            custom_metadata={},
        )

    async def delete_object(self, bucket: str, key: str) -> bool:
        target_bucket = bucket or self.config.s3_bucket_name
        path = self._get_path(target_bucket, key)
        if path.exists():
            path.unlink()
        return True
