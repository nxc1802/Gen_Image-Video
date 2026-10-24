"""
factory/apps/control-api/src/storage/provider.py
Abstract base class and models for storage providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PresignedUpload:
    upload_url: str
    method: str
    headers: Dict[str, str]
    storage_path: str
    expires_in_seconds: int


@dataclass(frozen=True)
class StorageObjectMetadata:
    bucket: str
    key: str
    size_bytes: int
    content_type: str
    sha256: str
    etag: Optional[str]
    last_modified: datetime
    custom_metadata: Dict[str, str]


class StorageProvider(ABC):
    """Abstract interface defining the factory storage engine contract."""

    @abstractmethod
    def generate_presigned_upload_url(
        self,
        bucket: str,
        key: str,
        mime_type: str,
        max_size_bytes: int,
        expires_in_seconds: int = 3600,
    ) -> PresignedUpload:
        """Generates a presigned upload URL locked to a specific MIME type and expiration."""
        ...

    @abstractmethod
    def generate_presigned_download_url(
        self,
        bucket: str,
        key: str,
        expires_in_seconds: int = 86400,
        filename: Optional[str] = None,
    ) -> str:
        """Generates a presigned download GET URL with expiration (default 24h)."""
        ...

    @abstractmethod
    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        mime_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObjectMetadata:
        """Direct upload of binary bytes to storage backend."""
        ...

    @abstractmethod
    async def download_bytes(self, bucket: str, key: str) -> bytes:
        """Downloads raw bytes of an object from storage."""
        ...

    @abstractmethod
    async def head_object(self, bucket: str, key: str) -> Optional[StorageObjectMetadata]:
        """Inspects object metadata without reading full content; returns None if not found."""
        ...

    @abstractmethod
    async def delete_object(self, bucket: str, key: str) -> bool:
        """Deletes object from storage; returns True if deleted or did not exist."""
        ...
