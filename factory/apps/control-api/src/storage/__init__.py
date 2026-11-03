"""
factory/apps/control-api/src/storage/__init__.py
Public exports for Factory v4 Lite storage subsystem.
"""
from .config import StorageConfig, load_storage_config
from .provider import StorageProvider, PresignedUpload, StorageObjectMetadata
from .local import LocalStorageProvider
from .keys import format_artifact_key, format_asset_key, sanitize_id
from .integrity import compute_bytes_sha256, compute_stream_sha256, verify_sha256
from .threshold import (
    MAX_INLINE_BYTES,
    DEPRECATION_HEADERS,
    enforce_payload_thresholds,
    should_attach_deprecation_header,
)
from .router import router as storage_router, get_storage_provider
from .scratch_disk import ScratchDiskConfig, ScratchDiskManager, get_scratch_disk

# S3 is deprecated in Factory v4 Lite; alias to LocalStorageProvider for compatibility
S3StorageProvider = LocalStorageProvider

__all__ = [
    "StorageConfig",
    "load_storage_config",
    "StorageProvider",
    "PresignedUpload",
    "StorageObjectMetadata",
    "S3StorageProvider",
    "LocalStorageProvider",
    "ScratchDiskConfig",
    "ScratchDiskManager",
    "get_scratch_disk",
    "format_artifact_key",
    "format_asset_key",
    "sanitize_id",
    "compute_bytes_sha256",
    "compute_stream_sha256",
    "verify_sha256",
    "MAX_INLINE_BYTES",
    "DEPRECATION_HEADERS",
    "enforce_payload_thresholds",
    "should_attach_deprecation_header",
    "storage_router",
    "get_storage_provider",
]
