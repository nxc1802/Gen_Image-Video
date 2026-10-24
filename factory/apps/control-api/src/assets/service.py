"""
factory/apps/control-api/src/assets/service.py
SSRF-Safe Asset Ingestion and Media Management Service
"""
import io
import hashlib
import uuid
from typing import Dict, Any, Optional
import httpx
import httpcore

from factory_contracts.errors import (
    PayloadTooLargeError,
    InvalidMediaError,
    SSRFBlockedError,
)
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.assets.media_validator import validate_media_content
from factory.apps.control_api.src.assets.transport import PinningAsyncHTTPTransport

MAX_PAYLOAD_SIZE = 50 * 1024 * 1024  # 50MB


class AssetService:
    """Service governing remote ingestion and local asset validation."""

    def __init__(self, max_size_bytes: int = MAX_PAYLOAD_SIZE):
        self.max_size_bytes = max_size_bytes

    def ingest_bytes(self, data: bytes, claimed_mime: Optional[str] = None) -> Dict[str, Any]:
        """Validates and stores raw media bytes."""
        if len(data) > self.max_size_bytes:
            raise PayloadTooLargeError(
                message=f"Payload size {len(data)} bytes exceeds maximum permitted limit ({self.max_size_bytes} bytes)."
            )

        stream = io.BytesIO(data)
        header = data[:32]
        detected_mime = validate_media_content(header, stream, claimed_mime)
        sha256 = hashlib.sha256(data).hexdigest()
        asset_id = f"ast-{uuid.uuid4()}"

        return {
            "artifact_id": asset_id,
            "mime_type": detected_mime,
            "size_bytes": len(data),
            "sha256": sha256,
            "storage_path": f"assets/{sha256[:2]}/{sha256}",
            "download_url": f"/v1/assets/{asset_id}/download",
        }

    async def ingest_url(self, target_url: str) -> Dict[str, Any]:
        """
        Validates target URL against SSRF, pins TCP connection to pre-verified IPs,
        streams content with byte limits, and verifies media integrity.
        """
        # 1. SSRF Validation and DNS Resolution Pinning
        hostname, resolved_ips = validate_target_url(target_url)
        allowed_ips = [str(ip) for ip in resolved_ips]

        # 2. Secure Streaming download with pinned transport and byte cap
        downloaded = bytearray()
        transport = PinningAsyncHTTPTransport(allowed_ips=allowed_ips)

        async with httpx.AsyncClient(
            transport=transport,
            timeout=30.0,
            follow_redirects=False,
        ) as client:
            try:
                async with client.stream("GET", target_url) as response:
                    # Enforce strict rejection of redirects (to prevent unpinned redirect hops)
                    if response.status_code in (301, 302, 303, 307, 308):
                        raise InvalidMediaError(
                            message=f"Remote server returned HTTP redirect {response.status_code}. Remote redirects are prohibited."
                        )
                    if response.status_code >= 400 or response.status_code != 200:
                        raise InvalidMediaError(message=f"Remote server returned HTTP {response.status_code}")

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            parsed_length = int(content_length.strip())
                            if parsed_length < 0:
                                raise InvalidMediaError(
                                    message=f"Remote asset returned negative Content-Length header: {content_length!r}"
                                )
                            if parsed_length > self.max_size_bytes:
                                raise PayloadTooLargeError(
                                    message=f"Remote asset declares size {content_length} bytes exceeding {self.max_size_bytes} limit."
                                )
                        except (ValueError, TypeError) as e:
                            raise InvalidMediaError(
                                message=f"Remote asset returned malformed Content-Length header: {content_length!r}"
                            ) from e

                    async for chunk in response.aiter_bytes():
                        downloaded.extend(chunk)
                        if len(downloaded) > self.max_size_bytes:
                            raise PayloadTooLargeError(
                                message=f"Remote payload exceeded {self.max_size_bytes} bytes during streaming."
                            )
            except (httpx.RequestError, httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                raise SSRFBlockedError(message=f"Failed to fetch remote asset: {e}")

        return self.ingest_bytes(bytes(downloaded))
