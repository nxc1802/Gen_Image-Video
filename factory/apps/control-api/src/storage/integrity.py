"""
factory/apps/control-api/src/storage/integrity.py
Streaming SHA-256 calculation and verification utilities.
"""
import hashlib
from typing import AsyncIterable, BinaryIO

CHUNK_SIZE = 64 * 1024  # 64KB


def compute_bytes_sha256(data: bytes) -> str:
    """Calculates hex-encoded SHA-256 for a bytes buffer."""
    return hashlib.sha256(data).hexdigest()


def compute_stream_sha256(stream: BinaryIO) -> str:
    """Calculates hex-encoded SHA-256 from a readable binary stream."""
    hasher = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        hasher.update(chunk)
    return hasher.hexdigest()


async def compute_async_stream_sha256(async_stream: AsyncIterable[bytes]) -> str:
    """Calculates hex-encoded SHA-256 from an async iterable stream of bytes."""
    hasher = hashlib.sha256()
    async for chunk in async_stream:
        hasher.update(chunk)
    return hasher.hexdigest()


def verify_sha256(data: bytes, expected_sha256: str) -> bool:
    """Constant-time verification of hex SHA-256 against bytes."""
    import secrets
    actual = compute_bytes_sha256(data)
    return secrets.compare_digest(actual.lower(), expected_sha256.lower())
