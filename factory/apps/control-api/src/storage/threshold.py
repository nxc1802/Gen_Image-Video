"""
factory/apps/control-api/src/storage/threshold.py
Strict 1MB threshold enforcement rejecting inline Base64 media with HTTP 413.
Attaches deprecation warning headers for payloads <= 1MB.
"""
import base64
import re
from typing import Any, Dict, List, Optional, Tuple
from factory_contracts.errors import PayloadTooLargeError

# 1MB Threshold = 1,048,576 bytes
MAX_INLINE_BYTES = 1024 * 1024

BASE64_ALPHABET_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "Wed, 01 Jan 2027 00:00:00 GMT",
    "Warning": '299 - "Inline base64 payloads are deprecated in Factory v4; use S3 signed URLs."',
}


def estimate_base64_size(b64_str: str) -> int:
    """Estimates decoded binary size from a Base64 string."""
    clean = b64_str.strip()
    if clean.startswith("data:") and ";base64," in clean:
        clean = clean.split(";base64,", 1)[1]
    
    # Remove whitespace
    clean = clean.replace("\n", "").replace("\r", "").replace(" ", "")
    if not clean:
        return 0
    padding = clean.count("=")
    return max(0, (len(clean) * 3 // 4) - padding)


def is_raw_base64_media(s: str) -> bool:
    """
    Determines whether a string is raw unadorned Base64-encoded media.
    Rejects strings with natural language whitespace (spaces, tabs),
    punctuation (., !, ?, etc.), or non-Base64 characters.
    """
    if len(s) <= 1024:
        return False
    if s.startswith("http://") or s.startswith("https://"):
        return False

    # Strip common MIME newlines
    clean = s.strip().replace("\r", "").replace("\n", "")

    # Standard Base64 media never contains spaces or natural language punctuation
    if " " in clean or "\t" in clean:
        return False

    if len(clean) % 4 != 0:
        return False

    if not BASE64_ALPHABET_PATTERN.fullmatch(clean):
        return False

    try:
        sample = clean[:1024]
        base64.b64decode(sample, validate=True)
        return True
    except Exception:
        return False


def find_inline_base64(obj: Any, path: str = "inputs") -> List[Tuple[str, int]]:
    """
    Recursively inspects a dictionary or list for all base64 media strings.
    Returns a list of (field_path, estimated_size) tuples.
    """
    results: List[Tuple[str, int]] = []

    if isinstance(obj, str):
        # Detect Data URI
        if obj.startswith("data:") and ";base64," in obj:
            size = estimate_base64_size(obj)
            results.append((path, size))
        # Or raw base64 string exceeding ~1KB that strictly matches encoded media
        elif is_raw_base64_media(obj):
            size = estimate_base64_size(obj)
            if size > 512:
                results.append((path, size))

    elif isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(find_inline_base64(v, f"{path}.{k}"))

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            results.extend(find_inline_base64(item, f"{path}[{idx}]"))

    return results


def enforce_payload_thresholds(inputs: Dict[str, Any], request_id: Optional[str] = None) -> None:
    """
    Validates that no inline Base64 media string in inputs exceeds 1MB.
    Raises PayloadTooLargeError (HTTP 413) if any field exceeds MAX_INLINE_BYTES.
    """
    matches = find_inline_base64(inputs)
    for field_path, size_bytes in matches:
        if size_bytes > MAX_INLINE_BYTES:
            raise PayloadTooLargeError(
                message=(
                    f"Inline base64 payload ({size_bytes:,} bytes) in field '{field_path}' exceeds the 1MB threshold. "
                    f"Inline base64 is deprecated for media. Please upload the asset to the "
                    f"Asset Service (POST /v1/assets/upload) and provide the returned s3_url."
                ),
                details={
                    "threshold_bytes": MAX_INLINE_BYTES,
                    "received_bytes": size_bytes,
                    "field": field_path,
                },
            )


def should_attach_deprecation_header(inputs: Dict[str, Any]) -> bool:
    """
    Returns True if input contains any inline Base64 payload <= 1MB
    AND no inline Base64 payload exceeding 1MB.
    """
    matches = find_inline_base64(inputs)
    if not matches:
        return False
    has_valid_inline = any(0 < size <= MAX_INLINE_BYTES for _, size in matches)
    has_oversized = any(size > MAX_INLINE_BYTES for _, size in matches)
    return has_valid_inline and not has_oversized

