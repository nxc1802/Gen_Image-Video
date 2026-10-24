"""
services/text/src/multimodal.py
Multimodal inputs parser and SSRF security validator.
"""
import re
from typing import Any, Dict, List, Tuple
from factory_contracts.errors import SSRFBlockedError, InvalidMediaError

BLOCKED_SSRF_PATTERNS = [
    r"^https?://127\.",
    r"^https?://localhost",
    r"^https?://\[::1\]",
    r"^https?://10\.",
    r"^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.",
    r"^https?://192\.168\.",
    r"^https?://169\.254\.",
    r"^https?://0\.0\.0\.0",
    r"^https?://metadata\.google\.internal",
]


def validate_url_security(url: str) -> None:
    """Validates that a URL does not target loopback, private network, or metadata endpoints."""
    if not url:
        return
    for pattern in BLOCKED_SSRF_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            raise SSRFBlockedError(
                message=f"Access to requested URL host or IP '{url}' is blocked for security."
            )


def extract_content_text_and_media(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts concatenated text and media parts (image_url, video_url) from message content.
    Validates any media URLs for SSRF.
    """
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return str(content or ""), []

    text_parts: List[str] = []
    media_parts: List[Dict[str, Any]] = []

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type == "text":
            text_parts.append(part.get("text", ""))
        elif part_type in ("image_url", "image"):
            img_data = part.get("image_url", {})
            if isinstance(img_data, dict):
                url = img_data.get("url", "")
            else:
                url = str(img_data)
            validate_url_security(url)
            media_parts.append({"type": "image", "url": url})
        elif part_type in ("video_url", "video"):
            vid_data = part.get("video_url", {})
            if isinstance(vid_data, dict):
                url = vid_data.get("url", "")
            else:
                url = str(vid_data)
            validate_url_security(url)
            media_parts.append({"type": "video", "url": url})

    return " ".join(text_parts), media_parts
