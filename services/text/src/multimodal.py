"""
Shared multimodal parsing and SSRF validation for compatibility gateways.

Production vLLM VLM serving additionally enforces --allowed-media-domains and
VLLM_MEDIA_URL_ALLOW_REDIRECTS=0.
"""
from typing import Any, Dict, List, Tuple

from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory_contracts.errors import InvalidMediaError, SSRFBlockedError


def validate_url_security(url: str) -> None:
    if not url:
        raise InvalidMediaError(message="Media URL cannot be empty.")
    if url.startswith("data:"):
        # Data URLs are handled in-process and do not trigger outbound network I/O.
        return
    validate_target_url(url)


def extract_content_text_and_media(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
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
        if part_type in ("text", "input_text"):
            text_parts.append(str(part.get("text", "")))
        elif part_type in ("image_url", "image"):
            raw = part.get("image_url", part.get("url", ""))
            url = raw.get("url", "") if isinstance(raw, dict) else str(raw)
            validate_url_security(url)
            media_parts.append({"type": "image", "url": url})
        elif part_type in ("video_url", "video"):
            raw = part.get("video_url", part.get("url", ""))
            url = raw.get("url", "") if isinstance(raw, dict) else str(raw)
            validate_url_security(url)
            media_parts.append({"type": "video", "url": url})

    return " ".join(text_parts), media_parts
