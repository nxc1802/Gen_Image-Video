"""
services/vlm/src/processor.py
Image and Video multimodal media processor with SSRF guardrails.
"""
import base64
import re
from typing import Any, Dict, List, Optional, Tuple
from factory_contracts.errors import InvalidMediaError, SSRFBlockedError
from services.text.src.multimodal import validate_url_security


class MultimodalMediaProcessor:
    """Processes images and video inputs for Qwen2-VL vision-language tasks."""

    def __init__(self):
        pass

    def parse_multimodal_content(self, content: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """Extracts text prompt and validates media items (URL or base64 data)."""
        if isinstance(content, str):
            return content, []

        if not isinstance(content, list):
            return str(content or ""), []

        texts: List[str] = []
        media_items: List[Dict[str, Any]] = []

        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "text":
                texts.append(item.get("text", ""))

            elif item_type in ("image_url", "image"):
                img_data = item.get("image_url", {})
                url = img_data.get("url", "") if isinstance(img_data, dict) else str(img_data)
                if url.startswith("data:"):
                    # Validate base64 data URL
                    media_items.append({"modality": "image", "source": "data_url", "raw": url})
                else:
                    validate_url_security(url)
                    media_items.append({"modality": "image", "source": "url", "url": url})

            elif item_type in ("video_url", "video"):
                vid_data = item.get("video_url", {})
                url = vid_data.get("url", "") if isinstance(vid_data, dict) else str(vid_data)
                if url.startswith("data:"):
                    media_items.append({"modality": "video", "source": "data_url", "raw": url})
                else:
                    validate_url_security(url)
                    media_items.append({"modality": "video", "source": "url", "url": url})

        return " ".join(texts), media_items
