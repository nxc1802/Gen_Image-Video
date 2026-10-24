"""
Strict HTTP adapter for real enhancement backends.

The Factory API does not fabricate media when Real-ESRGAN/RIFE/audio enhancement
workers are absent. Each processor delegates to a configured real backend or
returns MODEL_UNAVAILABLE.
"""
import os
from typing import Any, Dict

import httpx

from factory_contracts.errors import ModelUnavailableError


def call_backend(env_var: str, path: str, payload: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
    base_url = os.environ.get(env_var, "").strip().rstrip("/")
    if not base_url:
        raise ModelUnavailableError(
            message=f"Real enhancement backend is not configured for {path}.",
            details={"required_env": env_var},
        )
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("ENHANCEMENT_BACKEND_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url}{path}", json=payload, headers=headers)
            if response.status_code >= 400:
                raise ModelUnavailableError(
                    message="Enhancement backend rejected the request.",
                    details={"status_code": response.status_code, "path": path},
                )
            data = response.json()
    except ModelUnavailableError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelUnavailableError(
            message="Enhancement backend is unavailable or returned an invalid response.",
            details={"error_type": type(exc).__name__, "path": path},
        ) from exc

    required = (
        "artifact_id",
        "type",
        "mime_type",
        "size_bytes",
        "sha256",
        "storage_path",
        "download_url",
    )
    if any(key not in data for key in required):
        raise ModelUnavailableError(
            message="Enhancement backend returned incomplete artifact metadata.",
            details={"path": path, "required_fields": list(required)},
        )
    return data
