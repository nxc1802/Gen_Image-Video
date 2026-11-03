"""
packages/auth/src/factory_auth/bearer.py
FastAPI dependency enforcing single FACTORY_API_KEY with constant-time verification.
"""
import secrets
from typing import Annotated, Any, Dict, Optional
from fastapi import Depends, Header, Request

from factory_contracts.errors import UnauthorizedError
from factory_auth.config import AuthConfig, load_auth_config


def get_auth_config(request: Request) -> AuthConfig:
    if hasattr(request, "app") and hasattr(request.app.state, "auth_config"):
        return request.app.state.auth_config
    return load_auth_config()


async def verify_bearer_token(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    config: Annotated[AuthConfig, Depends(get_auth_config)] = None,
) -> Dict[str, Any]:
    """
    Validates single FACTORY_API_KEY using constant-time string comparison (secrets.compare_digest).
    Rejects missing, empty, or invalid tokens with HTTP 401 UnauthorizedError.
    """
    if not authorization:
        raise UnauthorizedError(
            message="Missing Authorization header. Expected 'Authorization: Bearer <token>'."
        )

    parts = authorization.split(maxsplit=1)
    if len(parts) == 0 or parts[0].lower() != "bearer":
        raise UnauthorizedError(
            message="Invalid Authorization header format. Expected 'Bearer <token>'."
        )

    if len(parts) < 2 or not parts[1].strip():
        raise UnauthorizedError(message="Bearer token is empty.")

    token = parts[1].strip()

    active_config = config or load_auth_config()
    expected_key = active_config.api_key if active_config else ""

    if not expected_key or not secrets.compare_digest(token, expected_key):
        raise UnauthorizedError(message="Invalid API key.")

    user_context = {"sub": "system", "role": "admin"}
    request.state.user = user_context
    return user_context
