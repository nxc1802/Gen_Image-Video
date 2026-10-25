"""
factory_auth/bearer.py
FastAPI Dependency for Mandatory Bearer Authentication
"""
import secrets
from typing import Annotated, Dict, Any, Optional
from fastapi import Header, Request, Depends
from factory_contracts.errors import UnauthorizedError
from factory_auth.config import AuthConfig, load_auth_config
from factory_auth.jwt_auth import decode_jwt_token


def get_auth_config(request: Request) -> AuthConfig:
    # Try request app state first, otherwise load
    if hasattr(request, "app") and hasattr(request.app.state, "auth_config"):
        return request.app.state.auth_config
    return load_auth_config()


async def verify_bearer_token(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    config: Annotated[AuthConfig, Depends(get_auth_config)] = None,
) -> Dict[str, Any]:
    """
    Validates Bearer token in constant time.
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

    # 1. Constant-time API Key verification
    if config and config.api_key and secrets.compare_digest(token, config.api_key):
        user_context = {"sub": "system:api_key", "role": "admin"}
        request.state.user = user_context
        return user_context

    # 2. JWT verification if token is dot-separated
    if config and config.jwt_secret and token.count(".") == 2:
        try:
            payload = decode_jwt_token(
                token,
                config.jwt_secret,
                algorithm=config.jwt_algorithm,
                audience=config.jwt_audience,
            )
            request.state.user = payload
            return payload
        except Exception:
            raise UnauthorizedError(message="Invalid or expired JWT token.")

    raise UnauthorizedError(message="Invalid API token.")
