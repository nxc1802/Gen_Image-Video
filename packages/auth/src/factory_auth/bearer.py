"""
packages/auth/src/factory_auth/bearer.py
FastAPI dependency enforcing Bearer authentication with role differentiation:
- Admin role via static FACTORY_API_KEY (constant-time verification)
- Worker role via worker JWT with sub=worker_id and role='worker'
- User role via standard user JWT
"""
import secrets
from typing import Annotated, Any, Dict, Optional
from fastapi import Depends, Header, Request

from factory_contracts.errors import UnauthorizedError
from factory_auth.config import AuthConfig, load_auth_config
from factory_auth.jwt_auth import decode_jwt_token


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
    Validates Bearer token in constant time.
    Supports both static FACTORY_API_KEY and cryptographic JWT verification with role differentiation.
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

    # 1. Constant-time API Key verification (Admin Role)
    expected_key = active_config.api_key if active_config else ""
    if expected_key and secrets.compare_digest(token, expected_key):
        user_context = {"sub": "system", "role": "admin"}
        request.state.user = user_context
        return user_context

    # 2. JWT verification if token is dot-separated and jwt_secret is configured
    if active_config and active_config.jwt_secret and token.count(".") == 2:
        payload = decode_jwt_token(
            token,
            active_config.jwt_secret,
            algorithm=active_config.jwt_algorithm,
            audience=active_config.jwt_audience,
            issuer=active_config.jwt_issuer,
        )

        # Role extraction and normalization
        raw_role = payload.get("role")
        if not raw_role and "roles" in payload:
            roles_list = payload["roles"]
            raw_role = roles_list[0] if isinstance(roles_list, list) and roles_list else str(roles_list)
        role = str(raw_role).lower() if raw_role else "user"

        sub = payload.get("sub")
        worker_id = payload.get("worker_id")

        if role == "worker":
            worker_id = worker_id or sub
            if not worker_id:
                raise UnauthorizedError(message="Worker token must contain 'sub' or 'worker_id' claim.")
            sub = worker_id

        user_context = dict(payload)
        user_context["role"] = role
        user_context["sub"] = sub or "user"
        if worker_id:
            user_context["worker_id"] = worker_id

        request.state.user = user_context
        return user_context

    # Neither matched: reject
    raise UnauthorizedError(message="Invalid API key or token.")
