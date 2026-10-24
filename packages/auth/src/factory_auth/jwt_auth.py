"""
JWT verification with sanitized client-facing failures.
"""
from typing import Any, Dict, Optional

import jwt

from factory_contracts.errors import UnauthorizedError


def decode_jwt_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        options = {"verify_aud": audience is not None}
        return jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            audience=audience,
            options=options,
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError(message="JWT token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # Never reflect parser/signature internals back to untrusted callers.
        raise UnauthorizedError(message="Invalid JWT token.") from exc
