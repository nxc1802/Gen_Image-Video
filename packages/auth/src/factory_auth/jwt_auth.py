"""
packages/auth/src/factory_auth/jwt_auth.py
Deprecated JWT verification module for Factory v4 Lite.
JWT verification has been replaced by single FACTORY_API_KEY.
"""
from typing import Any, Dict, Optional, Set

from factory_contracts.errors import UnauthorizedError

ALLOWED_ALGORITHMS: Set[str] = {"HS256", "RS256"}


def decode_jwt_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
    audience: Optional[str] = None,
    issuer: Optional[str] = None,
) -> Dict[str, Any]:
    """Decodes and validates a JWT token using strict security constraints."""
    if algorithm not in ALLOWED_ALGORITHMS:
        raise UnauthorizedError(message=f"Disallowed or unsupported JWT algorithm: '{algorithm}'.")

    try:
        import jwt
    except ImportError:
        raise UnauthorizedError(message="JWT library not available.")

    try:
        options = {
            "verify_aud": audience is not None,
            "verify_iss": issuer is not None,
            "require": ["exp"],
        }
        return jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            audience=audience,
            issuer=issuer,
            options=options,
        )
    except jwt.MissingRequiredClaimError as exc:
        raise UnauthorizedError(message=f"JWT token missing required claim: {exc}") from exc
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError(message="JWT token has expired.") from exc
    except jwt.InvalidIssuerError as exc:
        raise UnauthorizedError(message=f"Invalid JWT issuer: {exc}") from exc
    except jwt.InvalidAudienceError as exc:
        raise UnauthorizedError(message=f"Invalid JWT audience: {exc}") from exc
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError(message="Invalid JWT token.") from exc

