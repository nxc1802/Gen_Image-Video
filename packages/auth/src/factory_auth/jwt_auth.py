"""
packages/auth/src/factory_auth/jwt_auth.py
JWT verification with sanitized client-facing failures and strict allowlist enforcement.
"""
from typing import Any, Dict, Optional, Set

import jwt

from factory_contracts.errors import UnauthorizedError

ALLOWED_ALGORITHMS: Set[str] = {"HS256", "RS256"}


def decode_jwt_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
    audience: Optional[str] = None,
    issuer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decodes and validates a JWT token against an algorithm allowlist, mandatory 'exp' claim,
    and optional audience / issuer claims.
    Sanitizes all internal exceptions into user-friendly UnauthorizedError messages.
    """
    if algorithm not in ALLOWED_ALGORITHMS:
        raise UnauthorizedError(
            message=f"Disallowed or unsupported JWT algorithm '{algorithm}'. "
            f"Allowed algorithms: {sorted(ALLOWED_ALGORITHMS)}."
        )

    options: Dict[str, Any] = {
        "verify_signature": True,
        "require": ["exp"],
        "verify_exp": True,
        "verify_aud": audience is not None,
        "verify_iss": issuer is not None,
    }

    try:
        return jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            audience=audience,
            issuer=issuer,
            options=options,
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError(message="JWT token has expired.") from exc
    except jwt.MissingRequiredClaimError as exc:
        raise UnauthorizedError(message="JWT token is missing required claim 'exp'.") from exc
    except jwt.InvalidIssuerError as exc:
        raise UnauthorizedError(message="Invalid JWT issuer.") from exc
    except jwt.InvalidAudienceError as exc:
        raise UnauthorizedError(message="Invalid JWT audience.") from exc
    except jwt.InvalidAlgorithmError as exc:
        raise UnauthorizedError(message="Invalid JWT algorithm.") from exc
    except jwt.InvalidTokenError as exc:
        # Never reflect parser/signature internals back to untrusted callers.
        raise UnauthorizedError(message="Invalid JWT token.") from exc
