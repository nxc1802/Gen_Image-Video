"""
factory_auth/config.py
Authentication Configuration and Fail-Fast Environment Validation
"""
import os
import secrets
import logging
from dataclasses import dataclass
from typing import Optional
from .jwt_auth import ALLOWED_ALGORITHMS

logger = logging.getLogger(__name__)


class AuthConfigError(ValueError, RuntimeError):
    """Raised when authentication configuration is invalid."""
    pass


@dataclass(frozen=True)
class AuthConfig:
    api_key: str
    jwt_secret: Optional[str]
    environment: str
    jwt_algorithm: str = "HS256"
    jwt_audience: Optional[str] = None
    jwt_issuer: Optional[str] = None

    def __post_init__(self) -> None:
        if self.jwt_algorithm not in ALLOWED_ALGORITHMS:
            raise AuthConfigError(
                f"Disallowed or unsupported JWT algorithm '{self.jwt_algorithm}'. "
                f"Allowed algorithms: {sorted(ALLOWED_ALGORITHMS)}."
            )


def load_auth_config() -> AuthConfig:
    """
    Loads authentication configuration.
    Enforces fail-fast startup guard in production if FACTORY_API_KEY is missing or empty.
    """
    env = os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower()
    raw_key = os.environ.get("FACTORY_API_KEY", "").strip()

    if not raw_key:
        if env == "production":
            raise RuntimeError(
                "FATAL: FACTORY_API_KEY environment variable is missing or empty in production mode. "
                "Startup aborted."
            )
        # Development fallback: generate ephemeral secure key
        raw_key = f"dev_{secrets.token_urlsafe(32)}"
        logger.warning(
            "\n" + "=" * 70 + "\n"
            f"⚠️  WARNING: FACTORY_API_KEY was not set! Generated ephemeral dev key:\n"
            f"   Authorization: Bearer {raw_key}\n"
            + "=" * 70
        )

    jwt_secret = os.environ.get("JWT_SECRET_KEY", os.environ.get("FACTORY_JWT_SECRET", "")).strip() or None
    jwt_algorithm = os.environ.get("JWT_ALGORITHM", "HS256").strip() or "HS256"
    jwt_audience = os.environ.get("JWT_AUDIENCE", "").strip() or None
    jwt_issuer = os.environ.get("JWT_ISSUER", "").strip() or None

    if jwt_algorithm not in ALLOWED_ALGORITHMS:
        raise AuthConfigError(
            f"FATAL: Unsupported or disallowed JWT_ALGORITHM '{jwt_algorithm}'. "
            f"Allowed algorithms: {sorted(ALLOWED_ALGORITHMS)}."
        )

    if env == "production" and jwt_secret and not jwt_audience:
        raise RuntimeError("FATAL: JWT_AUDIENCE must be explicitly configured in production mode when JWT is enabled.")

    return AuthConfig(
        api_key=raw_key,
        jwt_secret=jwt_secret,
        environment=env,
        jwt_algorithm=jwt_algorithm,
        jwt_audience=jwt_audience,
        jwt_issuer=jwt_issuer,
    )
