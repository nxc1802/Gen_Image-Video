"""
packages/auth/src/factory_auth/config.py
Authentication configuration for Factory v4 Lite.
Enforces single FACTORY_API_KEY with constant-time verification.
"""
from dataclasses import dataclass
import logging
import os
import secrets
from typing import Optional

logger = logging.getLogger(__name__)

class AuthConfigError(ValueError, RuntimeError):
    """Raised when authentication configuration is invalid."""
    pass


@dataclass(frozen=True)
class AuthConfig:
    api_key: str
    environment: str = "development"
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_audience: Optional[str] = None
    jwt_issuer: Optional[str] = None


def load_auth_config() -> AuthConfig:
    """
    Loads authentication configuration for Factory v4 Lite.
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

    return AuthConfig(
        api_key=raw_key,
        environment=env,
        jwt_secret=None,
        jwt_algorithm="HS256",
        jwt_audience=None,
        jwt_issuer=None,
    )

