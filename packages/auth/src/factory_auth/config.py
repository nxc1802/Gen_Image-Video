"""
factory_auth/config.py
Authentication Configuration and Fail-Fast Environment Validation
"""
import os
import secrets
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthConfig:
    api_key: str
    jwt_secret: Optional[str]
    environment: str
    jwt_algorithm: str = "HS256"
    jwt_audience: Optional[str] = None


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
    return AuthConfig(
        api_key=raw_key,
        jwt_secret=jwt_secret,
        environment=env,
        jwt_algorithm=jwt_algorithm,
        jwt_audience=jwt_audience,
    )
