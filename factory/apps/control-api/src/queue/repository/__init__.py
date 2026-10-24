"""
Job repository factory.

Production must use durable PostgreSQL. SQLite is intentionally limited to
development/test so a missing DATABASE_URL can never silently downgrade job
durability.
"""
import os
from typing import Optional

from .base import BaseJobRepository
from .postgres import PostgresJobRepository
from .sqlite import SqliteJobRepository

__all__ = [
    "BaseJobRepository",
    "SqliteJobRepository",
    "PostgresJobRepository",
    "get_job_repository",
]


def get_job_repository(
    backend: Optional[str] = None,
    db_url: Optional[str] = None,
) -> BaseJobRepository:
    env = os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower()
    be = (backend or os.environ.get("FACTORY_DB_BACKEND", "postgres" if env == "production" else "sqlite")).lower().strip()
    url = db_url or os.environ.get("DATABASE_URL")

    if be == "postgres":
        if not url:
            raise RuntimeError("DATABASE_URL is required when FACTORY_DB_BACKEND=postgres.")
        return PostgresJobRepository(dsn=url)

    if be != "sqlite":
        raise RuntimeError(f"Unsupported FACTORY_DB_BACKEND '{be}'.")

    if env == "production":
        raise RuntimeError("SQLite job storage is forbidden in production.")

    sqlite_path = os.environ.get("SQLITE_DB_PATH", ".factory_storage/factory.db")
    return SqliteJobRepository(db_path=sqlite_path)
