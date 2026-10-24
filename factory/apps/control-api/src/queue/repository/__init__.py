"""
factory/apps/control-api/src/queue/repository/__init__.py
Repository exports and factory loader.
"""
import os
from typing import Optional

from .base import BaseJobRepository
from .sqlite import SqliteJobRepository
from .postgres import PostgresJobRepository

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
    """Factory selecting PostgresJobRepository or SqliteJobRepository based on environment."""
    be = (backend or os.environ.get("FACTORY_DB_BACKEND", "sqlite")).lower().strip()
    url = db_url or os.environ.get("DATABASE_URL")

    if be == "postgres" and url:
        return PostgresJobRepository(dsn=url)

    # Default to SQLite fallback (in-memory or file)
    sqlite_path = os.environ.get("SQLITE_DB_PATH", ".factory_storage/factory.db")
    return SqliteJobRepository(db_path=sqlite_path)
