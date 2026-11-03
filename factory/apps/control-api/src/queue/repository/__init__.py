"""
factory/apps/control-api/src/queue/repository/__init__.py
Repository factory for Factory v4 Lite.
Default and only repository backend is InMemoryJobStore.
"""
from typing import Optional

from .base import BaseJobRepository
from .memory import CapacityExceededError, InMemoryJobStore

_default_repository: Optional[InMemoryJobStore] = None


def get_job_repository(
    backend: Optional[str] = None,
    db_url: Optional[str] = None,
    max_capacity: Optional[int] = None,
    default_ttl_seconds: Optional[int] = None,
) -> InMemoryJobStore:
    """Returns the single-process InMemoryJobStore for Factory v4 Lite."""
    global _default_repository
    if _default_repository is None:
        _default_repository = InMemoryJobStore(
            max_capacity=max_capacity,
            default_ttl_seconds=default_ttl_seconds,
        )
    return _default_repository


__all__ = [
    "BaseJobRepository",
    "InMemoryJobStore",
    "CapacityExceededError",
    "get_job_repository",
]
