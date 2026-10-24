"""
Queue broker factory.

Redis is mandatory in production. The in-memory broker is only for local
development/tests and can never be selected as an implicit production fallback.
"""
import os
from typing import Optional

from .base import BaseQueueBroker
from .memory_broker import MemoryQueueBroker
from .redis_broker import RedisQueueBroker

__all__ = [
    "BaseQueueBroker",
    "MemoryQueueBroker",
    "RedisQueueBroker",
    "get_queue_broker",
]


def get_queue_broker(redis_url: Optional[str] = None) -> BaseQueueBroker:
    env = os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower()
    url = redis_url or os.environ.get("FACTORY_REDIS_URL") or os.environ.get("REDIS_URL")
    if url:
        return RedisQueueBroker(redis_url=url)
    if env == "production":
        raise RuntimeError("REDIS_URL or FACTORY_REDIS_URL is required in production.")
    return MemoryQueueBroker()
