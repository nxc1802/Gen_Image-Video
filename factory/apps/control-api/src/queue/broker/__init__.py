"""
factory/apps/control-api/src/queue/broker/__init__.py
Queue broker exports and loader.
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
    url = redis_url or os.environ.get("FACTORY_REDIS_URL") or os.environ.get("REDIS_URL")
    if url:
        return RedisQueueBroker(redis_url=url)
    return MemoryQueueBroker()
