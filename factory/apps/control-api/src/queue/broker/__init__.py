"""
factory/apps/control-api/src/queue/broker/__init__.py
Queue broker factory for Factory v4 Lite.
Default and only broker backend is MemoryQueueBroker.
"""
from typing import Optional

from .base import BaseQueueBroker
from .memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler

# Backward compatibility alias
RedisQueueBroker = InMemoryScheduler

_default_broker: Optional[InMemoryScheduler] = None


def get_queue_broker(redis_url: Optional[str] = None) -> InMemoryScheduler:
    """Returns the in-memory queue broker for Factory v4 Lite."""
    global _default_broker
    if _default_broker is None:
        _default_broker = InMemoryScheduler()
    return _default_broker


__all__ = [
    "BaseQueueBroker",
    "MemoryQueueBroker",
    "RedisQueueBroker",
    "InMemoryScheduler",
    "get_queue_broker",
]
