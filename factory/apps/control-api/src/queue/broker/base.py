"""
factory/apps/control-api/src/queue/broker/base.py
Abstract base class for Queue Broker and Event Bus.
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional


class BaseQueueBroker(ABC):
    """Abstract interface for priority queues and event pub/sub."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initializes connections and buffers."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Closes connections."""
        ...

    @abstractmethod
    async def enqueue(self, job_id: str, service: str, priority: int = 10) -> None:
        """Pushes a job onto the service priority queue."""
        ...

    @abstractmethod
    async def dequeue(self, service: str) -> Optional[str]:
        """Pops the highest-priority job ID from the service queue."""
        ...

    @abstractmethod
    async def remove(self, job_id: str, service: str) -> bool:
        """Removes a specific job from the queue (e.g. upon cancellation)."""
        ...

    @abstractmethod
    async def get_queue_depth(self, service: Optional[str] = None) -> int:
        """Returns total waiting jobs across all services or for a specific service."""
        ...

    @abstractmethod
    async def publish_event(self, channel: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publishes an event to a topic channel."""
        ...

    @abstractmethod
    async def subscribe_events(self, channel: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribes to events on a topic channel."""
        ...
