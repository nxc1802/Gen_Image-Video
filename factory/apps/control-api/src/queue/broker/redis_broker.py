"""
factory/apps/control-api/src/queue/broker/redis_broker.py
Production Redis queue broker using sorted sets and pub/sub.
"""
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

import redis.asyncio as redis

from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker


class RedisQueueBroker(BaseQueueBroker):
    """Production Redis broker using priority sorted sets and pub/sub channels."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None

    async def initialize(self) -> None:
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        # Fail startup immediately if Redis is unreachable or credentials are invalid.
        await self.client.ping()

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None

    def _calc_score(self, priority: int) -> float:
        # Score = (priority * 1e12) - (current_time_ms)
        # Higher priority has higher score; older jobs have slightly higher score
        now_ms = time.time() * 1000.0
        return (priority * 1_000_000_000_000.0) - now_ms

    async def enqueue(self, job_id: str, service: str, priority: int = 10) -> None:
        key = f"factory:queue:{service}"
        score = self._calc_score(priority)
        await self.client.zadd(key, {job_id: score})

    async def dequeue(self, service: str) -> Optional[str]:
        key = f"factory:queue:{service}"
        # Pop member with highest score (highest priority first)
        results = await self.client.zpopmax(key, count=1)
        if results:
            member, _ = results[0]
            return member
        return None

    async def remove(self, job_id: str, service: str) -> bool:
        key = f"factory:queue:{service}"
        removed = await self.client.zrem(key, job_id)
        return removed > 0

    async def get_queue_depth(self, service: Optional[str] = None) -> int:
        if service:
            return await self.client.zcard(f"factory:queue:{service}")
        keys = await self.client.keys("factory:queue:*")
        total = 0
        for k in keys:
            total += await self.client.zcard(k)
        return total

    async def publish_event(self, channel: str, event_type: str, data: Dict[str, Any]) -> None:
        payload = {"event": event_type, "data": data, "timestamp": time.time()}
        await self.client.publish(channel, json.dumps(payload))

    async def subscribe_events(self, channel: str) -> AsyncGenerator[Dict[str, Any], None]:
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    try:
                        data = json.loads(msg["data"])
                        yield data
                    except Exception:
                        pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
