"""
factory/apps/control-api/src/streaming/history.py
Ring-buffer event history tracking for SSE Last-Event-ID reconnection replay.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


@dataclass
class StreamEvent:
    event_id: int
    event_type: str
    data: Dict[str, Any]
    timestamp: float


class EventHistoryBuffer:
    """Stores the last N events per channel/job_id for replay upon client reconnection."""

    def __init__(self, max_events_per_channel: int = 100):
        self.max_events = max_events_per_channel
        self._channels: Dict[str, Deque[StreamEvent]] = {}
        self._seq: Dict[str, int] = {}

    def record(self, channel: str, event_type: str, data: Dict[str, Any], timestamp: float) -> int:
        if channel not in self._channels:
            self._channels[channel] = deque(maxlen=self.max_events)
            self._seq[channel] = 0

        self._seq[channel] += 1
        event_id = self._seq[channel]
        event = StreamEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            timestamp=timestamp,
        )
        self._channels[channel].append(event)
        return event_id

    def get_since(self, channel: str, last_event_id: str) -> List[StreamEvent]:
        """Returns all events with ID greater than last_event_id."""
        if channel not in self._channels:
            return []

        try:
            target_id = int(last_event_id.strip())
        except (ValueError, TypeError):
            return []

        return [e for e in self._channels[channel] if e.event_id > target_id]

    def clear(self, channel: str) -> None:
        self._channels.pop(channel, None)
        self._seq.pop(channel, None)
