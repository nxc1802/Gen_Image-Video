"""
factory/apps/control-api/src/telemetry/decorators.py
Tracing decorators and async context manager using ContextVars for Factory v4 Lite.
"""
from contextlib import asynccontextmanager
import functools
from typing import Any, AsyncGenerator, Callable, Optional
import uuid

from factory.apps.control_api.src.telemetry.context import ctx_span_id


def generate_span_id() -> str:
    """Generates a 16-hex span ID."""
    return uuid.uuid4().hex[:16]


@asynccontextmanager
async def trace_scope(name: str, **attributes: Any) -> AsyncGenerator[Optional[str], None]:
    """Async context manager binding task-local span_id without external OpenTelemetry."""
    new_span_id = generate_span_id()
    token = ctx_span_id.set(new_span_id)
    try:
        yield new_span_id
    finally:
        ctx_span_id.reset(token)


def traced_span(name: str, **default_attributes: Any) -> Callable:
    """Decorator for tracing async functions."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with trace_scope(name, **default_attributes):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator


__all__ = ["generate_span_id", "trace_scope", "traced_span"]
