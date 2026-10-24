"""
factory/apps/control-api/src/telemetry/decorators.py
Tracing decorators and async context manager for OpenTelemetry spans.
"""
import functools
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from factory.apps.control_api.src.telemetry.context import (
    ctx_span_id,
    get_correlation_dict,
    set_correlation_context,
)
from factory.apps.control_api.src.telemetry.tracer import generate_span_id, get_tracer

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


@asynccontextmanager
async def trace_scope(name: str, **attributes: Any) -> AsyncGenerator[Any, None]:
    """Async context manager creating a traced span with automatic 5-hop correlation binding."""
    tracer = get_tracer()
    new_span_id = generate_span_id()
    token = ctx_span_id.set(new_span_id)

    if tracer:
        with tracer.start_as_current_span(name) as span:
            corr = get_correlation_dict()
            for k, v in corr.items():
                if v:
                    span.set_attribute(f"factory.{k}", v)
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
            try:
                yield span
            except Exception as exc:
                if hasattr(span, "record_exception"):
                    span.record_exception(exc)
                if Status and StatusCode and hasattr(span, "set_status"):
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                ctx_span_id.reset(token)
    else:
        try:
            yield None
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
