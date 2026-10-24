"""
factory/apps/control-api/src/telemetry/tracer.py
OpenTelemetry TracerProvider initialization and fallback tracer.
"""
import os
import uuid
from typing import Any, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None


def generate_trace_id() -> str:
    """Generates a valid W3C 128-bit trace ID (32 lowercase hex characters)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generates a valid W3C 64-bit span ID (16 lowercase hex characters)."""
    return uuid.uuid4().hex[:16]


_tracer = None


def init_tracer(service_name: str = "factory.control-api") -> Any:
    global _tracer
    if OPENTELEMETRY_AVAILABLE:
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            resource = Resource.create({"service.name": service_name, "service.version": "4.0.0"})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name, "4.0.0")
        return _tracer
    return None


def get_tracer():
    global _tracer
    if _tracer is None:
        init_tracer()
    return _tracer
