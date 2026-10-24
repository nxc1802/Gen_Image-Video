"""
factory/apps/control-api/src/telemetry/propagation.py
W3C TraceContext (traceparent & tracestate) injection and extraction.
"""
import re
from typing import Any, Dict, Optional, Tuple

from factory.apps.control_api.src.telemetry.tracer import generate_span_id, generate_trace_id

# W3C traceparent regex: 00-32hex-16hex-01
TRACEPARENT_REGEX = re.compile(r"^00-([0-9a-fA-F]{32})-([0-9a-fA-F]{16})-([0-9a-fA-F]{2})$")


def build_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """Builds a W3C traceparent header string."""
    flags = "01" if sampled else "00"
    return f"00-{trace_id.lower()}-{span_id.lower()}-{flags}"


def extract_traceparent(headers: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts (trace_id, span_id) from request headers.
    Checks 'traceparent' (case-insensitive) or 'x-trace-id'.
    """
    traceparent = None
    for k, v in headers.items():
        if k.lower() == "traceparent":
            traceparent = str(v).strip()
            break

    if traceparent:
        match = TRACEPARENT_REGEX.match(traceparent)
        if match:
            return match.group(1).lower(), match.group(2).lower()

    # Fallback to X-Trace-ID
    for k, v in headers.items():
        if k.lower() == "x-trace-id":
            val = str(v).strip().lower()
            if len(val) == 32:
                return val, generate_span_id()

    return None, None


def inject_traceparent(
    carrier: Dict[str, Any],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    sampled: bool = True,
) -> str:
    """Injects W3C traceparent and X-Trace-ID into carrier dictionary (headers or metadata)."""
    t_id = trace_id or generate_trace_id()
    s_id = span_id or generate_span_id()
    tp = build_traceparent(t_id, s_id, sampled=sampled)
    carrier["traceparent"] = tp
    carrier["x-trace-id"] = t_id
    return tp
