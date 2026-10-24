"""
factory/apps/control-api/src/telemetry/context.py
ContextVar task-local distributed correlation context across async coroutines.
5-Hop Correlation: request_id -> job_id -> worker_id -> model_id -> artifact_id
"""
import contextvars
from typing import Any, Dict, Optional

ctx_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
ctx_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("job_id", default=None)
ctx_worker_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("worker_id", default=None)
ctx_model_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("model_id", default=None)
ctx_artifact_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("artifact_id", default=None)
ctx_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
ctx_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("span_id", default=None)


def set_correlation_context(
    request_id: Optional[str] = None,
    job_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    model_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    if request_id is not None:
        ctx_request_id.set(request_id)
    if job_id is not None:
        ctx_job_id.set(job_id)
    if worker_id is not None:
        ctx_worker_id.set(worker_id)
    if model_id is not None:
        ctx_model_id.set(model_id)
    if artifact_id is not None:
        ctx_artifact_id.set(artifact_id)
    if trace_id is not None:
        ctx_trace_id.set(trace_id)
    if span_id is not None:
        ctx_span_id.set(span_id)


def get_correlation_dict() -> Dict[str, Optional[str]]:
    return {
        "request_id": ctx_request_id.get(),
        "job_id": ctx_job_id.get(),
        "worker_id": ctx_worker_id.get(),
        "model_id": ctx_model_id.get(),
        "artifact_id": ctx_artifact_id.get(),
        "trace_id": ctx_trace_id.get(),
        "span_id": ctx_span_id.get(),
    }


def clear_correlation_context() -> None:
    ctx_request_id.set(None)
    ctx_job_id.set(None)
    ctx_worker_id.set(None)
    ctx_model_id.set(None)
    ctx_artifact_id.set(None)
    ctx_trace_id.set(None)
    ctx_span_id.set(None)
