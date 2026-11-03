"""
factory/apps/control-api/src/telemetry/logging.py
Structured JSON logging using standard library logging for Factory v4 Lite.
"""
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict

from factory.apps.control_api.src.telemetry.context import get_correlation_dict


class LiteJsonLogFormatter(logging.Formatter):
    """
    Lightweight JSON log formatter using standard library.
    Emits structured single-line JSON with correlation identifiers
    (request_id, job_id, worker_id, trace_id, span_id, model_id, artifact_id).
    """

    def format(self, record: logging.LogRecord) -> str:
        corr = get_correlation_dict()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": corr.get("request_id"),
            "job_id": corr.get("job_id"),
            "worker_id": corr.get("worker_id"),
            "trace_id": corr.get("trace_id"),
            "span_id": corr.get("span_id"),
            "model_id": corr.get("model_id"),
            "artifact_id": corr.get("artifact_id"),
        }

        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        return json.dumps(log_entry, default=str)


# Alias for backward compatibility
TelemetryLogFormatter = LiteJsonLogFormatter


def configure_lite_logging(level: int = logging.INFO) -> None:
    """Configures root logger with LiteJsonLogFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(LiteJsonLogFormatter())

    root_logger.handlers = [handler]


# Alias for backward compatibility
configure_telemetry_logging = configure_lite_logging

__all__ = [
    "LiteJsonLogFormatter",
    "TelemetryLogFormatter",
    "configure_lite_logging",
    "configure_telemetry_logging",
]
