"""
factory/apps/control-api/src/telemetry/logging.py
Structured JSON log formatter binding 5-hop correlation and OpenTelemetry trace IDs.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from factory.apps.control_api.src.telemetry.context import get_correlation_dict


class TelemetryLogFormatter(logging.Formatter):
    """
    Emits structured JSON logs containing:
    - trace_id and span_id (OpenTelemetry W3C)
    - request_id, job_id, worker_id, model_id, artifact_id (5-hop correlation)
    - level, logger, message, timestamp
    - sanitized exception details for internal logs
    """

    def format(self, record: logging.LogRecord) -> str:
        corr = get_correlation_dict()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": corr.get("trace_id"),
            "span_id": corr.get("span_id"),
            "request_id": corr.get("request_id"),
            "job_id": corr.get("job_id"),
            "worker_id": corr.get("worker_id"),
            "model_id": corr.get("model_id"),
            "artifact_id": corr.get("artifact_id"),
        }

        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        return json.dumps(log_entry, default=str)


def configure_telemetry_logging(level: int = logging.INFO) -> None:
    """Configures root logger to use TelemetryLogFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(TelemetryLogFormatter())

    # Replace existing handlers
    root_logger.handlers = [handler]
