"""
factory_contracts/errors/error_contract.py
Standardized Error Response Envelope, Error Codes, and Core Exceptions
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(str, Enum):
    # 401 Unauthorized
    UNAUTHORIZED = "UNAUTHORIZED"
    # 403 Forbidden
    FORBIDDEN = "FORBIDDEN"
    # 400 Bad Request
    INVALID_REQUEST = "INVALID_REQUEST"
    BAD_REQUEST = "BAD_REQUEST"
    CAPABILITY_NOT_SUPPORTED = "CAPABILITY_NOT_SUPPORTED"
    SSRF_BLOCKED = "SSRF_BLOCKED"
    INVALID_MEDIA_TYPE = "INVALID_MEDIA_TYPE"
    # 404 Not Found
    NOT_FOUND = "NOT_FOUND"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    # 405 Method Not Allowed
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    # 409 Conflict
    JOB_CANCELLED = "JOB_CANCELLED"
    CONFLICT = "CONFLICT"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    # 413 Payload Too Large
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    # 415 Unsupported Media Type
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    # 422 Unprocessable Entity
    VALIDATION_ERROR = "VALIDATION_ERROR"
    # 429 Too Many Requests
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    QUEUE_FULL = "QUEUE_FULL"
    # 502 Bad Gateway
    BAD_GATEWAY = "BAD_GATEWAY"
    # 503 Service Unavailable
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    # 504 Gateway Timeout
    TIMEOUT = "TIMEOUT"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    WORKER_DISCONNECTED = "WORKER_DISCONNECTED"
    # 500 Internal Error
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    # Unified §4.1 Error Taxonomy Codes
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    WEIGHTS_MISSING = "WEIGHTS_MISSING"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INSUFFICIENT_VRAM = "INSUFFICIENT_VRAM"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    BACKEND_CRASHED = "BACKEND_CRASHED"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    LICENSE_RESTRICTED = "LICENSE_RESTRICTED"


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode = Field(..., description="Machine-readable error enumeration code")
    message: str = Field(..., description="Human-readable sanitized error description (no stack traces)")
    request_id: str = Field(..., description="Distributed tracing request UUID correlating with server logs")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO8601 UTC timestamp of error generation"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured, sanitized contextual details"
    )


class UnifiedErrorResponse(BaseModel):
    """Top-level JSON envelope returned by all HTTP 4xx and 5xx responses."""
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(..., description="Standardized error body")


class FactoryError(Exception):
    """Base exception for all Factory v4 application and domain errors."""
    def __init__(
        self,
        message: str,
        code: str = ErrorCode.INTERNAL_SERVER_ERROR.value,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class UnauthorizedError(FactoryError):
    def __init__(
        self,
        message: str = "Missing or invalid authorization credentials",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 401,
    ):
        super().__init__(message=message, code=ErrorCode.UNAUTHORIZED.value, status_code=status_code, details=details)


class ForbiddenError(FactoryError):
    def __init__(self, message: str = "Access forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.FORBIDDEN.value, status_code=403, details=details)


class InvalidRequestError(FactoryError):
    def __init__(self, message: str = "Invalid request payload or parameters", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.INVALID_REQUEST.value, status_code=400, details=details)


class SSRFBlockedError(FactoryError):
    def __init__(self, message: str = "Access to requested URL host or IP is blocked for security", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.SSRF_BLOCKED.value, status_code=400, details=details)


class InvalidMediaError(FactoryError):
    def __init__(self, message: str = "Unsupported or invalid media format", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.INVALID_MEDIA_TYPE.value, status_code=400, details=details)


class NotFoundError(FactoryError):
    def __init__(self, message: str = "Requested resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.NOT_FOUND.value, status_code=404, details=details)


class ArtifactNotFoundError(FactoryError):
    def __init__(self, message: str = "Artifact file not found on scratch disk.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.ARTIFACT_NOT_FOUND.value, status_code=404, details=details)


class ConflictError(FactoryError):
    def __init__(self, message: str = "Resource state conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.CONFLICT.value, status_code=409, details=details)


class PayloadTooLargeError(FactoryError):
    def __init__(self, message: str = "Payload exceeds maximum allowed size (50MB)", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.PAYLOAD_TOO_LARGE.value, status_code=413, details=details)


class RateLimitExceededError(FactoryError):
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.RATE_LIMIT_EXCEEDED.value, status_code=429, details=details)


class ModelUnavailableError(FactoryError):
    def __init__(
        self,
        message: str = "Requested model is currently unavailable",
        details: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code=code or ErrorCode.MODEL_UNAVAILABLE.value,
            status_code=503,
            details=details,
        )


class ServiceUnavailableError(FactoryError):
    def __init__(self, message: str = "Requested service is temporarily unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.SERVICE_UNAVAILABLE.value, status_code=503, details=details)


class GatewayTimeoutError(FactoryError):
    def __init__(self, message: str = "Upstream generation request timed out", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.GATEWAY_TIMEOUT.value, status_code=504, details=details)


class InternalServerError(FactoryError):
    def __init__(self, message: str = "An unexpected internal server error occurred", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.INTERNAL_SERVER_ERROR.value, status_code=500, details=details)


class ModelNotLoadedError(FactoryError):
    def __init__(self, message: str = "Requested model is not loaded in GPU runtime", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.MODEL_NOT_LOADED.value, status_code=503, details=details)


class InsufficientVRAMError(FactoryError):
    def __init__(self, message: str = "Insufficient GPU VRAM available to load or execute model", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.INSUFFICIENT_VRAM.value, status_code=503, details=details)


class BackendCrashedError(FactoryError):
    def __init__(self, message: str = "Backend process crashed during runtime execution", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.BACKEND_CRASHED.value, status_code=500, details=details)


class UnsupportedCapabilityError(InvalidRequestError):
    def __init__(self, message: str = "Requested task or variant capability is not supported by model", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.code = ErrorCode.UNSUPPORTED_CAPABILITY.value
        self.status_code = 400



class WeightsMissingError(FactoryError):
    def __init__(self, message: str = "Required checkpoint weights or model files are missing", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.WEIGHTS_MISSING.value, status_code=503, details=details)


def classify_worker_error(
    exc: Exception,
    subsystem: str = "backend",
    stage: str = "execution",
) -> tuple[ErrorCode, str, bool]:
    """
    Classifies worker runtime exceptions into the actionable §4.1 ErrorCode taxonomy
    with stage and retry semantics.
    Returns (error_code, sanitized_message, retryable).
    Guarantees sanitized output without leaking stack traces or local paths.
    """
    err_str = str(exc).lower()

    # 1. Unsupported capability (must precede InvalidRequestError since UnsupportedCapabilityError subclasses it)
    if (
        isinstance(exc, UnsupportedCapabilityError)
        or "capability" in err_str
        or "not supported" in err_str
        or "task unsupported" in err_str
        or "variant unsupported" in err_str
        or ("unsupported" in err_str and "unsupported parameter" not in err_str)
    ):
        return (ErrorCode.UNSUPPORTED_CAPABILITY, f"Task or capability unsupported by {subsystem}.", False)

    # 2. Invalid input
    if isinstance(exc, (ValueError, InvalidRequestError, InvalidMediaError)) or "invalid" in err_str or "unsupported parameter" in err_str:
        return (ErrorCode.INVALID_INPUT, f"Invalid input parameters for {subsystem} generation.", False)

    # 3. Timeouts
    if isinstance(exc, (TimeoutError, GatewayTimeoutError)) or "timeout" in err_str or "timed out" in err_str:
        return (ErrorCode.INFERENCE_TIMEOUT, f"{subsystem.capitalize()} inference timed out during stage '{stage}'.", True)

    # 4. Out of Memory (OOM) / VRAM
    if "out of memory" in err_str or "cuda oom" in err_str or "insufficient vram" in err_str or "oom" in err_str:
        return (ErrorCode.INSUFFICIENT_VRAM, f"Insufficient GPU VRAM to complete {subsystem} job.", False)

    # 5. Missing Weights / Checkpoint
    if "weights" in err_str or "checkpoint" in err_str or ("not found" in err_str and ("gguf" in err_str or "safetensors" in err_str or "pth" in err_str)):
        return (ErrorCode.WEIGHTS_MISSING, f"Required model weights or checkpoints missing for {subsystem}.", False)

    # 6. Artifact missing / empty
    if "artifact" in err_str or "output metadata" in err_str or "empty output" in err_str or "0 bytes" in err_str:
        return (ErrorCode.ARTIFACT_MISSING, f"Expected {subsystem} output artifact was missing or empty.", False)

    # 7. Backend crashed
    if "crash" in err_str or "sigkill" in err_str or "terminated" in err_str or "died" in err_str:
        return (ErrorCode.BACKEND_CRASHED, f"{subsystem.capitalize()} backend process crashed during {stage}.", True)

    # 8. Backend / service unavailable
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ModelUnavailableError, ServiceUnavailableError)) or "connection refused" in err_str or "unavailable" in err_str:
        return (ErrorCode.BACKEND_UNAVAILABLE, f"{subsystem.capitalize()} backend is unavailable or unreachable.", True)

    # Default
    return (ErrorCode.BACKEND_UNAVAILABLE, f"{subsystem.capitalize()} backend encountered an error. Inspect correlated worker logs.", True)

