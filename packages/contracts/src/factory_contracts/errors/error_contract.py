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
    # 500 Internal Error
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


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
    def __init__(self, message: str = "Missing or invalid authorization credentials", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.UNAUTHORIZED.value, status_code=401, details=details)


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
    def __init__(self, message: str = "Requested model is currently unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.MODEL_UNAVAILABLE.value, status_code=503, details=details)


class ServiceUnavailableError(FactoryError):
    def __init__(self, message: str = "Requested service is temporarily unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.SERVICE_UNAVAILABLE.value, status_code=503, details=details)


class GatewayTimeoutError(FactoryError):
    def __init__(self, message: str = "Upstream generation request timed out", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.GATEWAY_TIMEOUT.value, status_code=504, details=details)


class InternalServerError(FactoryError):
    def __init__(self, message: str = "An unexpected internal server error occurred", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=ErrorCode.INTERNAL_SERVER_ERROR.value, status_code=500, details=details)
