"""
factory_contracts/resilience.py
Production-grade Circuit Breaker pattern supporting CLOSED, OPEN, and HALF_OPEN states.
"""
import asyncio
import functools
import inspect
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

from factory_contracts.errors.error_contract import FactoryError

logger = logging.getLogger("factory.contracts.resilience")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerError(FactoryError):
    """Base exception for circuit breaker faults."""
    def __init__(
        self,
        message: str = "Circuit breaker tripped",
        code: str = "SERVICE_UNAVAILABLE",
        status_code: int = 503,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, code=code, status_code=status_code, details=details)


class CircuitOpenError(CircuitBreakerError):
    """Raised when call is rejected because the circuit is in OPEN state."""
    def __init__(
        self,
        name: str = "default",
        time_until_retry: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        det = details or {}
        det.update({"circuit_name": name, "retry_after_seconds": max(0.0, round(time_until_retry, 2))})
        super().__init__(
            message=f"Circuit breaker '{name}' is OPEN. Requests are fast-failing to protect downstream service.",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=det,
        )


class CircuitBreaker:
    """
    Thread-safe and async-safe Circuit Breaker implementation.
    
    State Machine:
    - CLOSED: Normal operation. Successes reset failure count. Reaching failure_threshold moves to OPEN.
    - OPEN: Fast-fail calls immediately. After recovery_timeout, moves to HALF_OPEN.
    - HALF_OPEN: Allows limited probes. Any failure moves to OPEN. Consecutive successes move to CLOSED.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_trials: int = 3,
        half_open_concurrency: Optional[int] = None,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        fallback: Optional[Callable[..., Any]] = None,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)
        self.half_open_max_trials = max(1, half_open_max_trials)
        self.half_open_concurrency = max(
            1,
            half_open_concurrency if half_open_concurrency is not None else self.half_open_max_trials,
        )
        self.expected_exceptions = expected_exceptions
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.monotonic()
        self._lock = asyncio.Lock()
        self._trial_semaphore = asyncio.Semaphore(self.half_open_concurrency)

    @property
    def state(self) -> CircuitState:
        return self._state

    def get_state(self) -> CircuitState:
        now = time.monotonic()
        if self._state == CircuitState.OPEN:
            if now - self._last_state_change >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.get_state().value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_max_trials": self.half_open_max_trials,
            "half_open_concurrency": self.half_open_concurrency,
        }

    def reset(self) -> None:
        """Manually resets the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.monotonic()
        self._trial_semaphore = asyncio.Semaphore(self.half_open_concurrency)
        logger.info(f"CircuitBreaker '{self.name}' manually reset to CLOSED.")

    def trip(self) -> None:
        """Manually forces the circuit breaker into OPEN state."""
        self._state = CircuitState.OPEN
        self._failure_count = self.failure_threshold
        self._last_state_change = time.monotonic()
        logger.warning(f"CircuitBreaker '{self.name}' manually tripped to OPEN.")

    async def _before_call(self) -> bool:
        """Validates circuit state before execution. Returns True if circuit is in HALF_OPEN."""
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                elapsed = now - self._last_state_change
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._last_state_change = now
                    logger.info(f"CircuitBreaker '{self.name}' transitioned from OPEN to HALF_OPEN.")
                else:
                    time_until_retry = self.recovery_timeout - elapsed
                    raise CircuitOpenError(name=self.name, time_until_retry=time_until_retry)
            return self._state == CircuitState.HALF_OPEN

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_trials:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_state_change = time.monotonic()
                    logger.info(
                        f"CircuitBreaker '{self.name}' recovered: transitioned from HALF_OPEN to CLOSED "
                        f"after {self.half_open_max_trials} successful trial calls."
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._failure_count = self.failure_threshold
                self._success_count = 0
                self._last_state_change = now
                logger.warning(
                    f"CircuitBreaker '{self.name}' trial failed during HALF_OPEN ({exc}). Re-opening circuit."
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.error(
                        f"CircuitBreaker '{self.name}' tripped: {self._failure_count} consecutive failures. "
                        f"Transitioned to OPEN for {self.recovery_timeout}s."
                    )

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Executes an async or sync callable through the circuit breaker."""
        try:
            is_half_open = await self._before_call()
        except CircuitOpenError:
            if self.fallback:
                if inspect.iscoroutinefunction(self.fallback):
                    return await self.fallback(*args, **kwargs)
                return self.fallback(*args, **kwargs)
            raise

        if is_half_open:
            # Bound concurrent trial executions in HALF_OPEN state
            await self._trial_semaphore.acquire()
            try:
                # Re-verify state under lock in case another trial probe failed
                async with self._lock:
                    if self._state == CircuitState.OPEN:
                        elapsed = time.monotonic() - self._last_state_change
                        time_until_retry = max(0.0, self.recovery_timeout - elapsed)
                        raise CircuitOpenError(name=self.name, time_until_retry=time_until_retry)

                try:
                    if inspect.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    await self._on_success()
                    return result
                except self.expected_exceptions as exc:
                    await self._on_failure(exc)
                    if self.fallback:
                        if inspect.iscoroutinefunction(self.fallback):
                            return await self.fallback(*args, **kwargs)
                        return self.fallback(*args, **kwargs)
                    raise
            except CircuitOpenError:
                if self.fallback:
                    if inspect.iscoroutinefunction(self.fallback):
                        return await self.fallback(*args, **kwargs)
                    return self.fallback(*args, **kwargs)
                raise
            finally:
                self._trial_semaphore.release()
        else:
            # CLOSED state: normal unconstrained execution without semaphore overhead
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                await self._on_success()
                return result
            except self.expected_exceptions as exc:
                await self._on_failure(exc)
                if self.fallback:
                    if inspect.iscoroutinefunction(self.fallback):
                        return await self.fallback(*args, **kwargs)
                    return self.fallback(*args, **kwargs)
                raise

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator wrapper for functions."""
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(self.call(func, *args, **kwargs))
            return sync_wrapper
