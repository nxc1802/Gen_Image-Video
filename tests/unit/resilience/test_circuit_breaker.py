"""
tests/unit/resilience/test_circuit_breaker.py
Unit tests for CircuitBreaker pattern in factory_contracts.resilience.
Verifies state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED), fast-fail, recovery, and fallback.
"""
import asyncio
import time
import pytest

from factory_contracts.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    CircuitOpenError,
)


@pytest.mark.asyncio
async def test_initial_state_is_closed():
    cb = CircuitBreaker(name="test-cb", failure_threshold=3, recovery_timeout=0.2)
    assert cb.state == CircuitState.CLOSED
    assert cb.get_state() == CircuitState.CLOSED

    async def successful_call():
        return "success"

    res = await cb.call(successful_call)
    assert res == "success"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_trips_to_open_on_consecutive_failures():
    cb = CircuitBreaker(name="test-cb", failure_threshold=3, recovery_timeout=0.2)

    invocations = 0

    async def failing_call():
        nonlocal invocations
        invocations += 1
        raise ValueError("Downstream service failure")

    # 1st failure
    with pytest.raises(ValueError):
        await cb.call(failing_call)
    assert cb.state == CircuitState.CLOSED

    # 2nd failure
    with pytest.raises(ValueError):
        await cb.call(failing_call)
    assert cb.state == CircuitState.CLOSED

    # 3rd failure reaches threshold -> state becomes OPEN
    with pytest.raises(ValueError):
        await cb.call(failing_call)
    assert cb.state == CircuitState.OPEN
    assert invocations == 3

    # Subsequent call fast-fails immediately without calling target function
    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call(failing_call)
    assert exc_info.value.code == "SERVICE_UNAVAILABLE"
    assert exc_info.value.status_code == 503
    # Ensure failing_call was NOT invoked during fast-fail
    assert invocations == 3


@pytest.mark.asyncio
async def test_circuit_recovery_half_open_to_closed():
    # Use short recovery timeout (0.15s)
    cb = CircuitBreaker(name="test-recover", failure_threshold=2, recovery_timeout=0.15, half_open_max_trials=2)

    async def failing_call():
        raise RuntimeError("Service down")

    # Trip the circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_call)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout to elapse
    await asyncio.sleep(0.18)

    # Next call transitions to HALF_OPEN
    async def recovered_call():
        return "recovered"

    # 1st success in HALF_OPEN
    res1 = await cb.call(recovered_call)
    assert res1 == "recovered"
    assert cb.state == CircuitState.HALF_OPEN

    # 2nd success in HALF_OPEN reaches half_open_max_trials -> transitions back to CLOSED
    res2 = await cb.call(recovered_call)
    assert res2 == "recovered"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_re_opens_circuit():
    cb = CircuitBreaker(name="test-reopen", failure_threshold=2, recovery_timeout=0.1)

    async def failing_call():
        raise RuntimeError("Error")

    # Trip to OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_call)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.12)

    # Call fails during HALF_OPEN -> should immediately re-open
    with pytest.raises(RuntimeError):
        await cb.call(failing_call)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_fallback_execution():
    async def fallback_handler():
        return {"status": "degraded_cached_response"}

    cb = CircuitBreaker(
        name="test-fallback",
        failure_threshold=1,
        recovery_timeout=1.0,
        fallback=fallback_handler,
    )

    async def broken_call():
        raise ConnectionError("Connection refused")

    # 1st failure executes broken_call, catches error, trips circuit, returns fallback
    res1 = await cb.call(broken_call)
    assert res1 == {"status": "degraded_cached_response"}
    assert cb.state == CircuitState.OPEN

    # 2nd call fast-fails and executes fallback directly
    res2 = await cb.call(broken_call)
    assert res2 == {"status": "degraded_cached_response"}


@pytest.mark.asyncio
async def test_manual_trip_and_reset():
    cb = CircuitBreaker(name="test-manual")
    assert cb.state == CircuitState.CLOSED

    cb.trip()
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_bounded_concurrency_on_success():
    """Verify that concurrent calls in HALF_OPEN are strictly bounded by half_open_concurrency."""
    cb = CircuitBreaker(
        name="test-bound-success",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_max_trials=2,
        half_open_concurrency=2,
    )
    # Trip circuit to OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.12)
    assert cb.get_state() == CircuitState.HALF_OPEN

    in_flight = 0
    max_in_flight = 0

    async def slow_probe(idx: int):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.04)
        in_flight -= 1
        return f"result-{idx}"

    results = await asyncio.gather(*(cb.call(slow_probe, i) for i in range(10)))
    assert len(results) == 10
    assert max_in_flight <= 2
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_bounded_concurrency_on_failure_shields_downstream():
    """Verify that a failure in HALF_OPEN re-trips circuit and excess queued calls fast-fail."""
    cb = CircuitBreaker(
        name="test-bound-fail",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_max_trials=2,
        half_open_concurrency=2,
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.12)

    invocations = 0

    async def failing_probe(idx: int):
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0.04)
        if idx == 0:
            raise ConnectionError("downstream fault")
        return f"ok-{idx}"

    results = await asyncio.gather(*(cb.call(failing_probe, i) for i in range(10)), return_exceptions=True)
    open_errors = [r for r in results if isinstance(r, CircuitOpenError)]
    conn_errors = [r for r in results if isinstance(r, ConnectionError)]
    assert len(conn_errors) == 1
    assert len(open_errors) == 8
    assert invocations <= 2  # Downstream was shielded from 8 of the 10 calls
    assert cb.state == CircuitState.OPEN

