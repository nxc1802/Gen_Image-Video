"""
tests/adversarial/test_adversarial_p4_challenger2.py
Empirical Challenger P4.2 Adversarial Stress Testing Suite:
1. CircuitBreaker State Machine & Concurrency Stress (thresholds, fast-fail, recovery, half-open trials, fallbacks).
2. RequestTimeoutMiddleware Boundary, Exemption & Cancellation Invariants (504, streaming exemptions, query params).
3. Distributed Trace Propagation Invariants (W3C traceparent, corrupt headers, fallback, DB persistence, broker events).
4. Backup & Restore Scripts Disaster Recovery Invariants (bash syntax, --dry-run, tool absence, tamper detection).
"""
import asyncio
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from factory_contracts.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitOpenError,
    CircuitState,
)
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobSubmissionRequest,
    JobState,
)
from factory.apps.control_api.src.api.middleware.timeout import RequestTimeoutMiddleware
from factory.apps.control_api.src.telemetry.context import (
    clear_correlation_context,
    ctx_trace_id,
    set_correlation_context,
)
from factory.apps.control_api.src.telemetry.logging import TelemetryLogFormatter
from factory.apps.control_api.src.telemetry.middleware import TelemetryTracingMiddleware
from factory.apps.control_api.src.telemetry.propagation import (
    build_traceparent,
    extract_traceparent,
    inject_traceparent,
)
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.streaming.manager import StreamManager

REPO_ROOT = Path(__file__).resolve().parents[2]
INFRA_DIR = REPO_ROOT / "infrastructure"


# ==============================================================================
# SECTION 1: CircuitBreaker Adversarial Stress Tests
# ==============================================================================

class TestCircuitBreakerAdversarial:
    """Stress tests and boundary checks for factory_contracts.resilience.CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_threshold_exact_boundary_closed_to_open(self):
        """Verify state transitions to OPEN on exactly the N-th failure and fast-fails on N+1."""
        failure_threshold = 4
        cb = CircuitBreaker(
            name="boundary-cb",
            failure_threshold=failure_threshold,
            recovery_timeout=0.2,
        )
        invocations = 0

        async def failing():
            nonlocal invocations
            invocations += 1
            raise RuntimeError(f"Failure #{invocations}")

        # Invocations 1 through failure_threshold - 1: circuit remains CLOSED
        for i in range(1, failure_threshold):
            with pytest.raises(RuntimeError):
                await cb.call(failing)
            assert cb.state == CircuitState.CLOSED
            assert cb.get_state() == CircuitState.CLOSED
            assert invocations == i

        # N-th invocation: reaches failure_threshold -> transitions to OPEN
        with pytest.raises(RuntimeError):
            await cb.call(failing)
        assert cb.state == CircuitState.OPEN
        assert cb.get_state() == CircuitState.OPEN
        assert invocations == failure_threshold

        # N+1-th invocation: MUST fast-fail with CircuitOpenError without calling failing()
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(failing)
        assert exc_info.value.code == "SERVICE_UNAVAILABLE"
        assert exc_info.value.status_code == 503
        assert exc_info.value.details["circuit_name"] == "boundary-cb"
        assert exc_info.value.details["retry_after_seconds"] >= 0.0
        # Critical invariant: invocations did NOT increment during fast-fail
        assert invocations == failure_threshold

    @pytest.mark.asyncio
    async def test_fast_fails_repeated_burst_without_invoking_target(self):
        """Burst of 25 calls during OPEN must all fast-fail with 0 target invocations."""
        cb = CircuitBreaker(name="burst-cb", failure_threshold=2, recovery_timeout=0.5)
        calls = 0

        async def buggy():
            nonlocal calls
            calls += 1
            raise ConnectionError("Remote unreachable")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await cb.call(buggy)
        assert calls == 2
        assert cb.state == CircuitState.OPEN

        # Burst of 25 calls
        for _ in range(25):
            with pytest.raises(CircuitOpenError):
                await cb.call(buggy)

        # Invariant: calls count remains strictly 2
        assert calls == 2

    @pytest.mark.asyncio
    async def test_recovery_timeout_and_half_open_trial_success_to_closed(self):
        """Test full recovery lifecycle: OPEN -> wait recovery_timeout -> HALF_OPEN -> successes -> CLOSED."""
        recovery_timeout = 0.15
        max_trials = 3
        cb = CircuitBreaker(
            name="recovery-cb",
            failure_threshold=2,
            recovery_timeout=recovery_timeout,
            half_open_max_trials=max_trials,
        )

        async def fail():
            raise TimeoutError("Timeout")

        # Trip to OPEN
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # Call before recovery_timeout expires -> fast-fail
        with pytest.raises(CircuitOpenError):
            await cb.call(fail)

        # Wait until recovery_timeout expires
        await asyncio.sleep(recovery_timeout + 0.05)

        # get_state() should reflect HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN

        async def succeed():
            return "ok"

        # Execute trial calls
        for i in range(1, max_trials):
            res = await cb.call(succeed)
            assert res == "ok"
            assert cb.state == CircuitState.HALF_OPEN

        # Final trial success reaches half_open_max_trials -> transitions back to CLOSED
        res_final = await cb.call(succeed)
        assert res_final == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_single_trial_failure_reopens_immediately(self):
        """In HALF_OPEN state, even if 2/3 trials passed, 1 failure immediately re-trips to OPEN."""
        recovery_timeout = 0.12
        cb = CircuitBreaker(
            name="reopen-cb",
            failure_threshold=2,
            recovery_timeout=recovery_timeout,
            half_open_max_trials=3,
        )

        async def fail():
            raise IOError("Disk error")

        async def succeed():
            return "recovered"

        # Trip to OPEN
        for _ in range(2):
            with pytest.raises(IOError):
                await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(recovery_timeout + 0.05)

        # Trial 1 & 2 succeed
        await cb.call(succeed)
        await cb.call(succeed)
        assert cb.state == CircuitState.HALF_OPEN

        # Trial 3 fails -> MUST immediately re-open
        with pytest.raises(IOError):
            await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # Immediate next call fast-fails without executing
        with pytest.raises(CircuitOpenError):
            await cb.call(succeed)

    @pytest.mark.asyncio
    async def test_unexpected_exceptions_bypass_circuit_tripping(self):
        """Exceptions NOT in expected_exceptions must bubble through without tripping the circuit."""
        cb = CircuitBreaker(
            name="filter-cb",
            failure_threshold=2,
            expected_exceptions=(ConnectionError, TimeoutError),
        )

        async def raise_value_error():
            raise ValueError("Invalid user input, not downstream service fault")

        # Raise ValueError 5 times
        for _ in range(5):
            with pytest.raises(ValueError):
                await cb.call(raise_value_error)

        # Invariant: Circuit is still CLOSED because ValueError is not an expected failure
        assert cb.state == CircuitState.CLOSED
        assert cb.get_stats()["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_load_stress_trips_cleanly_without_race_conditions(self):
        """Concurrent coroutines tripping circuit: subsequent calls fast-fail immediately."""
        cb = CircuitBreaker(
            name="concurrency-cb",
            failure_threshold=3,
            recovery_timeout=0.3,
        )
        invocations = 0

        async def failing_worker(idx: int):
            nonlocal invocations
            invocations += 1
            raise RuntimeError(f"Worker {idx} failure")

        # Trip circuit sequentially to OPEN
        for i in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(failing_worker, i)
        assert cb.state == CircuitState.OPEN
        assert invocations == 3

        # Now launch 30 concurrent calls against the OPEN circuit
        tasks = [cb.call(failing_worker, i + 10) for i in range(30)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        circuit_open_errors = [r for r in results if isinstance(r, CircuitOpenError)]
        # All 30 must fast-fail with CircuitOpenError
        assert len(circuit_open_errors) == 30
        # Invocations must NOT increase
        assert invocations == 3

    @pytest.mark.asyncio
    async def test_fallback_execution_on_failure_and_on_open(self):
        """Fallback is executed on initial failures AND when fast-failing during OPEN."""
        fallback_invocations = 0

        async def async_fallback(*args, **kwargs):
            nonlocal fallback_invocations
            fallback_invocations += 1
            return {"fallback": True, "arg": args[0], "key": kwargs.get("k")}

        cb = CircuitBreaker(
            name="fallback-cb",
            failure_threshold=1,
            recovery_timeout=1.0,
            fallback=async_fallback,
        )

        async def failing_call(val: str, k: str = ""):
            raise ConnectionRefusedError("Dead")

        # Call 1: trips circuit, catches ConnectionRefusedError, calls fallback
        res1 = await cb.call(failing_call, "arg1", k="key1")
        assert res1 == {"fallback": True, "arg": "arg1", "key": "key1"}
        assert cb.state == CircuitState.OPEN
        assert fallback_invocations == 1

        # Call 2: circuit is OPEN, catches CircuitOpenError, calls fallback without invoking failing_call
        res2 = await cb.call(failing_call, "arg2", k="key2")
        assert res2 == {"fallback": True, "arg": "arg2", "key": "key2"}
        assert fallback_invocations == 2

    @pytest.mark.asyncio
    async def test_fallback_exception_propagation_does_not_corrupt_state(self):
        """If fallback itself raises an error, exception bubbles up and state remains intact."""
        def buggy_fallback(*args, **kwargs):
            raise KeyError("Fallback failed")

        cb = CircuitBreaker(
            name="buggy-fallback-cb",
            failure_threshold=1,
            fallback=buggy_fallback,
        )

        async def target():
            raise ConnectionError("Primary failed")

        with pytest.raises(KeyError):
            await cb.call(target)

        assert cb.state == CircuitState.OPEN

    def test_sync_function_decorator_execution(self):
        """Verify decorator usage on synchronous function executed outside running event loop."""
        cb = CircuitBreaker(name="sync-dec-cb", failure_threshold=2, recovery_timeout=0.2)

        @cb
        def sync_add(a: int, b: int) -> int:
            return a + b

        assert sync_add(10, 20) == 30
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_manual_trip_and_reset_recovers_state(self):
        """Manual trip() forces OPEN, reset() restores CLOSED."""
        cb = CircuitBreaker(name="manual-cb", failure_threshold=10)
        assert cb.state == CircuitState.CLOSED

        cb.trip()
        assert cb.state == CircuitState.OPEN
        assert cb.get_stats()["state"] == "OPEN"

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.get_stats()["failure_count"] == 0


# ==============================================================================
# SECTION 2: RequestTimeoutMiddleware Adversarial Tests
# ==============================================================================

class TestRequestTimeoutMiddlewareAdversarial:
    """Stress tests and boundary checks for RequestTimeoutMiddleware."""

    def _create_app(self, timeout_seconds: float = 0.15, enabled: bool = True) -> FastAPI:
        app = FastAPI()
        app.add_middleware(
            RequestTimeoutMiddleware,
            timeout_seconds=timeout_seconds,
            enabled=enabled,
        )

        @app.get("/api/v1/fast")
        async def fast():
            return {"status": "fast"}

        @app.get("/api/v1/slow")
        async def slow():
            await asyncio.sleep(0.35)
            return {"status": "slow"}

        @app.get("/api/v1/jobs/{job_id}/events")
        async def stream_events(job_id: str):
            await asyncio.sleep(0.25)
            return {"stream": "events", "job_id": job_id}

        @app.get("/api/v1/jobs/{job_id}/tokens")
        async def stream_tokens(job_id: str):
            await asyncio.sleep(0.25)
            return {"stream": "tokens", "job_id": job_id}

        @app.get("/api/v1/jobs/{job_id}/ws")
        async def stream_ws(job_id: str):
            await asyncio.sleep(0.25)
            return {"stream": "ws", "job_id": job_id}

        @app.get("/api/v1/jobs/{job_id}/stream")
        async def stream_generic(job_id: str):
            await asyncio.sleep(0.25)
            return {"stream": "generic", "job_id": job_id}

        @app.get("/docs/custom")
        async def docs_custom():
            await asyncio.sleep(0.25)
            return {"docs": "ok"}

        @app.get("/api/v1/events/nested")
        async def not_exempt_nested():
            # Does NOT end with /events
            await asyncio.sleep(0.25)
            return {"nested": "fail"}

        @app.get("/api/v1/websocket-upgrade-endpoint")
        async def ws_upgraded():
            await asyncio.sleep(0.25)
            return {"upgraded": True}

        return app

    def test_slow_request_triggers_504_gateway_timeout_payload(self):
        """Slow request exceeding timeout returns HTTP 504 with canonical error envelope."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        resp = client.get("/api/v1/slow", headers={"X-Request-ID": "req-timeout-adv-1"})
        assert resp.status_code == 504
        assert resp.headers.get("X-Request-ID") == "req-timeout-adv-1"

        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "GATEWAY_TIMEOUT"
        assert "exceeded timeout limit" in data["error"]["message"]
        assert data["error"]["request_id"] == "req-timeout-adv-1"
        assert data["error"]["details"]["timeout_seconds"] == 0.1

    def test_all_streaming_suffixes_exempt_from_timeout(self):
        """Paths ending with /events, /tokens, /ws, /stream must not be timed out."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        for path in [
            "/api/v1/jobs/j-1/events",
            "/api/v1/jobs/j-1/tokens",
            "/api/v1/jobs/j-1/ws",
            "/api/v1/jobs/j-1/stream",
        ]:
            resp = client.get(path)
            assert resp.status_code == 200, f"Expected 200 for exempt path {path}, got {resp.status_code}"

    def test_query_parameters_do_not_break_suffix_exemption(self):
        """Requests with query strings like ?filter=live&offset=100 must still be recognized as exempt."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        resp = client.get("/api/v1/jobs/j-1/events?filter=live&offset=100&timeout=999")
        assert resp.status_code == 200
        assert resp.json()["stream"] == "events"

    def test_prefix_exemptions_for_docs_and_schemas(self):
        """Paths starting with /docs are exempt even if slow."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        resp = client.get("/docs/custom")
        assert resp.status_code == 200
        assert resp.json()["docs"] == "ok"

    def test_websocket_upgrade_header_exemption(self):
        """Requests with Upgrade: websocket header are exempt even on non-exempt paths."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        resp = client.get(
            "/api/v1/websocket-upgrade-endpoint",
            headers={"Upgrade": "websocket", "Connection": "Upgrade"},
        )
        assert resp.status_code == 200
        assert resp.json()["upgraded"] is True

    def test_nested_path_not_ending_in_suffix_is_timed_out(self):
        """Paths like /api/v1/events/nested do NOT end with /events and must be timed out."""
        app = self._create_app(timeout_seconds=0.1)
        client = TestClient(app)

        resp = client.get("/api/v1/events/nested")
        assert resp.status_code == 504
        assert resp.json()["error"]["code"] == "GATEWAY_TIMEOUT"

    def test_disabled_middleware_allows_slow_requests(self):
        """When enabled=False, slow requests proceed to completion without timeout."""
        app = self._create_app(timeout_seconds=0.05, enabled=False)
        client = TestClient(app)

        resp = client.get("/api/v1/slow")
        assert resp.status_code == 200
        assert resp.json() == {"status": "slow"}


# ==============================================================================
# SECTION 3: Distributed Trace Propagation Adversarial Tests
# ==============================================================================

class TestDistributedTracePropagationAdversarial:
    """Stress tests and boundary checks for W3C traceparent extraction, propagation, and persistence."""

    def test_extract_traceparent_valid_w3c_header(self):
        """Valid W3C header: 00-32hex-16hex-flags."""
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        span_id = "00f067aa0ba902b7"
        tp = f"00-{trace_id}-{span_id}-01"

        extracted_trace, extracted_span = extract_traceparent({"traceparent": tp})
        assert extracted_trace == trace_id
        assert extracted_span == span_id

    @pytest.mark.parametrize(
        "corrupt_header",
        [
            "",                                                              # Empty
            "not-a-traceparent",                                             # Plain string
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",     # Unsupported version != 00
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",      # 31 hex chars (short)
            "00-4bf92f3577b34da6a3ce929d0e0e47366-00f067aa0ba902b7-01",     # 33 hex chars (long)
            "00-4bf92f3577b34da6a3ce929d0e0e473Z-00f067aa0ba902b7-01",     # Non-hex 'Z'
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01",      # 15 hex span (short)
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",       # 1-char flag
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",# Trailing data
        ],
    )
    def test_extract_traceparent_corrupt_headers_fail_gracefully(self, corrupt_header: str):
        """Corrupt headers return (None, None) without throwing unhandled exceptions."""
        extracted_trace, extracted_span = extract_traceparent({"traceparent": corrupt_header})
        assert extracted_trace is None
        assert extracted_span is None

    def test_extract_traceparent_fallback_to_x_trace_id(self):
        """When traceparent is missing or corrupt, valid 32-hex x-trace-id is accepted."""
        custom_trace = "abcdef0123456789abcdef0123456789"
        extracted_trace, extracted_span = extract_traceparent({"x-trace-id": custom_trace})
        assert extracted_trace == custom_trace
        assert extracted_span is not None
        assert len(extracted_span) == 16

    def test_telemetry_tracing_middleware_handles_corrupt_headers_by_minting(self):
        """Incoming corrupt traceparent results in newly minted valid 32-hex trace_id attached to response."""
        app = FastAPI()
        app.add_middleware(TelemetryTracingMiddleware)

        @app.get("/test")
        async def endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test", headers={"traceparent": "00-corrupt-header-01"})
        assert resp.status_code == 200
        assert "traceparent" in resp.headers
        assert "X-Trace-ID" in resp.headers

        minted_trace_id = resp.headers["X-Trace-ID"]
        assert len(minted_trace_id) == 32
        assert all(c in "0123456789abcdef" for c in minted_trace_id.lower())

    @pytest.mark.asyncio
    async def test_end_to_end_trace_id_persistence_in_job_and_events(self):
        """trace_id passed to submit_job persists to DB and attaches to repository events."""
        repo = SqliteJobRepository(":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager(repository=repo)
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

        test_trace_id = "0af7651916cd43dd8448eb211c80319c"
        req = JobSubmissionRequest(
            service="image",
            model="sdxl",
            task="text_to_image",
            inputs={"prompt": "Adversarial trace verification"},
        )

        job, created = await manager.submit_job(
            req,
            request_id="req-trace-test-1",
            trace_id=test_trace_id,
        )

        assert created is True
        assert job.trace_id == test_trace_id

        # Verify DB persistence of FactoryJob.trace_id
        from_db = await repo.get_job(job.job_id)
        assert from_db is not None
        assert from_db.trace_id == test_trace_id

        # Verify event history recorded in repository contains trace_id
        events = await repo.get_events_since(job.job_id, 0)
        assert len(events) >= 1
        initial_event = events[0]
        event_data = initial_event.get("data") or {}
        assert event_data.get("trace_id") == test_trace_id
        assert event_data.get("job_id") == job.job_id

        await repo.close()
        await broker.close()

    def test_structured_log_formatter_correlates_all_trace_fields(self):
        """TelemetryLogFormatter serializes trace_id, span_id, request_id, job_id into JSON."""
        formatter = TelemetryLogFormatter()
        clear_correlation_context()

        set_correlation_context(
            request_id="req-corr-1",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            job_id="job-9999",
            worker_id="worker-node-1",
        )

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=100,
            msg="Executing job worker dispatch",
            args=(),
            exc_info=None,
        )

        formatted_json = formatter.format(record)
        parsed = json.loads(formatted_json)

        assert parsed["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert parsed["span_id"] == "00f067aa0ba902b7"
        assert parsed["request_id"] == "req-corr-1"
        assert parsed["job_id"] == "job-9999"
        assert parsed["worker_id"] == "worker-node-1"
        assert parsed["message"] == "Executing job worker dispatch"

        clear_correlation_context()


# ==============================================================================
# SECTION 4: Disaster Recovery & Backup/Restore Scripts Adversarial Tests
# ==============================================================================

class TestDisasterRecoveryScriptsAdversarial:
    """Stress tests and boundary checks for infrastructure disaster recovery bash scripts."""

    BACKUP_SCRIPTS = [
        INFRA_DIR / "postgres" / "backup.sh",
        INFRA_DIR / "redis" / "backup.sh",
        INFRA_DIR / "s3" / "backup.sh",
    ]

    RESTORE_SCRIPTS = [
        INFRA_DIR / "postgres" / "restore.sh",
        INFRA_DIR / "redis" / "restore.sh",
        INFRA_DIR / "s3" / "restore.sh",
    ]

    def test_bash_syntax_check_all_scripts(self):
        """bash -n verification for all 6 scripts."""
        for script in self.BACKUP_SCRIPTS + self.RESTORE_SCRIPTS:
            res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            assert res.returncode == 0, f"Syntax check failed for {script}: {res.stderr}"

    @pytest.mark.parametrize("script", BACKUP_SCRIPTS)
    def test_backup_scripts_reject_unknown_command_line_options(self, script: Path):
        """Passing unknown flags to backup scripts must exit with code 1."""
        res = subprocess.run([str(script), "--unknown-flag-adversarial"], capture_output=True, text=True)
        assert res.returncode == 1
        assert "Unknown option" in res.stderr

    @pytest.mark.parametrize("script", RESTORE_SCRIPTS)
    def test_restore_scripts_require_argument(self, script: Path):
        """Running restore scripts with no argument must exit with code 1."""
        res = subprocess.run([str(script)], capture_output=True, text=True)
        assert res.returncode == 1
        assert "ERROR:" in res.stderr

    @pytest.mark.parametrize("script", RESTORE_SCRIPTS)
    def test_restore_scripts_reject_nonexistent_file_or_dir(self, script: Path):
        """Targeting non-existent file or directory must exit with code 2."""
        res = subprocess.run([str(script), "/nonexistent/fake/path"], capture_output=True, text=True)
        assert res.returncode == 2
        assert "ERROR:" in res.stderr
        assert "does not exist" in res.stderr

    @pytest.mark.parametrize("script", RESTORE_SCRIPTS)
    def test_restore_scripts_reject_multiple_positional_arguments(self, script: Path, tmp_path: Path):
        """Passing multiple positional arguments must exit with code 1."""
        dummy = tmp_path / "dummy.dump"
        dummy.write_text("dummy")
        res = subprocess.run([str(script), str(dummy), "unexpected_extra_arg"], capture_output=True, text=True)
        assert res.returncode == 1
        assert "Unexpected argument" in res.stderr

    def test_checksum_mismatch_aborts_restore_immediately(self, tmp_path: Path):
        """Tampered backup file must trigger checksum failure and exit non-zero (1)."""
        pg_restore = INFRA_DIR / "postgres" / "restore.sh"
        redis_restore = INFRA_DIR / "redis" / "restore.sh"

        for script, ext in [(pg_restore, ".dump"), (redis_restore, ".rdb")]:
            backup_file = tmp_path / f"test_tampered{ext}"
            backup_file.write_text("original authentic content")

            # Generate real checksum
            sha_file = tmp_path / f"test_tampered{ext}.sha256"
            shasum_proc = subprocess.run(
                ["shasum", "-a", "256", str(backup_file)],
                capture_output=True,
                text=True,
            )
            sha_file.write_text(shasum_proc.stdout)

            # Tamper the backup file
            backup_file.write_text("corrupted tampered malicious payload")

            # Run restore script (dry-run still verifies checksum)
            res = subprocess.run([str(script), "--dry-run", str(backup_file)], capture_output=True, text=True)
            assert res.returncode != 0, f"Expected checksum failure for {script}"
            assert "FAILED" in res.stderr or "FAILED" in res.stdout or res.returncode != 0

    def test_postgres_backup_fails_if_pg_dump_missing(self, tmp_path: Path):
        """When pg_dump is not in PATH, backup script must exit with code 2."""
        backup_script = INFRA_DIR / "postgres" / "backup.sh"
        res = subprocess.run(
            [str(backup_script)],
            env={"PATH": "/usr/bin:/bin", "BACKUP_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "pg_dump utility not found in PATH" in res.stderr

    def test_redis_backup_fails_if_redis_cli_missing(self, tmp_path: Path):
        """When redis-cli is not in PATH, backup script must exit with code 2."""
        backup_script = INFRA_DIR / "redis" / "backup.sh"
        res = subprocess.run(
            [str(backup_script)],
            env={"PATH": "/usr/bin:/bin", "BACKUP_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "redis-cli utility not found in PATH" in res.stderr

    def test_s3_backup_fails_if_all_s3_tools_missing(self, tmp_path: Path):
        """When aws, mc, and boto3 are missing, s3 backup must exit with code 2."""
        backup_script = INFRA_DIR / "s3" / "backup.sh"
        res = subprocess.run(
            [str(backup_script)],
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/nonexistent", "BACKUP_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "Neither aws, mc, nor boto3 is available" in res.stderr

    def test_dry_run_creates_no_files_or_directories(self, tmp_path: Path):
        """--dry-run must be strictly non-mutating and create no artifacts."""
        dry_dir = tmp_path / "dry_run_test_dir"
        for script in [INFRA_DIR / "postgres" / "backup.sh", INFRA_DIR / "redis" / "backup.sh", INFRA_DIR / "s3" / "backup.sh"]:
            res = subprocess.run(
                [str(script), "--dry-run"],
                env={"PATH": os.environ["PATH"], "BACKUP_DIR": str(dry_dir)},
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0
            assert "[DRY-RUN]" in res.stdout
            assert not dry_dir.exists(), f"{script} created directory during --dry-run"
