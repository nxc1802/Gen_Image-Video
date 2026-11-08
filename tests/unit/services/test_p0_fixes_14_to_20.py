"""
tests/unit/services/test_p0_fixes_14_to_20.py
Regression tests for Phase 0 Bug Fixes FIX-14 through FIX-20.
"""
import json
import os
from pathlib import Path
import pytest
import httpx

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    classify_worker_error,
)
from services.enhancement.server import app as enhance_app


# ---------------------------------------------------------------------------
# FIX-14: Enhancement health reflects model availability
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fix_14_enhancement_health_reflects_backend_availability(monkeypatch):
    """FIX-14: /health never signals infer_ready: True without configured backends."""
    # Ensure env vars are empty
    for k in [
        "IMAGE_SR_BACKEND_URL",
        "VIDEO_SR_BACKEND_URL",
        "RIFE_BACKEND_URL",
        "INTERPOLATION_BACKEND_URL",
        "AUDIO_ENHANCE_BACKEND_URL",
    ]:
        monkeypatch.delenv(k, raising=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=enhance_app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["subsystem"] == "enhancement"
        assert data["infer_ready"] is False
        assert data["status"] == "healthy"
        assert data["backends"]["image_sr"] is False

    # Configure only one backend -> degraded
    monkeypatch.setenv("IMAGE_SR_BACKEND_URL", "http://127.0.0.1:8100")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=enhance_app), base_url="http://test") as client:
        res = await client.get("/health")
        data = res.json()
        assert data["infer_ready"] is True
        assert data["backends"]["image_sr"] is True
        assert data["backends"]["video_sr"] is False


# ---------------------------------------------------------------------------
# FIX-15: E2E suite split and labeling
# ---------------------------------------------------------------------------
def test_fix_15_e2e_suite_split_and_labeling(monkeypatch):
    """FIX-15: Default MockTransport tests never count as live inference."""
    from tests.e2e.conftest import contract_mock_client, live_client
    from tests.e2e.mock_contract_server import FactoryContractServer

    server = FactoryContractServer()
    mock_gen = contract_mock_client.__wrapped__(server)
    mock_cl = next(mock_gen)
    assert mock_cl.headers.get("X-Factory-Test-Mode") == "contract-mock"

    # Ensure live_client skips when LIVE_MODE is False
    monkeypatch.setenv("FACTORY_E2E_LIVE", "0")
    live_gen = live_client.__wrapped__()
    with pytest.raises(pytest.skip.Exception) as exc_info:
        next(live_gen)
    assert "requires live cluster" in str(exc_info.value)


# ---------------------------------------------------------------------------
# FIX-16: Kaggle simulated reports cannot carry LIVE_PASS
# ---------------------------------------------------------------------------
def test_fix_16_kaggle_simulated_report_cannot_carry_live_pass():
    """FIX-16: Simulated report cannot carry LIVE_PASS."""
    from deployments.kaggle.certify_kernel import PreflightContractSimulator

    simulator = PreflightContractSimulator()
    res = simulator.execute_all()
    assert res["all_gates_passed"] is True
    assert res["live_pass"] is False
    assert res["certification_classification"] == "SIMULATED_PASS"

    # Check saved artifacts
    cert_path = Path("deployments/kaggle/certification_results.json")
    if cert_path.exists():
        data = json.loads(cert_path.read_text(encoding="utf-8"))
        assert data.get("live_pass") is False
        assert data.get("certification_classification") == "SIMULATED_PASS"

    live_report_path = Path("deployments/kaggle/live_hardware_attested_report.json")
    if live_report_path.exists():
        data = json.loads(live_report_path.read_text(encoding="utf-8"))
        assert data.get("live_pass") is False
        assert data.get("execution_mode") == "synthetic_tensor_benchmark"


# ---------------------------------------------------------------------------
# FIX-17: Worker error taxonomy, stage and retry semantics
# ---------------------------------------------------------------------------
def test_fix_17_worker_error_taxonomy_and_retry_semantics():
    """FIX-17: Errors not all reduced to generic SERVICE_UNAVAILABLE; actionable taxonomy."""
    # 1. Invalid input -> INVALID_INPUT, non-retryable
    code, msg, retry = classify_worker_error(InvalidRequestError("Bad parameter"), subsystem="image")
    assert code == ErrorCode.INVALID_INPUT
    assert retry is False
    assert "/Users" not in msg and "/Volumes" not in msg

    # 2. Timeout -> INFERENCE_TIMEOUT, retryable
    code, msg, retry = classify_worker_error(TimeoutError("Inference timed out"), subsystem="video", stage="sampling")
    assert code == ErrorCode.INFERENCE_TIMEOUT
    assert retry is True
    assert "sampling" in msg

    # 3. OOM -> INSUFFICIENT_VRAM, non-retryable
    code, msg, retry = classify_worker_error(RuntimeError("CUDA out of memory in backward pass"), subsystem="video")
    assert code == ErrorCode.INSUFFICIENT_VRAM
    assert retry is False

    # 4. Weights missing -> WEIGHTS_MISSING, non-retryable
    code, msg, retry = classify_worker_error(FileNotFoundError("Model weights checkpoint not found: qwen.gguf"), subsystem="image")
    assert code == ErrorCode.WEIGHTS_MISSING
    assert retry is False

    # 5. Connection refused -> BACKEND_UNAVAILABLE, retryable
    code, msg, retry = classify_worker_error(ConnectionRefusedError("Connection refused on port 8008"), subsystem="music")
    assert code == ErrorCode.BACKEND_UNAVAILABLE
    assert retry is True

    # 6. Backend crashed -> BACKEND_CRASHED, retryable
    code, msg, retry = classify_worker_error(RuntimeError("Process terminated with SIGKILL"), subsystem="image", stage="sampling")
    assert code == ErrorCode.BACKEND_CRASHED
    assert retry is True


# ---------------------------------------------------------------------------
# FIX-18: Studio Web streaming inspector distinguishes demo telemetry
# ---------------------------------------------------------------------------
def test_fix_18_streaming_inspector_no_unlabeled_fake_telemetry():
    """FIX-18: Live UI contains no unlabeled fake telemetry."""
    tsx_path = Path("factory/apps/studio-web/src/components/views/StreamingInspectorView.tsx")
    assert tsx_path.exists()
    content = tsx_path.read_text(encoding="utf-8")

    assert "DEMO SIMULATION" in content
    assert "SYNTHETIC WAVEFORM" in content
    assert "STATIC PREVIEW" in content
    assert "[SYNTHETIC DEMO]" in content


# ---------------------------------------------------------------------------
# FIX-19: CI security checks and updated dependencies
# ---------------------------------------------------------------------------
def test_fix_19_security_checks_and_dependencies():
    """FIX-19: Deprecated pynvml replaced by nvidia-ml-py and CI enforces blocking checks."""
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "nvidia-ml-py>=12.535.133" in pyproject
    assert "pynvml>=" not in pyproject

    security_yml = Path(".github/workflows/security.yml")
    if security_yml.exists():
        ci_security = security_yml.read_text(encoding="utf-8")
        # Verify uv audit is blocking without swallowed warnings
        assert "uv audit ||" not in ci_security
        assert "npm audit --audit-level=critical" in ci_security


# ---------------------------------------------------------------------------
# FIX-20: Deploy config trims dead Redis and documented deploy path
# ---------------------------------------------------------------------------
def test_fix_20_deploy_config_trimmed():
    """FIX-20: Trim dead optional Redis/S3/legacy assumptions and document single deploy path."""
    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert "REDIS_URL" not in env_example
    assert "POSTGRES_HOST" not in env_example
    assert "QUEUE_BACKEND=memory" in env_example
    assert "STORAGE_BACKEND=ephemeral_disk" in env_example

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Single Working Deploy Path" in readme
    assert "uvicorn factory_control_api.server:app" in readme
