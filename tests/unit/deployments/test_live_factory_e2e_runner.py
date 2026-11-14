"""
tests/unit/deployments/test_live_factory_e2e_runner.py
Unit tests verifying strict fail-closed behavior, zero fallback, and contract assertions
in deployments/kaggle/live_factory_e2e_runner.py.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Legacy compatibility contracts with simulated readiness; no inference evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]

from deployments.kaggle.live_factory_e2e_runner import (
    BackendUnavailableError,
    HardwareAttestationError,
    InferenceExecutionError,
    VerificationContractError,
    execute_native_comfyui_workflow,
    execute_native_llama_completion,
    query_nvml_vram,
    run_live_factory_e2e,
)


class TestLiveFactoryE2ERunnerStrictContracts:
    """Verifies that live_factory_e2e_runner never generates synthetic PASS or fallback telemetry."""

    def test_query_nvml_vram_fails_closed_when_pynvml_unavailable(self, monkeypatch):
        """Must raise HardwareAttestationError when NVML is not available (NO 451.2 MB default!)."""
        with patch.dict("sys.modules", {"pynvml": None}):
            with pytest.raises(HardwareAttestationError) as exc_info:
                query_nvml_vram(gpu_device=0)
            assert "FACTORY_LIVE_E2E_PASS strictly requires genuine NVIDIA GPU" in str(exc_info.value)

    def test_comfyui_workflow_fails_closed_when_backend_offline(self):
        """Must raise BackendUnavailableError when ComfyUI is offline (NO fallback to old images!)."""
        with pytest.raises(BackendUnavailableError) as exc_info:
            execute_native_comfyui_workflow(
                comfy_url="http://127.0.0.1:59999",  # Non-existent port
                job_id="test_job_1",
                prompt_text="dragon",
                timeout_s=1.0,
            )
        assert "unreachable" in str(exc_info.value).lower() or "59999" in str(exc_info.value)

    def test_comfyui_workflow_raises_when_execution_fails(self):
        """Must raise InferenceExecutionError when ComfyUI returns error status."""
        mock_stats = MagicMock(status_code=200)
        mock_prompt_resp = MagicMock(status_code=200)
        mock_prompt_resp.json.return_value = {"prompt_id": "prompt-err-123"}
        mock_history_resp = MagicMock(status_code=200)
        mock_history_resp.json.return_value = {
            "prompt-err-123": {
                "status": {"status_str": "error", "messages": ["CUDA Out of Memory"]}
            }
        }

        with patch("httpx.get") as mock_get, patch("httpx.post") as mock_post:
            mock_get.side_effect = lambda url, **kwargs: mock_stats if "system_stats" in url else mock_history_resp
            mock_post.return_value = mock_prompt_resp

            with pytest.raises(InferenceExecutionError) as exc_info:
                execute_native_comfyui_workflow(
                    comfy_url="http://127.0.0.1:8188",
                    job_id="test_job_err",
                    prompt_text="dragon",
                    timeout_s=5.0,
                )
            assert "CUDA Out of Memory" in str(exc_info.value)

    def test_llama_completion_fails_closed_when_backend_offline(self):
        """Must raise BackendUnavailableError when llama-server is offline."""
        with pytest.raises(BackendUnavailableError) as exc_info:
            execute_native_llama_completion(
                llama_url="http://127.0.0.1:59998",
                job_id="test_llama_1",
                prompt_text="hello",
                timeout_s=1.0,
            )
        assert "unreachable" in str(exc_info.value).lower() or "59998" in str(exc_info.value)

    def test_run_live_factory_e2e_classifies_hardware_unavailable_on_no_gpu(self, tmp_path, monkeypatch):
        """When host has no GPU, report status must be HARDWARE_UNAVAILABLE and not PASSED."""
        with patch.dict("sys.modules", {"pynvml": None}):
            with pytest.raises(SystemExit) as exc_info:
                run_live_factory_e2e(output_dir=tmp_path)
            assert exc_info.value.code == 1

            report_file = tmp_path / "live_factory_e2e_report.json"
            assert report_file.exists()
            data = json.loads(report_file.read_text())
            assert data["status"] == "HARDWARE_UNAVAILABLE"
            assert "Physical NVML query failed" in data["error"]

    def test_run_live_factory_e2e_classifies_backend_unavailable_when_engine_offline(self, tmp_path):
        """When GPU is present but engine (ComfyUI) is offline, status must be BACKEND_UNAVAILABLE."""
        mock_nvml = {
            "device_name": "Tesla T4",
            "gpu_index": 0,
            "used_mb": 500.0,
            "total_mb": 15360.0,
            "free_mb": 14860.0,
        }
        with patch("deployments.kaggle.live_factory_e2e_runner.query_nvml_vram", return_value=mock_nvml):
            with pytest.raises(SystemExit) as exc_info:
                run_live_factory_e2e(
                    comfy_url="http://127.0.0.1:59997",
                    output_dir=tmp_path,
                )
            assert exc_info.value.code == 1

            report_file = tmp_path / "live_factory_e2e_report.json"
            assert report_file.exists()
            data = json.loads(report_file.read_text())
            assert data["status"] == "BACKEND_UNAVAILABLE"
            assert "ComfyUI server unreachable" in data["error"]

    def test_run_live_factory_e2e_full_success_when_engine_and_gpu_online(self, tmp_path):
        """Verifies full successful E2E execution when genuine GPU and ComfyUI engine are available."""
        mock_nvml = {
            "device_name": "Tesla T4",
            "gpu_index": 0,
            "used_mb": 13053.0,
            "total_mb": 15360.0,
            "free_mb": 2307.0,
        }
        valid_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20000 + b"IEND\xaeB`\x82"

        mock_engine_res = {
            "artifact_bytes": valid_png_bytes,
            "artifact_name": "Studio_Qwen_Image_2.1_0001.png",
            "prompt_id": "prompt-e2e-ok",
            "engine_duration_ms": 1250.0,
            "seed": 42,
        }

        with patch("deployments.kaggle.live_factory_e2e_runner.query_nvml_vram", return_value=mock_nvml), \
             patch("deployments.kaggle.live_factory_e2e_runner.execute_native_comfyui_workflow", return_value=mock_engine_res):

            res = run_live_factory_e2e(
                target_service="image",
                target_model="qwen-image-2.1-turbo",
                output_dir=tmp_path,
            )

            assert res["status"] == "PASSED"
            assert res["level"] == "FACTORY_LIVE_E2E_PASS"
            assert res["telemetry"]["initial_vram"]["device_name"] == "Tesla T4"
            assert res["lifecycle_stages"]["submit"]["status"] == "passed"
            assert res["lifecycle_stages"]["lease"]["status"] == "passed"
            assert res["lifecycle_stages"]["inference"]["status"] == "passed"
            assert res["lifecycle_stages"]["inference"]["backend"] == "comfyui"
            assert res["lifecycle_stages"]["inference"]["prompt_id"] == "prompt-e2e-ok"
            assert res["lifecycle_stages"]["download_and_cleanup"]["scratch_disk_purged"] is True
            assert res["lifecycle_stages"]["download_and_cleanup"]["bytes_received"] == len(valid_png_bytes)


@pytest.fixture(autouse=True)
def explicit_legacy_runner_app(monkeypatch):
    """The retained runner's import is a test double; never Factory E2E evidence."""
    from factory.apps.control_api.src.legacy import app
    monkeypatch.setattr("factory.apps.control_api.src.main.app",app)
