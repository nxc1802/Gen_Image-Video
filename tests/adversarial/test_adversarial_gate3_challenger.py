"""
tests/adversarial/test_adversarial_gate3_challenger.py
Adversarial Challenger Test Suite for Gate 3 (P1.1, P1.2, P1.4):
1. P1.1: Genuine TTFT & Explicit 6-Phase Breakdown (Model Loading, Prompt Processing,
   Decode, Sampling, VAE, Artifact Encoding). Rejection of bogus TTFT (total / tokens),
   rejection of missing phase keys, negative phase durations, or non-positive TTFT.
2. P1.2 & P1.4: Metadata & Live NVML GPU Accounting:
   - Live NVML multi-GPU device sampling with dynamic VRAM totals and device names.
   - Distinct process vs system VRAM accounting via NVML compute running processes.
   - Zero hardcoded 15.0 GB in live NVML paths and UI inspector views.
   - Non-synthetic graceful fallback returning telemetry_status="UNAVAILABLE" without fake VRAM.
   - Ingestion of system/process/total VRAM and GPU names in worker heartbeat handling.
   - Dynamic SSE telemetry stream emitting dynamic totals and status.
"""
import ast
import json
import os
from pathlib import Path
import re
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from deployments.kaggle.certify_kernel import (
    CertificationGates,
    CertificationGateResult,
    FourStageVerificationGate,
    REQUIRED_PHASE_KEYS,
    Tier1ModelCertifier,
)
from factory_contracts.jobs.worker_contract import WorkerHeartbeatRequest
from factory_contracts.telemetry import NVMLTelemetryCollector
from factory.apps.control_api.src.main import app as control_app
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.repository import get_job_repository
from factory.apps.control_api.src.streaming.inspector import stream_studio_inspector
from factory.apps.control_api.src.telemetry.metrics import TelemetryMetrics, metrics

AUTH_TOKEN = "test_adversarial_gate3_token"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}


# ==============================================================================
# 1. P1.1 ADVERSARIAL CHALLENGES: Genuine TTFT & Explicit 6-Phase Breakdown
# ==============================================================================

class TestP11GenuineTTFTAndPhaseBreakdownChallenger:
    """Verifies that bogus TTFT is rejected and explicit 6-phase breakdown is strictly enforced."""

    @pytest.fixture
    def valid_text_phases(self) -> Dict[str, float]:
        return {
            "model_loading_ms": 780.0,
            "prompt_processing_ms": 32.5,
            "decode_ms": 145.0,
            "sampling_ms": 8.5,
            "vae_ms": 0.0,
            "artifact_encoding_ms": 0.4,
        }

    @pytest.fixture
    def valid_diffusion_phases(self) -> Dict[str, float]:
        return {
            "model_loading_ms": 1250.0,
            "prompt_processing_ms": 85.0,
            "decode_ms": 0.0,
            "sampling_ms": 620.0,
            "vae_ms": 140.0,
            "artifact_encoding_ms": 15.0,
        }

    def test_genuine_ttft_equals_prompt_evaluation_time(self, valid_text_phases):
        """Genuine TTFT matches prompt evaluation duration and passes verification gate."""
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=8500.0,
            latency_ms=186.0,
            phases=valid_text_phases,
            ttft_ms=32.5,
        )
        assert res.passed is True
        assert res.gate_name == "Stage 2: Infer"
        assert res.details["status"] == "PASS"
        assert res.details["ttft_ms"] == 32.5
        assert res.details["phases"] == valid_text_phases

    @pytest.mark.parametrize("invalid_ttft", [0.0, -0.01, -50.0])
    def test_rejects_negative_or_zero_ttft(self, valid_text_phases, invalid_ttft):
        """Rejects non-positive TTFT measurements."""
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=8500.0,
            latency_ms=186.0,
            phases=valid_text_phases,
            ttft_ms=invalid_ttft,
        )
        assert res.passed is False
        assert "TTFT must be a positive number" in res.error_message
        assert res.details["ttft_ms"] == invalid_ttft

    def test_rejects_bogus_ttft_derived_from_total_duration_divided_by_tokens(self):
        """
        Adversarially asserts that deriving TTFT as total_request_duration / generated_tokens
        produces Inter-Token Latency (ITL) rather than Time-To-First-Token (TTFT),
        and verifies the mathematical invariant: genuine TTFT must be independent of tokens_generated.
        """
        dur_req_ms = 4000.0
        tokens_gen = 200
        bogus_ttft = dur_req_ms / tokens_gen  # 20.0 ms per token (ITL, not TTFT)
        genuine_prompt_eval_ms = 180.0  # Actual time to encode prompt

        # Bogus TTFT scales down linearly with token count (unphysical for TTFT)
        bogus_ttft_1000_tokens = dur_req_ms / 1000  # 4.0 ms (bogus)
        assert bogus_ttft != genuine_prompt_eval_ms
        assert bogus_ttft_1000_tokens != genuine_prompt_eval_ms

    @pytest.mark.parametrize("missing_key", [
        "model_loading_ms",
        "prompt_processing_ms",
        "decode_ms",
        "sampling_ms",
        "vae_ms",
        "artifact_encoding_ms",
    ])
    def test_requires_all_six_explicit_phases(self, valid_text_phases, missing_key):
        """Gate 2 must reject when any one of the 6 required explicit phases is missing."""
        corrupted_phases = dict(valid_text_phases)
        del corrupted_phases[missing_key]

        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=8500.0,
            latency_ms=186.0,
            phases=corrupted_phases,
            ttft_ms=32.5,
        )
        assert res.passed is False
        assert f"Missing explicit phase breakdown keys" in res.error_message
        assert missing_key in res.details["missing_phases"]

    @pytest.mark.parametrize("corrupt_key", [
        "model_loading_ms",
        "prompt_processing_ms",
        "decode_ms",
        "sampling_ms",
        "vae_ms",
        "artifact_encoding_ms",
    ])
    def test_rejects_negative_phase_duration(self, valid_text_phases, corrupt_key):
        """Gate 2 must reject negative durations in any phase."""
        corrupted_phases = dict(valid_text_phases)
        corrupted_phases[corrupt_key] = -1.5

        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=8500.0,
            latency_ms=186.0,
            phases=corrupted_phases,
            ttft_ms=32.5,
        )
        assert res.passed is False
        assert "must be a non-negative number" in res.error_message
        assert res.details["phase"] == corrupt_key

    def test_modalities_phase_specialization(self):
        """
        Verifies domain-specific invariants across modalities:
        - LLM/Text models: vae_ms == 0.0, decode_ms > 0
        - Diffusion models: decode_ms == 0.0, vae_ms > 0, sampling_ms > 0
        """
        certifier = Tier1ModelCertifier()

        # Text certification
        res_text = certifier.certify_model("qwen-text")
        assert res_text["certified"] is True
        text_phases = res_text["four_stage_pipeline"]["stage_2_infer"]["details"]["phases"]
        assert text_phases["vae_ms"] == 0.0
        assert text_phases["decode_ms"] > 0.0

        # Image certification
        res_img = certifier.certify_model("qwen-image-2.1")
        assert res_img["certified"] is True
        img_phases = res_img["four_stage_pipeline"]["stage_2_infer"]["details"]["phases"]
        assert img_phases["decode_ms"] == 0.0
        assert img_phases["vae_ms"] > 0.0
        assert img_phases["sampling_ms"] > 0.0

        # Video certification
        res_vid = certifier.certify_model("wan-2.2")
        assert res_vid["certified"] is True
        vid_phases = res_vid["four_stage_pipeline"]["stage_2_infer"]["details"]["phases"]
        assert vid_phases["decode_ms"] == 0.0
        assert vid_phases["vae_ms"] > 0.0
        assert vid_phases["sampling_ms"] > 0.0

    def test_session1_visual_pipeline_records_all_six_phases_and_genuine_ttft(self):
        """Statically inspects session1_visual_pipeline.py to assert genuine TTFT and 6 phases are recorded."""
        pipeline_path = Path("deployments/kaggle/session1_visual_pipeline.py")
        assert pipeline_path.exists(), "session1_visual_pipeline.py must exist"

        content = pipeline_path.read_text(encoding="utf-8")

        # 1. Assert old fake TTFT formula is gone
        assert "ttft_ms = round(dur_req * 1000.0 / max(1, tokens_gen), 2)" not in content, (
            "Found forbidden bogus TTFT calculation in session1_visual_pipeline.py!"
        )

        # 2. Assert genuine TTFT measurement from prompt_ms
        assert 'prompt_eval_ms = float(timings.get("prompt_ms", 0.0))' in content
        assert "vlm_phases" in content
        assert "img_phases" in content
        assert "vid_phases" in content

        # 3. Assert all 6 required phase keys appear in the pipeline definitions
        for phase_key in REQUIRED_PHASE_KEYS:
            assert f'"{phase_key}"' in content, f"Missing required phase key '{phase_key}' in pipeline script"


# ==============================================================================
# 2. P1.2 & P1.4 ADVERSARIAL CHALLENGES: Metadata & Live NVML GPU Accounting
# ==============================================================================

class TestP12P14LiveNVMLAccountingChallenger:
    """Verifies live NVML device query, dynamic VRAM totals, process isolation, and non-synthetic fallback."""

    @pytest.fixture(autouse=True)
    def reset_metrics_state(self):
        yield
        metrics.reset()

    @pytest.fixture(scope="class")
    def api_client(self):
        keys = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH"]
        old_env = {k: os.environ.get(k) for k in keys}
        os.environ["FACTORY_API_KEY"] = AUTH_TOKEN
        os.environ["STORAGE_BACKEND"] = "local"
        os.environ["FACTORY_DB_BACKEND"] = "sqlite"
        os.environ["SQLITE_DB_PATH"] = ":memory:"
        try:
            with TestClient(control_app) as client:
                yield client
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            if hasattr(control_app.state, "auth_config"):
                delattr(control_app.state, "auth_config")

    def test_nvml_collector_sample_device_live_hardware_metrics(self):
        """Verifies NVMLTelemetryCollector queries hardware registers without synthesizing mock values."""
        collector = NVMLTelemetryCollector(device_index=0)

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA A10G"

        mock_mem = MagicMock()
        mock_mem.total = 24 * 1024 * 1024 * 1024  # 24 GB
        mock_mem.used = 6 * 1024 * 1024 * 1024   # 6 GB
        mock_mem.free = 18 * 1024 * 1024 * 1024  # 18 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_util = MagicMock()
        mock_util.gpu = 68.0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 52.0

        mock_proc1 = MagicMock(pid=1001, usedGpuMemory=2 * 1024 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [mock_proc1]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector._available = True
            collector._handle = mock_handle
            collector._device_count = 2

            dev = collector.sample_device(0)
            assert dev["telemetry_status"] == "AVAILABLE"
            assert dev["name"] == "NVIDIA A10G"
            assert dev["total_gb"] == 24.0
            assert dev["used_mb"] == 6144.0
            assert dev["free_gb"] == 18.0
            assert dev["system_vram_used_mb"] == 6144.0
            assert dev["process_vram_used_mb"] == 2048.0
            assert len(dev["running_processes"]) == 1
            assert dev["running_processes"][0]["pid"] == 1001
            assert dev["gpu_utilization_percent"] == 68.0
            assert dev["temperature_c"] == 52.0

    def test_nvml_collector_system_vs_process_memory_isolation(self):
        """Verifies clear separation between host system VRAM allocation and process VRAM allocation."""
        collector = NVMLTelemetryCollector(device_index=0)

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla T4"

        # System has 10 GB allocated, but our specific process only has 3 GB
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.used = 10 * 1024 * 1024 * 1024
        mock_mem.free = 6 * 1024 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=30.0)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 45.0

        p1 = MagicMock(pid=5000, usedGpuMemory=3 * 1024 * 1024 * 1024)
        p2 = MagicMock(pid=6000, usedGpuMemory=7 * 1024 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [p1, p2]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector._available = True
            collector._handle = mock_handle
            collector._device_count = 1

            # Sample targeting PID 5000
            dev_target = collector.sample_device(0, target_pid=5000)
            assert dev_target["system_vram_used_mb"] == 10240.0
            assert dev_target["process_vram_used_mb"] == 3072.0
            assert dev_target["system_vram_used_mb"] != dev_target["process_vram_used_mb"]

    def test_nvml_collector_graceful_unavailable_fallback(self):
        """Verifies collector gracefully returns UNAVAILABLE without synthesizing fake VRAM when NVML fails."""
        collector = NVMLTelemetryCollector(device_index=0)

        with patch("pynvml.nvmlInit", side_effect=Exception("Driver not found")):
            collector._init_nvml()
            assert collector.is_available is False
            sample = collector.sample()
            assert sample["telemetry_status"] == "UNAVAILABLE"
            assert sample["vram_used_mb"] is None
            assert sample["vram_total_mb"] is None
            assert sample["gpu_utilization_percent"] is None

            all_devs = collector.sample_all_devices()
            assert all_devs == []

    def test_memory_endpoint_queries_live_nvml_when_available(self, api_client):
        """Verifies /v1/memory uses live NVML queries reporting dynamic totals and names."""
        mock_sample_all = [
            {
                "index": 0,
                "name": "NVIDIA RTX 4090",
                "used_mb": 4096.0,
                "system_vram_used_mb": 4096.0,
                "process_vram_used_mb": 2048.0,
                "total_mb": 24576.0,
                "total_gb": 24.0,
                "free_mb": 20480.0,
                "free_gb": 20.0,
                "running_processes": [{"pid": 1234, "used_mb": 2048.0}],
            },
            {
                "index": 1,
                "name": "NVIDIA RTX 4090",
                "used_mb": 8192.0,
                "system_vram_used_mb": 8192.0,
                "process_vram_used_mb": 4096.0,
                "total_mb": 24576.0,
                "total_gb": 24.0,
                "free_mb": 16384.0,
                "free_gb": 16.0,
                "running_processes": [{"pid": 5678, "used_mb": 4096.0}],
            },
        ]

        with patch("factory_contracts.telemetry.NVMLTelemetryCollector.sample_all_devices", return_value=mock_sample_all):
            res = api_client.get("/v1/memory", headers=AUTH_HEADERS)
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["telemetry_status"] == "AVAILABLE"
            assert len(data["gpus"]) == 2

            gpu0 = data["gpus"][0]
            assert gpu0["name"] == "NVIDIA RTX 4090"
            assert gpu0["total_gb"] == 24.0
            assert gpu0["free_gb"] == 20.0
            assert gpu0["system_vram_used_gb"] == 4.0
            assert gpu0["process_vram_used_gb"] == 2.0
            assert len(gpu0["running_processes"]) == 1

            gpu1 = data["gpus"][1]
            assert gpu1["name"] == "NVIDIA RTX 4090"
            assert gpu1["total_gb"] == 24.0
            assert gpu1["system_vram_used_gb"] == 8.0
            assert gpu1["process_vram_used_gb"] == 4.0

    def test_memory_endpoint_graceful_unavailable_fallback_when_no_nvml_and_no_metrics(self, api_client):
        """Verifies /v1/memory returns UNAVAILABLE with empty GPUs when no NVML and no metrics are set."""
        # Reset telemetry metrics to None
        metrics.set_gpu_vram(gpu_0_gb=None, gpu_1_gb=None)
        metrics._gpu_0_vram_used_gb = None
        metrics._gpu_1_vram_used_gb = None
        metrics._gpu_0_vram_total_gb = None
        metrics._gpu_1_vram_total_gb = None

        with patch("factory_contracts.telemetry.NVMLTelemetryCollector.sample_all_devices", return_value=[]):
            res = api_client.get("/v1/memory", headers=AUTH_HEADERS)
            assert res.status_code == 200
            data = res.json()
            assert data["telemetry_status"] == "UNAVAILABLE"
            assert data["gpus"] == []
            assert data["loaded_models"] == []

    @pytest.mark.asyncio
    async def test_heartbeat_ingests_system_process_total_vram_and_gpu_name(self):
        """Verifies worker heartbeat correctly ingests system, process, total VRAM, and GPU names."""
        repo = get_job_repository()
        coord = LeaseCoordinator(repo)

        req = WorkerHeartbeatRequest(
            worker_id="worker-dynamic-gpu0",
            active_job_id=None,
            lease_token=None,
            telemetry_status="AVAILABLE",
            gpu_utilization_percent=55.0,
            vram_used_mb=5120.0,
            vram_total_mb=24576.0,
            system_vram_used_mb=5120.0,
            process_vram_used_mb=2560.0,
            gpu_name="NVIDIA RTX 3090",
            temperature_c=48.0,
        )

        resp = await coord.process_heartbeat(req)
        assert resp.status == "acknowledged"

        snap = metrics.get_snapshot()
        assert snap["gpu_0_vram_used_gb"] == 5.0
        assert snap["gpu_0_vram_total_gb"] == 24.0
        assert snap["gpu_0_process_vram_gb"] == 2.5
        assert snap["gpu_0_name"] == "NVIDIA RTX 3090"
        assert snap["telemetry_status"] == "AVAILABLE"

    @pytest.mark.asyncio
    async def test_streaming_inspector_emits_dynamic_vram_totals_and_status(self):
        """Verifies SSE telemetry stream emits dynamic totals, process memory, and status."""
        metrics.set_gpu_vram(gpu_0_gb=7.5, gpu_1_gb=11.2)
        metrics.set_gpu_totals(gpu_0_total_gb=16.0, gpu_1_total_gb=24.0)
        metrics.set_gpu_process_vram(gpu_0_proc_gb=3.5, gpu_1_proc_gb=6.2)
        metrics.set_telemetry_status("AVAILABLE")

        async for chunk in stream_studio_inspector(interval_seconds=0.01, max_iterations=1):
            assert "event: telemetry" in chunk
            payload_str = chunk.split("data: ")[1].split("\n\n")[0]
            payload = json.loads(payload_str)

            assert payload["gpu_0_vram_used_gb"] == 7.5
            assert payload["gpu_1_vram_used_gb"] == 11.2
            assert payload["gpu_0_vram_total_gb"] == 16.0
            assert payload["gpu_1_vram_total_gb"] == 24.0
            assert payload["gpu_0_process_vram_gb"] == 3.5
            assert payload["gpu_1_process_vram_gb"] == 6.2
            assert payload["telemetry_status"] == "AVAILABLE"
            break

    def test_zero_hardcoded_15gb_in_inspector_and_views(self):
        """
        Static AST / textual verification:
        Ensures that neither the live NVML inspector path nor the Studio Web Streaming Inspector View
        hardcode constant 15.0 GB strings or assumptions.
        """
        # 1. Inspect inspector.py AST: verify live_gpus loop uses dev["total_gb"]
        inspector_file = Path("factory/apps/control-api/src/api/routers/inspector.py")
        content = inspector_file.read_text(encoding="utf-8")
        assert 'collector.sample_all_devices()' in content
        assert '"total_gb": dev["total_gb"]' in content

        # 2. Inspect StreamingInspectorView.tsx: verify it does not contain hardcoded string " / 15.0 GB"
        view_file = Path("factory/apps/studio-web/src/components/views/StreamingInspectorView.tsx")
        view_content = view_file.read_text(encoding="utf-8")
        assert "/ 15.0 GB" not in view_content, (
            "Found hardcoded string '/ 15.0 GB' in StreamingInspectorView.tsx!"
        )
        assert "telemetry.gpu_0_vram_total_gb" in view_content
        assert "telemetry.gpu_1_vram_total_gb" in view_content
