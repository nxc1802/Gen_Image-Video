"""
tests/unit/deployments/test_smoke_runner.py
Unit tests verifying the 4-stage real inference smoke runner:
- Stage 1: Load
- Stage 2: Infer
- Stage 3: Validate (detects empty outputs, None outputs)
- Stage 4: Profile
- Unverified models without real GPU telemetry retain validation_status: pending_live
"""
import pytest
from unittest.mock import patch, MagicMock

from deployments.kaggle.smoke_runner import ModelSmokeRunner, SmokeReport


@pytest.mark.asyncio
class TestSmokeRunner:
    async def test_full_4stage_pipeline_success(self):
        runner = ModelSmokeRunner(gpu_device=0)

        # Mock inference function returning valid non-empty output
        async def mock_infer(inputs):
            return b"\x89PNG\r\n\x1a\nvalid_png_content"

        # Mock client ready
        mock_client = MagicMock()
        mock_client.is_ready = MagicMock(return_value=True)

        report = await runner.execute_smoke(
            model_id="qwen-image-2.1",
            service="image",
            infer_callable=mock_infer,
            inputs={"prompt": "A test prompt"},
            client=mock_client,
        )

        assert report.model_id == "qwen-image-2.1"
        assert report.success is True
        assert len(report.stages) == 4
        assert [s.stage for s in report.stages] == ["load", "infer", "validate", "profile"]
        assert all(s.status == "passed" for s in report.stages)
        # Without real NVML, validation_status must remain pending_live (fail-closed, no fake pass)
        assert report.validation_status == "pending_live"

    async def test_infer_failure_halts_pipeline(self):
        runner = ModelSmokeRunner(gpu_device=0)

        def failing_infer(inputs):
            raise RuntimeError("CUDA out of memory during inference")

        report = await runner.execute_smoke(
            model_id="wan-2.2",
            service="video",
            infer_callable=failing_infer,
            inputs={"prompt": "OOM video"},
        )

        assert report.success is False
        assert report.validation_status == "pending_live"
        assert len(report.stages) == 2
        assert report.stages[0].status == "passed"
        assert report.stages[1].status == "failed"
        assert "CUDA out of memory" in report.stages[1].error

    async def test_empty_output_fails_validate_stage(self):
        runner = ModelSmokeRunner(gpu_device=0)

        def empty_infer(inputs):
            return b""  # Empty 0 bytes

        report = await runner.execute_smoke(
            model_id="qwen-image-2.1",
            service="image",
            infer_callable=empty_infer,
            inputs={"prompt": "empty"},
        )

        assert report.success is False
        assert report.validation_status == "pending_live"
        assert len(report.stages) == 3
        assert report.stages[2].stage == "validate"
        assert report.stages[2].status == "failed"
        assert "empty (0 bytes)" in report.stages[2].error

    async def test_real_nvml_hardware_verification(self):
        runner = ModelSmokeRunner(gpu_device=0)

        with patch.object(runner, "_sample_nvml", return_value={"vram_used_mb": 4500.0, "gpu_utilization_percent": 85}):
            async def mock_infer(inputs):
                return {"result": "ok"}

            report = await runner.execute_smoke(
                model_id="qwen-decision",
                service="decision",
                infer_callable=mock_infer,
                inputs={"prompt": "decision"},
            )

            assert report.success is True
            # Real hardware telemetry present -> verified
            assert report.validation_status == "verified"
            assert report.peak_vram_mb == 4500.0
