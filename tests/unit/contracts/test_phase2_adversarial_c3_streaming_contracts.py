"""
tests/unit/contracts/test_phase2_adversarial_c3_streaming_contracts.py
Adversarial stress-test suite for Phase 2 API Contracts, Query Auth & Media Streaming:
- Submodule re-exports under `contracts` vs `factory_contracts`
- Task and model normalization matrix, aliases, whitespace, casing, and edge cases
- OpenAI Chat Completions & reasoning_effort boundaries ('none', 'low', 'medium', 'high', invalid 'ultra', empty)
- Reasoning content decoupling in Chat Completions and Job Result Delivery
- Query parameter token authentication (?token=...) for browser media streaming
- HTTP 206 Partial Content video Range header parsing and inline Content-Disposition
"""
import os
import asyncio
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

import factory_contracts
import contracts
from factory_contracts.jobs.job_contract import FactoryJob, JobState, JobProgress, JobSubmissionRequest
from factory_contracts.tasks import (
    normalize_task,
    normalize_model,
    CANONICAL_TASK_MAP,
    CANONICAL_MODEL_MAP,
)
from services.text.src.models import ChatCompletionRequest, ChatMessage
from services.text.src.engine import decouple_think_tokens

VALID_TOKEN = "sk-factory-prod-challenger-token"
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture(scope="module")
def control_client():
    keys_to_preserve = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH", "VLLM_TEXT_URL"]
    old_env = {k: os.environ.get(k) for k in keys_to_preserve}
    os.environ["FACTORY_API_KEY"] = VALID_TOKEN
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"
    os.environ["VLLM_TEXT_URL"] = "http://mock-vllm-text:8000"

    from factory_control_api import app
    try:
        with TestClient(app) as client:
            yield client
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")


# =========================================================================
# 1. Contracts & Alias Re-export Probes
# =========================================================================

class TestPhase2ContractsReExport:
    """Adversarially probe contracts re-exports and direct submodule imports."""

    def test_contracts_all_7_submodules_accessible(self):
        """
        Verify that `import contracts` allows accessing all 7 submodules
        (errors, jobs, manifests, models, resilience, tasks, telemetry)
        and that their objects are identical to `factory_contracts`.
        """
        submodules = ["errors", "jobs", "manifests", "models", "resilience", "tasks", "telemetry"]
        for sub in submodules:
            assert hasattr(contracts, sub), f"contracts missing submodule attribute: {sub}"
            c_mod = getattr(contracts, sub)
            f_mod = getattr(factory_contracts, sub)
            assert c_mod is f_mod, f"contracts.{sub} is not identical to factory_contracts.{sub}"

    def test_contracts_all_list_contains_submodules(self):
        """Verify contracts.__all__ contains all 7 submodules and factory_contracts symbols."""
        for sub in ["errors", "jobs", "manifests", "models", "resilience", "tasks", "telemetry"]:
            assert sub in contracts.__all__, f"{sub} missing from contracts.__all__"
        assert "JobSubmissionRequest" in contracts.__all__
        assert "ModelManifest" in contracts.__all__

    def test_from_contracts_import_submodules(self):
        """Verify `from contracts import errors, jobs, ...` syntax works."""
        from contracts import errors, jobs, manifests, models, resilience, tasks, telemetry
        assert errors.ErrorCode is factory_contracts.errors.ErrorCode
        assert jobs.JobSubmissionRequest is factory_contracts.jobs.JobSubmissionRequest
        assert manifests.ModelManifest is factory_contracts.manifests.ModelManifest
        assert resilience.CircuitBreaker is factory_contracts.resilience.CircuitBreaker
        assert tasks.normalize_task is factory_contracts.tasks.normalize_task
        assert telemetry.NVMLTelemetryCollector is factory_contracts.telemetry.NVMLTelemetryCollector

    def test_contracts_deep_submodule_import_limitation(self):
        """
        Adversarially document Python import machinery behavior:
        Direct dotted imports like `import contracts.jobs` fail with ModuleNotFoundError
        unless sys.modules['contracts.<submodule>'] is registered.
        """
        import sys
        # Without explicit sys.modules mapping, importing contracts.jobs as a module fails
        if "contracts.jobs" not in sys.modules:
            with pytest.raises(ModuleNotFoundError):
                __import__("contracts.jobs")


# =========================================================================
# 2. Task & Model Normalization Matrix & Edge Cases
# =========================================================================

class TestPhase2NormalizationMatrix:
    """Adversarially probe normalization with noise, casing, whitespace, and aliases."""

    @pytest.mark.parametrize("alias,canonical", [
        ("  IMAGE.T2I  ", "text_to_image"),
        ("\timage.i2i\n", "image_to_image"),
        ("  VIDEO.T2V \r\n", "text_to_video"),
        (" video.i2v ", "image_to_video"),
        ("video.fl2v", "first_last_frame_to_video"),
        ("MUSIC.T2M", "text_to_music"),
        ("SPEECH.TTS", "speech_synthesis"),
        ("speech.asr", "speech_transcription"),
        ("enhance.sr_image", "image_super_resolution"),
        ("text.chat", "chat_completion"),
        ("vision.vlm", "visual_language"),
        ("  T2I  ", "text_to_image"),
        ("  T2V  ", "text_to_video"),
        ("  FL2V  ", "first_last_frame_to_video"),
        ("  TTS  ", "speech_synthesis"),
    ])
    def test_normalize_task_aliases_with_noise(self, alias: str, canonical: str):
        assert normalize_task(alias) == canonical

    @pytest.mark.parametrize("alias,canonical", [
        ("  QWEN-IMAGE  ", "qwen-image-2.1"),
        ("\tqwen-image-turbo\n", "qwen-image-2.1"),
        ("  WAN  ", "wan-2.2"),
        ("  wan2.2  ", "wan-2.2"),
        ("  MINIMAX  ", "minimax-h3"),
        ("ace-step", "ace-step-1.5"),
        ("  yue  ", "yue2"),
        ("  QWEN3-TTS  ", "qwen-speech"),
        ("qwen3-asr", "qwen-speech"),
        ("enhancement-suite", "enhancement-suite-v1"),
    ])
    def test_normalize_model_aliases_with_noise(self, alias: str, canonical: str):
        assert normalize_model(alias) == canonical

    def test_job_submission_request_validator_normalizes_fields(self):
        """Verify JobSubmissionRequest normalizes task and model automatically."""
        req = JobSubmissionRequest(
            service="image",
            model="  QWEN-IMAGE  \n",
            task="  IMAGE.T2I \t",
            inputs={"prompt": "cyberpunk city"},
        )
        assert req.model == "qwen-image-2.1"
        assert req.task == "text_to_image"

    def test_job_submission_request_none_and_empty_edge_cases(self):
        """Verify empty and None values in JobSubmissionRequest do not crash validators."""
        req1 = JobSubmissionRequest(
            service="image",
            model="",
            task="",
            inputs={"prompt": "test"},
        )
        assert req1.model == ""
        assert req1.task == ""

    def test_post_v1_jobs_normalizes_aliased_payload(self, control_client):
        """Verify POST /v1/jobs normalizes task and model during admission."""
        resp = control_client.post(
            "/v1/jobs",
            headers=AUTH_HEADERS,
            json={
                "service": "image",
                "model": "  QWEN-IMAGE  ",
                "task": "  IMAGE.T2I  ",
                "inputs": {"prompt": "a high resolution landscape"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model"] == "qwen-image-2.1"
        assert data["task"] == "text_to_image"
        assert data["service"] == "image"


# =========================================================================
# 3. OpenAI Chat Completions & Reasoning Effort Boundaries
# =========================================================================

class TestPhase2OpenAIChatCompletions:
    """Adversarially probe /v1/chat/completions, reasoning_effort, and <think> decoupling."""

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "NONE", "LOW", "Medium", "HIGH"])
    def test_reasoning_effort_valid_boundaries(self, effort: str):
        req = ChatCompletionRequest(
            model="qwen-text",
            messages=[ChatMessage(role="user", content="Solve equation")],
            reasoning_effort=effort,
        )
        assert req.reasoning_effort == effort.lower()

    @pytest.mark.parametrize("invalid_effort", ["ultra", "extreme", "maximum", "0", "false", ""])
    def test_reasoning_effort_invalid_fails_closed_schema(self, invalid_effort: str):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="qwen-text",
                messages=[ChatMessage(role="user", content="Test")],
                reasoning_effort=invalid_effort,
            )

    def test_chat_completions_invalid_effort_rejected_422(self, control_client):
        """Verify invalid reasoning_effort sent to /v1/chat/completions fails with HTTP 422."""
        res = control_client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "qwen-text",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "ultra",
            },
        )
        assert res.status_code == 422
        err = res.json()["error"]
        assert err["code"] == "VALIDATION_ERROR"
        assert "reasoning_effort" in str(err["details"])

    def test_chat_completions_none_effort_dispatches_cleanly(self, control_client):
        """Verify reasoning_effort='none' succeeds and passes through."""
        mock_resp = {
            "id": "chatcmpl-none-effort-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Direct response without thinking."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        with patch("services.text.src.engine.httpx.Client") as mock_client_cls:
            mock_inst = MagicMock()
            mock_inst.__enter__.return_value = mock_inst
            mock_inst.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp)
            mock_client_cls.return_value = mock_inst

            res = control_client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "qwen-text",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "reasoning_effort": "none",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["choices"][0]["message"]["content"] == "Direct response without thinking."

    def test_chat_completions_decouples_reasoning_content(self, control_client):
        """Verify <think> tokens are decoupled into reasoning_content on /v1/chat/completions."""
        mock_resp = {
            "id": "chatcmpl-decouple-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>\nStep 1: Compute matrix rank.\nStep 2: Conclude invertibility.\n</think>\nThe matrix is invertible.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 28, "total_tokens": 40},
        }
        with patch("services.text.src.engine.httpx.Client") as mock_client_cls:
            mock_inst = MagicMock()
            mock_inst.__enter__.return_value = mock_inst
            mock_inst.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp)
            mock_client_cls.return_value = mock_inst

            res = control_client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "qwen-text",
                    "messages": [{"role": "user", "content": "Is matrix invertible?"}],
                    "reasoning_effort": "high",
                },
            )
            assert res.status_code == 200
            msg = res.json()["choices"][0]["message"]
            assert msg["reasoning_content"] == "Step 1: Compute matrix rank.\nStep 2: Conclude invertibility."
            assert msg["content"] == "The matrix is invertible."
            assert "<think>" not in msg["content"]
            assert "</think>" not in msg["content"]

    def test_jobs_result_text_decouples_reasoning_content(self, control_client):
        """Verify GET /v1/jobs/{id}/result surfaces decoupled reasoning_content for text jobs."""
        from factory_control_api import app
        manager = app.state.job_manager

        job_id = "job-44444444-5555-6666-7777-888888888888"
        job = FactoryJob(
            job_id=job_id,
            request_id="req-44444444-5555-6666-7777-888888888888",
            service="text",
            model="qwen-text",
            task="chat_completion",
            inputs={"messages": [{"role": "user", "content": "analyze"}]},
            state=JobState.SUCCEEDED,
            progress=JobProgress(stage="completed", step=1, total=1, percent=100.0),
            result={"text": "<think>Deliberating step by step</think>Final verified response."},
        )
        asyncio.run(manager.repository.create_job(job))

        res = control_client.get(f"/v1/jobs/{job_id}/result", headers=AUTH_HEADERS)
        assert res.status_code == 200
        data = res.json()
        assert data["reasoning_content"] == "Deliberating step by step"
        assert data["text"] == "Final verified response."
        assert "<think>" not in data["text"]


# =========================================================================
# 4. Query Parameter Token Auth & Video Range Delivery
# =========================================================================

class TestPhase2QueryAuthAndMediaDelivery:
    """Adversarially probe ?token= auth, video Range requests, and Content-Disposition."""

    @pytest.fixture(autouse=True)
    def setup_video_artifact(self, control_client):
        from factory_control_api import app
        manager = app.state.job_manager
        scratch = app.state.scratch_disk

        self.job_id = "job-33333333-4444-5555-6666-777777777777"
        job = FactoryJob(
            job_id=self.job_id,
            request_id="req-33333333-4444-5555-6666-777777777777",
            service="video",
            model="wan-2.2",
            task="text_to_video",
            inputs={"prompt": "ocean waves in ultra slow motion"},
            state=JobState.SUCCEEDED,
            progress=JobProgress(stage="completed", step=10, total=10, percent=100.0),
        )
        asyncio.run(manager.repository.create_job(job))

        # Write 100-byte test video file
        job_dir = scratch.get_job_dir(self.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = job_dir / f"{self.job_id}.mp4"
        self.video_bytes = bytes([b % 256 for b in range(100)])
        self.video_path.write_bytes(self.video_bytes)

    def test_query_param_token_valid_key_returns_200(self, control_client):
        """GET /v1/jobs/{id}/result without Authorization header but with ?token=<valid> returns 200."""
        res = control_client.get(f"/v1/jobs/{self.job_id}/result?token={VALID_TOKEN}")
        assert res.status_code == 200
        assert res.headers.get("content-type") == "video/mp4"
        assert res.headers.get("content-disposition") == f'inline; filename="{self.job_id}.mp4"'
        assert len(res.content) == 100
        assert res.content == self.video_bytes

    def test_query_param_token_invalid_key_returns_401(self, control_client):
        """GET /v1/jobs/{id}/result with ?token=<invalid> returns 401 Unauthorized."""
        res = control_client.get(f"/v1/jobs/{self.job_id}/result?token=bad-token-xyz")
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "UNAUTHORIZED"

    def test_query_param_token_empty_returns_401(self, control_client):
        """GET /v1/jobs/{id}/result with ?token= (empty) returns 401 Unauthorized."""
        res = control_client.get(f"/v1/jobs/{self.job_id}/result?token=")
        assert res.status_code == 401

    def test_missing_auth_header_and_token_returns_401(self, control_client):
        """GET /v1/jobs/{id}/result without header and without token returns 401 Unauthorized."""
        res = control_client.get(f"/v1/jobs/{self.job_id}/result")
        assert res.status_code == 401

    def test_video_range_request_partial_content_206(self, control_client):
        """
        Send Range: bytes=0-49 with valid ?token=
        Assert:
        - HTTP 206 Partial Content
        - Accept-Ranges: bytes
        - Content-Range: bytes 0-49/100
        - Content-Disposition: inline
        - Content length is exactly 50 bytes
        """
        res = control_client.get(
            f"/v1/jobs/{self.job_id}/result?token={VALID_TOKEN}",
            headers={"Range": "bytes=0-49"},
        )
        assert res.status_code == 206
        assert res.headers.get("accept-ranges") == "bytes"
        assert res.headers.get("content-range") == "bytes 0-49/100"
        assert res.headers.get("content-disposition") == f'inline; filename="{self.job_id}.mp4"'
        assert len(res.content) == 50
        assert res.content == self.video_bytes[0:50]

    def test_video_range_request_open_ended_and_suffix(self, control_client):
        """Test open-ended range (bytes=50-) and suffix range (bytes=-20)."""
        # Open-ended: bytes 50 to 99
        res_open = control_client.get(
            f"/v1/jobs/{self.job_id}/result?token={VALID_TOKEN}",
            headers={"Range": "bytes=50-"},
        )
        assert res_open.status_code == 206
        assert res_open.headers.get("content-range") == "bytes 50-99/100"
        assert len(res_open.content) == 50
        assert res_open.content == self.video_bytes[50:]

        # Suffix: last 20 bytes (bytes 80 to 99)
        res_suf = control_client.get(
            f"/v1/jobs/{self.job_id}/result?token={VALID_TOKEN}",
            headers={"Range": "bytes=-20"},
        )
        assert res_suf.status_code == 206
        assert res_suf.headers.get("content-range") == "bytes 80-99/100"
        assert len(res_suf.content) == 20
        assert res_suf.content == self.video_bytes[80:]

    def test_video_range_request_unsatisfiable_416(self, control_client):
        """Test out-of-range request (bytes=100-200) returns HTTP 416."""
        res = control_client.get(
            f"/v1/jobs/{self.job_id}/result?token={VALID_TOKEN}",
            headers={"Range": "bytes=100-200"},
        )
        assert res.status_code == 416
