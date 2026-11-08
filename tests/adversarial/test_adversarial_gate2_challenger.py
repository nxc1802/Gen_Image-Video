"""
tests/adversarial/test_adversarial_gate2_challenger.py
Empirical Adversarial Verification Suite for Gate 2 Acceptance Criteria (P0.2, P0.5, P0.6).

Challenger: teamwork_challenger_g2_1
Criteria challenged:
1. P0.2: Reasoning token isolation (<think> tokens decoupled from content in streaming and non-streaming)
   and HTTP 400 rejection on invalid reasoning_effort.
2. P0.5: Studio Web & Control Plane Contract Sync: Canonical task aliases (image.t2i, video.t2v)
   resolution without CAPABILITY_NOT_SUPPORTED, /v1/memory dual-T4 telemetry contract, and TypeScript compilation.
3. P0.6: Worker lease auth & strict ownership: Forgery attacks, missing/invalid lease_token,
   mismatched worker_id, unleased mutation attempts, and heartbeat hijacking/expiration attacks rejected with HTTP 403.
"""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from factory_auth.config import AuthConfig
from factory_contracts.errors import UnauthorizedError, InvalidRequestError
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)
from factory.apps.control_api.src.main import app as control_app
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository import InMemoryJobStore
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.metrics import metrics

from services.text.src.app import app as text_app
from services.text.src.engine import StreamingReasoningFilter, TextEngine, decouple_think_tokens
from services.text.src.models import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
)

AUTH_TOKEN = "sk-factory-test-token-gate2-challenger"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


# ==============================================================================
# 1. P0.2 ADVERSARIAL CHALLENGES: Reasoning Token Decoupling & Effort Validation
# ==============================================================================

class TestP02ReasoningAdversarialChallenger:
    @pytest.fixture
    def text_client(self):
        text_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        return TestClient(text_app)

    @pytest.mark.parametrize("invalid_effort", [
        "ultra",
        "extreme",
        "none",
        "default",
        "max",
        "123",
        "LOW_EFFORT",
    ])
    def test_invalid_reasoning_effort_string_rejected_400(self, text_client, invalid_effort):
        """Adversarially verify all invalid reasoning_effort strings trigger HTTP 400 INVALID_REQUEST."""
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Adversarial effort test"}],
            "reasoning_effort": invalid_effort,
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400, f"Expected 400 for reasoning_effort='{invalid_effort}', got {res.status_code}"
        body = res.json()
        assert body["error"]["code"] == "INVALID_REQUEST"

    @pytest.mark.parametrize("invalid_nested", [
        {"effort": "invalid_value"},
        {"effort": "SUPER"},
        {"effort": "extreme"},
    ])
    def test_invalid_nested_reasoning_effort_rejected_400(self, text_client, invalid_nested):
        """Adversarially verify invalid string reasoning={'effort': ...} triggers HTTP 400 INVALID_REQUEST."""
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Adversarial nested effort test"}],
            "reasoning": invalid_nested,
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400, f"Expected 400 for reasoning='{invalid_nested}', got {res.status_code}"
        body = res.json()
        assert body["error"]["code"] == "INVALID_REQUEST"

    @pytest.mark.parametrize("vulnerable_payload", [
        {"reasoning": {"effort": 999}},
        {"reasoning": {"effort": True}},
        {"reasoning_effort": ""},
    ])
    def test_non_string_or_empty_effort_rejected_400_vulnerability(self, text_client, vulnerable_payload):
        """Resolved vulnerability: Non-string reasoning.effort or empty string rejected with HTTP 400 validation."""
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Adversarial bypass test"}],
            **vulnerable_payload,
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400, f"Expected HTTP 400, got status {res.status_code}"

    def test_unclosed_think_with_leading_whitespace_vulnerability(self):
        """Resolved vulnerability: LLMs emitting '\\n<think>' that run out of tokens decouple cleanly."""
        r, c = decouple_think_tokens("\n<think>Deliberating before truncation")
        # Should decouple reasoning and keep clean empty
        assert r == "Deliberating before truncation"
        assert c == ""


    def test_decouple_think_tokens_adversarial_patterns(self):
        """Stress-test decouple_think_tokens against edge cases: empty tags, unclosed tags, nested whitespace."""
        # Case 1: Empty think block
        r, c = decouple_think_tokens("<think></think>Clean answer only")
        assert r is None or r == ""
        assert c == "Clean answer only"

        # Case 2: Whitespace only in think block
        r, c = decouple_think_tokens("<think>   \n  \t </think>Clean answer")
        assert r is None
        assert c == "Clean answer"

        # Case 3: Complete reasoning with newlines
        r, c = decouple_think_tokens("<think>\nStep 1: Check inputs.\nStep 2: Synthesize.\n</think>\nFinal result.")
        assert r == "Step 1: Check inputs.\nStep 2: Synthesize."
        assert c == "Final result."

        # Case 4: Unclosed think tag (model cut off)
        r, c = decouple_think_tokens("<think>Model ran out of tokens mid-thought")
        assert r == "Model ran out of tokens mid-thought"
        assert c == ""

        # Case 5: Normal text without think tag
        r, c = decouple_think_tokens("Pure answer without any tags")
        assert r is None
        assert c == "Pure answer without any tags"

    def test_streaming_reasoning_filter_stress_scenarios(self):
        """Stress-test StreamingReasoningFilter across multi-chunk delta streams."""
        filter_obj = StreamingReasoningFilter()

        # Chunk 1: <think> prefix and first thought
        d1 = filter_obj.process_chunk_delta({"content": "<think>Deliberating step 1"})
        assert d1.get("reasoning_content") == "Deliberating step 1"
        assert "content" not in d1

        # Chunk 2: thought continuation
        d2 = filter_obj.process_chunk_delta({"content": ", continuing thought"})
        assert d2.get("reasoning_content") == ", continuing thought"
        assert "content" not in d2

        # Chunk 3: closing </think> and initial response content
        d3 = filter_obj.process_chunk_delta({"content": " done reasoning</think>Final output begins"})
        assert d3.get("reasoning_content") == " done reasoning"
        assert d3.get("content") == "Final output begins"

        # Chunk 4: subsequent pure response tokens
        d4 = filter_obj.process_chunk_delta({"content": " and completes."})
        assert "reasoning_content" not in d4
        assert d4.get("content") == " and completes."

    def test_streaming_reasoning_filter_single_chunk_both_tags(self):
        """Verify single chunk containing both <think> and </think> is cleanly split."""
        filter_obj = StreamingReasoningFilter()
        d = filter_obj.process_chunk_delta({"content": "<think>Single chunk thought</think>Direct answer"})
        assert d.get("reasoning_content") == "Single chunk thought"
        assert d.get("content") == "Direct answer"

    def test_chat_completions_mock_upstream_decouples_think_tokens(self, text_client):
        """Verify full HTTP chat completion pipeline decouples <think> from content."""
        upstream_mock_response = {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "qwen-2.5-72b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>Pondering life, universe and everything.</think>The answer is 42.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35,
            },
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_mock_response

        with patch("services.text.src.engine.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with patch.dict(os.environ, {"VLLM_TEXT_URL": "http://mock-vllm:8000"}):
                res = text_client.post(
                    "/v1/chat/completions",
                    headers=AUTH_HEADERS,
                    json={
                        "model": "qwen-2.5-72b-instruct",
                        "messages": [{"role": "user", "content": "What is the answer?"}],
                        "reasoning_effort": "medium",
                    },
                )
                assert res.status_code == 200, res.text
                data = res.json()
                choice = data["choices"][0]["message"]
                # Must be decoupled
                assert choice.get("reasoning_content") == "Pondering life, universe and everything."
                assert choice.get("content") == "The answer is 42."
                assert "<think>" not in choice.get("content")
                assert "</think>" not in choice.get("content")


# ==============================================================================
# 2. P0.5 ADVERSARIAL CHALLENGES: Studio Web Contract Sync & Task Aliases
# ==============================================================================

class TestP05StudioSyncAdversarialChallenger:
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

    @pytest.mark.parametrize("task_alias,model,service,expected_canonical_task", [
        ("image.t2i", "qwen-image-2.1", "image", "text_to_image"),
        ("t2i", "qwen-image-2.1", "image", "text_to_image"),
        ("video.t2v", "minimax-h3", "video", "text_to_video"),
        ("t2v", "minimax-h3", "video", "text_to_video"),
        ("image.controlnet", "qwen-image-2.1", "image", "controlnet"),
    ])
    def test_canonical_task_aliases_accepted_without_capability_error(
        self, api_client, task_alias, model, service, expected_canonical_task
    ):
        """Adversarially submit various canonical task aliases from Studio Web; ensure HTTP 201 created."""
        payload = {
            "service": service,
            "model": model,
            "task": task_alias,
            "inputs": {"prompt": "Adversarial alias test"},
        }
        res = api_client.post("/v1/jobs", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 201, f"Failed for task='{task_alias}': {res.text}"
        data = res.json()
        assert data["task"] == expected_canonical_task
        assert data["state"] == "QUEUED"

    def test_memory_dual_t4_telemetry_guaranteed_contract(self, api_client):
        """Assert /v1/memory returns exact dual-T4 GPU telemetry conforming to Studio Web expectation."""
        # Set synthetic VRAM metrics
        metrics.set_gpu_vram(gpu_0_gb=4.5, gpu_1_gb=8.2)

        res = api_client.get("/v1/memory", headers=AUTH_HEADERS)
        assert res.status_code == 200, res.text
        data = res.json()

        assert "gpus" in data
        assert len(data["gpus"]) == 2

        gpu0 = data["gpus"][0]
        assert gpu0["index"] == 0
        assert gpu0["name"] == "Tesla T4 (GPU 0)"
        assert gpu0["hardware_used_gb"] == 4.5
        assert gpu0["total_gb"] == 15.0
        assert gpu0["free_gb"] == 10.5

        gpu1 = data["gpus"][1]
        assert gpu1["index"] == 1
        assert gpu1["name"] == "Tesla T4 (GPU 1)"
        assert gpu1["hardware_used_gb"] == 8.2
        assert gpu1["total_gb"] == 15.0
        assert gpu1["free_gb"] == 6.8

    def test_studio_web_typescript_compilation(self):
        """Empirically execute npx tsc --noEmit in studio-web and verify exit code 0."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        studio_web_dir = repo_root / "factory" / "apps" / "studio-web"
        if not (studio_web_dir / "node_modules").exists() or shutil.which("npx") is None:
            pytest.skip("studio-web/node_modules not installed or npx unavailable; skipping in Python-only environment")
        cmd = ["npx", "tsc", "--noEmit"]
        result = subprocess.run(cmd, cwd=str(studio_web_dir), capture_output=True, text=True)
        assert result.returncode == 0, f"TypeScript build failed:\n{result.stderr}\n{result.stdout}"


# ==============================================================================
# 3. P0.6 ADVERSARIAL CHALLENGES: Worker Lease Auth & Strict Ownership
# ==============================================================================

@pytest.mark.asyncio
class TestP06WorkerLeaseAdversarialChallenger:
    async def _setup(self):
        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)
        return repo, broker, stream_mgr, manager, coordinator

    async def test_forged_lease_token_on_progress_rejected_403(self):
        """Adversarial worker attempts to send progress with forged lease token -> HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Adversarial forgery"},
                ),
                request_id="req-p06-forge",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert lease_resp.lease_token is not None

            # Attacker presents legitimate worker_id but forged lease_token
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=5,
                        total=20,
                        percent=25.0,
                        worker_id="legit-worker-01",
                        lease_token="forged-token-xyz-123",
                    ),
                    caller_worker_id="legit-worker-01",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403
            assert "Invalid or missing lease token" in exc_info.value.message
        finally:
            await repo.close()

    async def test_imposter_worker_id_on_completion_rejected_403(self):
        """Imposter worker attempts to complete a job owned by legitimate worker -> HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Adversarial imposter"},
                ),
                request_id="req-p06-imposter",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            # Imposter worker attempts to complete job using legit token
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="imposter-worker-99",
                        lease_token=lease_resp.lease_token,
                        artifacts=[],
                    ),
                    caller_worker_id="imposter-worker-99",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403
        finally:
            await repo.close()

    async def test_heartbeat_hijack_with_wrong_lease_token_rejected_403(self):
        """Worker heartbeating on active job with incorrect lease_token must be rejected with HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Heartbeat tamper test"},
                ),
                request_id="req-p06-hb-tamper",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            # Heartbeat with wrong lease_token
            with pytest.raises(UnauthorizedError) as exc_info:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="legit-worker-01",
                        active_job_id=job.job_id,
                        lease_token="forged-hb-token",
                    )
                )
            assert exc_info.value.status_code == 403
            assert "Invalid lease token" in exc_info.value.message
        finally:
            await repo.close()

    async def test_heartbeat_on_expired_lease_rejected_403(self):
        """Worker heartbeating on an active job whose lease has expired must be rejected with HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Heartbeat expired test"},
                ),
                request_id="req-p06-hb-expired",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            # Force lease expiration into the past
            expired_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            leased_job = await repo.get_job(job.job_id)
            leased_job.worker.lease_expires_at = expired_time
            await repo.update_job(leased_job)

            # Heartbeat after expiration
            with pytest.raises(UnauthorizedError) as exc_info:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="legit-worker-01",
                        active_job_id=job.job_id,
                        lease_token=lease_resp.lease_token,
                    )
                )
            assert exc_info.value.status_code == 403
            assert "expired" in exc_info.value.message
        finally:
            await repo.close()

    async def test_legitimate_heartbeat_extends_lease(self):
        """Legitimate worker heartbeat extends lease timestamp and updates telemetry."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Legitimate heartbeat"},
                ),
                request_id="req-p06-hb-legit",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            hb_resp = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="legit-worker-01",
                    active_job_id=job.job_id,
                    lease_token=lease_resp.lease_token,
                    vram_used_mb=4096,
                    gpu_utilization_percent=55.0,
                )
            )
            assert hb_resp.status == "acknowledged"
            assert hb_resp.lease_extended_until is not None
            assert hb_resp.lease_extended_until > datetime.now(timezone.utc)
        finally:
            await repo.close()
