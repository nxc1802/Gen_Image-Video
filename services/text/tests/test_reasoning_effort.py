"""
services/text/tests/test_reasoning_effort.py
Comprehensive unit tests for OpenAI-compatible reasoning_effort and <think> token isolation.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from factory_auth.config import AuthConfig
from services.text.src.app import app
from services.text.src.engine import StreamingReasoningFilter, TextEngine, decouple_think_tokens
from services.text.src.models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionTokensDetails,
    UsageInfo,
)
from services.vlm.src.engine import VLMEngine

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def text_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    return TestClient(app)


def test_reasoning_effort_valid_values():
    """Verify reasoning_effort accepts 'low', 'medium', 'high' case-insensitively."""
    for val in ["low", "medium", "high", "LOW", "Medium", "HIGH"]:
        req = ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content="Hello")],
            reasoning_effort=val,
        )
        assert req.reasoning_effort == val.lower()


def test_reasoning_nested_effort_alias():
    """Verify reasoning={'effort': ...} alias resolves into canonical reasoning_effort."""
    req = ChatCompletionRequest(
        model="qwen-2.5-72b-instruct",
        messages=[ChatMessage(role="user", content="Hello")],
        reasoning={"effort": "medium"},
    )
    assert req.reasoning_effort == "medium"

    req_low = ChatCompletionRequest(
        model="qwen-2.5-72b-instruct",
        messages=[ChatMessage(role="user", content="Hello")],
        reasoning={"effort": "LOW"},
    )
    assert req_low.reasoning_effort == "low"


def test_reasoning_effort_invalid_rejected_400(text_client):
    """Verify invalid reasoning_effort is rejected with HTTP 400 INVALID_REQUEST."""
    # 1. Pydantic schema validation raises ValueError
    with pytest.raises(Exception) as exc_info:
        ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content="Hello")],
            reasoning_effort="ultra",
        )
    assert "reasoning_effort" in str(exc_info.value)

    # 2. HTTP endpoint returns HTTP 400 INVALID_REQUEST
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "ultra",
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_decouple_think_tokens_helper():
    """Verify decouple_think_tokens properly handles complete, choked, and tag-free content."""
    # Complete thinking
    raw_complete = "<think>Analyze problem</think>Here is the solution."
    reasoning, clean = decouple_think_tokens(raw_complete)
    assert reasoning == "Analyze problem"
    assert clean == "Here is the solution."

    # Choked thinking
    raw_choked = "<think>Still thinking and ran out of tokens"
    reasoning, clean = decouple_think_tokens(raw_choked)
    assert reasoning == "Still thinking and ran out of tokens"
    assert clean == ""

    # No thinking
    raw_none = "Direct answer without reasoning."
    reasoning, clean = decouple_think_tokens(raw_none)
    assert reasoning is None
    assert clean == "Direct answer without reasoning."


def test_think_tokens_isolated_non_streaming(text_client):
    """Verify non-streaming endpoint decouples <think> into reasoning_content."""
    mock_resp = {
        "id": "cmpl-reasoning-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>\nStep 1: Compute optimal bounds.\nStep 2: Format output.\n</think>\nThe final answer is 42.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 30,
            "total_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }

    with patch("services.text.src.engine.httpx.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp)
        mock_client_cls.return_value = mock_instance

        app.state.engine = TextEngine(upstream_url="http://mock-vllm-text:8000")
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "reasoning_effort": "high",
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        msg = res.json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "Step 1: Compute optimal bounds.\nStep 2: Format output."
        assert msg["content"] == "The final answer is 42."
        assert "<think>" not in msg["content"]
        assert "</think>" not in msg["content"]


@pytest.mark.asyncio
async def test_think_tokens_isolated_streaming():
    """Verify streaming chunks emit reasoning_content deltas during thinking and content after."""
    engine = TextEngine(upstream_url="http://mock-vllm-text:8000")
    req = ChatCompletionRequest(
        model="qwen-2.5-72b-instruct",
        messages=[ChatMessage(role="user", content="Stream reasoning")],
        stream=True,
    )

    class MockStreamResponse:
        def __init__(self, lines):
            self.status_code = 200
            self._lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class MockAsyncStreamClient:
        def __init__(self, lines):
            self._lines = lines

        def __call__(self, *args, **kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, *args, **kwargs):
            return MockStreamResponse(self._lines)

    mock_lines = [
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"content": "<think>Thinking through step 1"}}]}',
        'data: {"choices": [{"delta": {"content": " and step 2</think>Here is the"}}]}',
        'data: {"choices": [{"delta": {"content": " answer."}}]}',
        'data: [DONE]',
    ]

    with patch("services.text.src.engine.httpx.AsyncClient", MockAsyncStreamClient(mock_lines)):
        chunks = [c async for c in engine.stream_generate(req)]
        assert len(chunks) == 5

        # Chunk 0: role
        p0 = json.loads(chunks[0][6:].strip())
        assert p0["choices"][0]["delta"].get("role") == "assistant"

        # Chunk 1: reasoning_content
        p1 = json.loads(chunks[1][6:].strip())
        assert p1["choices"][0]["delta"].get("reasoning_content") == "Thinking through step 1"
        assert "content" not in p1["choices"][0]["delta"]

        # Chunk 2: transition chunk (has both reasoning_content and content)
        p2 = json.loads(chunks[2][6:].strip())
        assert p2["choices"][0]["delta"].get("reasoning_content") == " and step 2"
        assert p2["choices"][0]["delta"].get("content") == "Here is the"

        # Chunk 3: content only
        p3 = json.loads(chunks[3][6:].strip())
        assert p3["choices"][0]["delta"].get("content") == " answer."
        assert "reasoning_content" not in p3["choices"][0]["delta"]

        # Chunk 4: [DONE]
        assert chunks[4].strip() == "data: [DONE]"


def test_completion_tokens_details_schema():
    """Verify UsageInfo schema supports completion_tokens_details."""
    usage = UsageInfo(
        prompt_tokens=15,
        completion_tokens=45,
        total_tokens=60,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=25),
    )
    dumped = usage.model_dump()
    assert dumped["completion_tokens_details"]["reasoning_tokens"] == 25


def test_vlm_engine_reasoning_validation_and_decoupling():
    """Verify VLMEngine validates reasoning_effort and decouples think tokens."""
    engine = VLMEngine()

    # Valid effort
    valid_req = ChatCompletionRequest(
        model="qwen-vl-max",
        messages=[ChatMessage(role="user", content="Describe image")],
        reasoning_effort="high",
    )
    engine.validate_request(valid_req)
    assert valid_req.reasoning_effort == "high"

    # Non-streaming decoupling in VLM
    mock_resp = {
        "id": "cmpl-vlm-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>Analyzing image visual features</think>A cat resting in sunlight.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    with patch("services.vlm.src.engine.httpx.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp)
        mock_client_cls.return_value = mock_instance

        vlm_engine = VLMEngine(upstream_url="http://mock-vllm-vlm:8000")
        res = vlm_engine.generate(valid_req)
        choice = res["choices"][0]["message"]
        assert choice["reasoning_content"] == "Analyzing image visual features"
        assert choice["content"] == "A cat resting in sunlight."
