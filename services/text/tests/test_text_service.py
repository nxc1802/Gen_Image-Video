"""
services/text/tests/test_text_service.py
Unit and contract tests for the Text Generation FastAPI Daemon.
"""
import pytest
from starlette.testclient import TestClient
from factory_auth.config import AuthConfig
from services.text.src.app import app

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def text_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    return TestClient(app)


def test_text_health(text_client):
    res = text_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "text-service"


def test_text_unauthorized(text_client):
    res = text_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer bad_token"},
        json={"model": "qwen-2.5-72b-instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "UNAUTHORIZED"


def test_text_chat_completions_standard(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "Explain the two-tier queue architecture in Factory v4."}],
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_text_chat_completions_streaming(text_client):
    from services.text.src.models import ChatCompletionRequest, ChatMessage
    from services.text.src.engine import TextEngine
    engine = TextEngine()
    req = ChatCompletionRequest(
        model="qwen-2.5-72b-instruct",
        messages=[ChatMessage(role="user", content="Stream this response")],
        stream=True,
    )
    with pytest.raises(Exception) as exc_info:
        import asyncio
        async def run():
            async for _ in engine.stream_generate(req):
                pass
        asyncio.run(run())
    assert "Native vLLM text backend is not configured" in str(exc_info.value)


def test_text_responses_api(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "input": "Calculate optimal GPU memory partition",
    }
    res = text_client.post("/v1/responses", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_text_offline_model_fail_fast(text_client):
    payload = {
        "model": "offline-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_text_empty_messages_rejected(text_client):
    payload = {"model": "qwen-2.5-72b-instruct", "messages": []}
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_text_context_overflow_rejected(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "A" * 600000}],
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_text_chat_completions_configured_proxy(text_client):
    from unittest.mock import patch, MagicMock
    from services.text.src.engine import TextEngine
    mock_resp = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Factory v4 is fully decoupled."}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 15},
    }
    with patch("services.text.src.engine.httpx.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(status_code=200, json=lambda: mock_resp)
        mock_client_cls.return_value = mock_instance

        app.state.engine = TextEngine(upstream_url="http://mock-vllm-text:8000")
        payload = {"model": "qwen-2.5-72b-instruct", "messages": [{"role": "user", "content": "Explain Factory v4"}]}
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        assert res.json()["choices"][0]["message"]["content"] == "Factory v4 is fully decoupled."
