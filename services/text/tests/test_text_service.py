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
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) >= 1
    assert "Factory v4" in data["choices"][0]["message"]["content"]
    assert data["usage"]["total_tokens"] > 0


def test_text_chat_completions_streaming(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "Stream this response"}],
        "stream": True,
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "data: " in text
    assert "data: [DONE]" in text


def test_text_responses_api(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "input": "Calculate optimal GPU memory partition",
    }
    res = text_client.post("/v1/responses", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "response"
    assert data["id"].startswith("resp-")
    assert data["reasoning_tokens"] > 0
    assert data["status"] == "completed"


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
        "messages": [{"role": "user", "content": "A" * 150000}],
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_text_chat_completions_system_persona(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [
            {"role": "system", "content": "You are an autonomous distributed systems architect."},
            {"role": "user", "content": "What are your primary capabilities?"},
        ],
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    content = res.json()["choices"][0]["message"]["content"]
    assert "autonomous distributed systems architect" in content.lower()


def test_text_chat_completions_json_response_format(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": "Analyze node latency metrics"}],
        "response_format": {"type": "json_object"},
    }
    res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    content = res.json()["choices"][0]["message"]["content"]
    import json
    data = json.loads(content)
    assert data["status"] == "success"
    assert data["model"] == "qwen-2.5-72b-instruct"
    assert "response" in data


def test_text_responses_dynamic_reasoning_decomposition(text_client):
    payload = {
        "model": "qwen-2.5-72b-instruct",
        "input": "Calculate cluster memory headroom where Headroom = (Total_RAM - OS_Reserve) / Worker_Nodes",
    }
    res = text_client.post("/v1/responses", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "response"
    assert "Reasoning Step 1" in data["output"]
    assert "Reasoning Step 2" in data["output"]
    assert "Reasoning Step 3" in data["output"]
    assert data["reasoning_tokens"] > 10
    assert data["status"] == "completed"
