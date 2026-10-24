"""
services/vlm/tests/test_vlm_service.py
Unit and contract tests for the Qwen2-VL Multimodal Vision-Language Daemon.
"""
import pytest
from starlette.testclient import TestClient
from factory_auth.config import AuthConfig
from services.vlm.src.app import app

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def vlm_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    return TestClient(app)


def test_vlm_health(vlm_client):
    res = vlm_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "vision-service"


def test_vlm_multimodal_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the multimodal input diagram"},
                    {"type": "image_url", "image_url": {"url": "https://s3.ai.example.com/assets/arch.png"}},
                ],
            }
        ],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert "diagram" in data["choices"][0]["message"]["content"].lower() or "factory v4" in data["choices"][0]["message"]["content"].lower()


def test_vlm_ssrf_blocked(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Access internal metadata"},
                    {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data"}},
                ],
            }
        ],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "SSRF_BLOCKED"


def test_vlm_offline_fail_fast(vlm_client):
    payload = {
        "model": "offline-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_vlm_video_modality_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze sequential frame consistency"},
                    {"type": "video_url", "video_url": {"url": "https://s3.ai.example.com/assets/sample_clip.mp4"}},
                ],
            }
        ],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    content = res.json()["choices"][0]["message"]["content"]
    assert "video" in content.lower()
    assert "temporal" in content.lower() or "frame" in content.lower()


def test_vlm_streaming_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Stream multimodal explanation"},
                    {"type": "image_url", "image_url": {"url": "https://s3.ai.example.com/assets/diagram.png"}},
                ],
            }
        ],
        "stream": True,
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    assert "data: " in res.text
    assert "data: [DONE]" in res.text


def test_vlm_text_only_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [{"role": "user", "content": "Evaluate system operational readiness"}],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    assert "Qwen2-VL" in res.json()["choices"][0]["message"]["content"]
