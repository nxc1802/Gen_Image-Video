"""
services/vlm/tests/test_vlm_service.py
Unit and contract tests for the Qwen2-VL Multimodal Vision-Language Daemon.
"""
from unittest.mock import MagicMock, patch

import pytest
from factory_auth.config import AuthConfig
from services.vlm.src.app import app
from services.vlm.src.engine import VLMEngine
from starlette.testclient import TestClient

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}
ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
ONE_FRAME_MP4_DATA_URL = "data:video/mp4;base64,AAAAHGZ0eXBtcDQyAAAAAGlzb21tcDQy"


@pytest.fixture
def vlm_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    app.state.engine = VLMEngine(upstream_url=None)
    return TestClient(app)


def test_vlm_health(vlm_client):
    res = vlm_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "vision-service"


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


def test_vlm_unconfigured_multimodal_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the multimodal input diagram"},
                    {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}},
                ],
            }
        ],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_vlm_unconfigured_video_modality_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze sequential frame consistency"},
                    {"type": "video_url", "video_url": {"url": ONE_FRAME_MP4_DATA_URL}},
                ],
            }
        ],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_vlm_unconfigured_text_only_chat(vlm_client):
    payload = {
        "model": "qwen-vl-max",
        "messages": [{"role": "user", "content": "Evaluate system operational readiness"}],
    }
    res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_vlm_configured_proxy_chat(vlm_client):
    app.state.engine = VLMEngine(upstream_url="http://mock-vllm-vlm:8000")
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the diagram"},
                    {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}},
                ],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "chat-vlm-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen-vl-max",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Detailed visual description of diagram."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = vlm_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "chat.completion"
        assert "diagram" in data["choices"][0]["message"]["content"]
