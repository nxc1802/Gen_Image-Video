"""
services/decision/tests/test_decision_service.py
Unit and contract tests for the Decision Routing Engine.
"""
from unittest.mock import MagicMock, patch

import pytest
from factory_auth.config import AuthConfig
from services.decision.src.app import app
from services.decision.src.engine import DecisionEngine
from starlette.testclient import TestClient

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def decision_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    app.state.engine = DecisionEngine(upstream_url=None)
    return TestClient(app)


def test_decision_health(decision_client):
    res = decision_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "decision-service"


def test_decision_fail_fast_offline_model(decision_client):
    payload = {
        "model": "offline-decision-model",
        "task": "predicate",
        "context": "c",
        "hypothesis": "h",
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    # Prohibit fake outputs
    assert "decision" not in res.json()


def test_decision_invalid_scale(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "c",
        "scale": [10.0, 5.0],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_decision_empty_choices(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "choice",
        "context": "c",
        "choices": [],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_decision_unconfigured_predicate_fail_closed(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "predicate",
        "context": "User prompt requests a video of a blue sports car on a racetrack.",
        "hypothesis": "The request involves vehicular motion.",
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_decision_unconfigured_choice_fail_closed(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "choice",
        "context": "Input contains high-frequency audio noise and hum.",
        "choices": ["audio_enhancement", "video_upscale", "text_generation"],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_decision_unconfigured_score_fail_closed(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "Generated image has sharp contrast, correct anatomy, and no blur.",
        "scale": [0.0, 1.0],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_decision_configured_predicate_proxy(decision_client):
    app.state.engine = DecisionEngine(upstream_url="http://mock-vllm-decision:8000")
    payload = {
        "model": "qwen-decision-8b",
        "task": "predicate",
        "context": "User prompt requests a video of a blue sports car on a racetrack.",
        "hypothesis": "The request involves vehicular motion.",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "dec-123",
        "model": "qwen-decision-8b",
        "answers": [
            {
                "name": "decision",
                "type": "predicate",
                "decision": True,
                "confidence": 0.98,
                "logprobs": {"true": -0.02, "false": -3.91},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "predicate"
        assert data["answer"]["decision"] is True
        assert data["answer"]["confidence"] == 0.98


def test_decision_configured_choice_proxy(decision_client):
    app.state.engine = DecisionEngine(upstream_url="http://mock-vllm-decision:8000")
    payload = {
        "model": "qwen-decision-8b",
        "task": "choice",
        "context": "Deploying web services and managing ingress traffic routing.",
        "choices": ["networking", "image_processing", "audio_synthesis", "database_storage"],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "dec-choice-123",
        "model": "qwen-decision-8b",
        "answers": [
            {
                "name": "decision",
                "type": "choice",
                "selected_choice": "networking",
                "probabilities": {
                    "networking": 0.85,
                    "image_processing": 0.05,
                    "audio_synthesis": 0.05,
                    "database_storage": 0.05,
                },
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "choice"
        assert data["answer"]["selected_choice"] == "networking"


def test_decision_configured_score_proxy(decision_client):
    app.state.engine = DecisionEngine(upstream_url="http://mock-vllm-decision:8000")
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "Production performance metrics indicate stable throughput and low latency.",
        "scale": [50.0, 150.0],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "dec-score-123",
        "model": "qwen-decision-8b",
        "answers": [
            {
                "name": "decision",
                "type": "score",
                "score": 135.0,
                "levels": [{"label": "135.0", "probability": 0.9}],
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "score"
        assert data["answer"]["score"] == 135.0
