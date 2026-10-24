"""
services/decision/tests/test_decision_service.py
Unit and contract tests for the Decision Routing Engine.
"""
import pytest
from starlette.testclient import TestClient
from factory_auth.config import AuthConfig
from services.decision.src.app import app

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def decision_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    return TestClient(app)


def test_decision_health(decision_client):
    res = decision_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "decision-service"


def test_decision_predicate_evaluation(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "predicate",
        "context": "User prompt requests a video of a blue sports car on a racetrack.",
        "hypothesis": "The request involves vehicular motion.",
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "predicate"
    assert data["decision"] is True
    assert 0.0 <= data["confidence"] <= 1.0
    assert "true" in data["logprobs"]
    assert "false" in data["logprobs"]


def test_decision_choice_routing(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "choice",
        "context": "Input contains high-frequency audio noise and hum.",
        "choices": ["audio_enhancement", "video_upscale", "text_generation"],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "choice"
    assert data["selected_choice"] == "audio_enhancement"
    assert len(data["probabilities"]) == 3
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01


def test_decision_score_continuous_and_calibrated(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "Generated image has sharp contrast, correct anatomy, and no blur.",
        "scale": [0.0, 1.0],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "score"
    assert 0.0 <= data["score"] <= 1.0
    calib = data["calibration_metrics"]
    assert "ece" in calib
    assert "brier_score" in calib
    assert calib["ece"] >= 0.0


def test_decision_custom_scale(decision_client):
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "Priority incident on production streaming cluster.",
        "scale": [10.0, 100.0],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    score = res.json()["score"]
    assert 10.0 <= score <= 100.0


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


def test_decision_predicate_mathematical_invariants(decision_client):
    """Verifies that predicate logprobs and probabilities sum to 1.0 ± 0.01 and confidence >= 0.5."""
    import math
    payload = {
        "model": "qwen-decision-8b",
        "task": "predicate",
        "context": "The system encountered network timeouts during high concurrency test runs.",
        "hypothesis": "The issue involves latency or connection timeout problems.",
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] is True
    assert data["confidence"] >= 0.5

    lp_true = data["logprobs"]["true"]
    lp_false = data["logprobs"]["false"]
    # Verify exp(lp_true) + exp(lp_false) == 1.0 ± 0.01
    sum_exp = math.exp(lp_true) + math.exp(lp_false)
    assert abs(sum_exp - 1.0) < 0.01


def test_decision_choice_mathematical_invariants(decision_client):
    """Verifies Softmax distribution sums to 1.0 ± 1e-5, all p_i > 0, and selected_choice == max(probs)."""
    payload = {
        "model": "qwen-decision-8b",
        "task": "choice",
        "context": "Deploying web services and managing ingress traffic routing.",
        "choices": ["networking", "image_processing", "audio_synthesis", "database_storage"],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    probs = data["probabilities"]

    assert abs(sum(probs.values()) - 1.0) < 1e-5
    for p in probs.values():
        assert 0.0 < p <= 1.0

    best_choice = max(probs.keys(), key=lambda k: probs[k])
    assert data["selected_choice"] == best_choice
    assert data["selected_choice"] == "networking"


def test_decision_score_mathematical_invariants(decision_client):
    """Verifies continuous score strictly within [scale[0], scale[1]] and calibration metrics in [0.0, 1.0]."""
    payload = {
        "model": "qwen-decision-8b",
        "task": "score",
        "context": "Production performance metrics indicate stable throughput and low latency.",
        "scale": [50.0, 150.0],
    }
    res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    score = data["score"]
    assert 50.0 <= score <= 150.0

    calib = data["calibration_metrics"]
    assert 0.0 <= calib["ece"] <= 1.0
    assert 0.0 <= calib["brier_score"] <= 1.0
