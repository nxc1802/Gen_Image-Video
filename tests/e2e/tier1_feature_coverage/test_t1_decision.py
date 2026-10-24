"""
Tier 1 Feature Coverage: Decision Engine (Features 12 - 14)
Validates happy path and contract adherence for:
- Feature 12: Decision Predicate (POST /v1/decisions task=predicate)
- Feature 13: Decision Choice (POST /v1/decisions task=choice)
- Feature 14: Decision Score (POST /v1/decisions task=score)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestDecisionEngine:
    # --- Feature 12: Decision Predicate ---
    def test_f12_decision_predicate_boolean(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": "User prompt requests a video of a blue sports car on a racetrack.",
            "hypothesis": "The request involves vehicular motion."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "predicate"
        assert isinstance(data["decision"], bool)
        assert data["decision"] is True

    def test_f12_decision_predicate_confidence(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": "A serene lake surrounded by pine trees under twilight.",
            "hypothesis": "The scene is situated outdoors in nature."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_f12_decision_predicate_logprobs(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": "The user submitted a Python code snippet requesting unit test generation.",
            "hypothesis": "The task is software development related."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "logprobs" in data
        assert "true" in data["logprobs"]
        assert "false" in data["logprobs"]

    def test_f12_decision_predicate_safety_gate(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": "Standard commercial product photography for an online store catalog.",
            "hypothesis": "The request is safe for public distribution."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["decision"] is True

    def test_f12_decision_predicate_routing(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": "Prompt: '10 second animation of flying through nebulae'",
            "hypothesis": "The user desires a video output rather than a static image."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["confidence"] >= 0.9

    # --- Feature 13: Decision Choice ---
    def test_f13_decision_choice_selection(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "Input contains high-frequency audio noise and hum.",
            "choices": ["audio_enhancement", "video_upscale", "text_generation"]
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "choice"
        assert data["selected_choice"] in payload["choices"]

    def test_f13_decision_choice_probabilities(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "User requests a portrait in oil painting style.",
            "choices": ["image_generation", "music_generation", "decision_score"]
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "probabilities" in data
        probs = data["probabilities"]
        assert len(probs) == len(payload["choices"])
        for p in probs.values():
            assert 0.0 <= p <= 1.0

    def test_f13_decision_choice_four_way_routing(self, client: httpx.Client, auth_headers):
        choices = ["fast_t4", "a100_high_vram", "cpu_worker", "edge_ngrok"]
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "Video model requires 15GB VRAM and heavy FP16 compute.",
            "choices": choices
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["selected_choice"] in choices

    def test_f13_decision_choice_agent_selection(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "Task requires analyzing security vulnerabilities in ngrok traffic policy.",
            "choices": ["SA-SEC", "SA-MEDIA", "SA-MUSIC"]
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert "SA-SEC" in res.json()["probabilities"]

    def test_f13_decision_choice_tool_routing(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "User asks for text translation from Vietnamese to English.",
            "choices": ["vllm_chat", "comfyui_image", "acestep_music"]
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["selected_choice"] == "vllm_chat"

    # --- Feature 14: Decision Score ---
    def test_f14_decision_score_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Generated image has sharp contrast, correct anatomy, and no blur."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "score"
        assert 0.0 <= data["score"] <= 1.0

    def test_f14_decision_score_custom_scale(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Evaluate render fidelity",
            "scale": [10.0, 100.0]
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "score" in data

    def test_f14_decision_score_calibration_metrics(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Calibration test on synthetic quality benchmark."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        calib = res.json()["calibration_metrics"]
        assert "ece" in calib
        assert "brier_score" in calib
        assert calib["ece"] >= 0.0

    def test_f14_decision_score_priority_evaluation(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Critical production incident affecting live streaming playback."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["score"] >= 0.8

    def test_f14_decision_score_quality_gate(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Generated video clip has smooth 60fps frame rate and zero artifacts."
        }
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["score"] > 0.5
