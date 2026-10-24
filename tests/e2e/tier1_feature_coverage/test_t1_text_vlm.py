"""
Tier 1 Feature Coverage: Text & VLM (Features 10 - 11)
Validates happy path and contract adherence for:
- Feature 10: Text Chat Completions (POST /v1/chat/completions)
- Feature 11: Text Responses API (POST /v1/responses)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestTextAndVLM:
    # --- Feature 10: Text Chat Completions ---
    def test_f10_chat_completions_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Explain the two-tier queue architecture in Factory v4."}]
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_f10_chat_completions_multiturn(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I assist you?"},
                {"role": "user", "content": "What is Factory v4?"}
            ]
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert "Factory v4" in res.json()["choices"][0]["message"]["content"]

    def test_f10_chat_completions_streaming_sse(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Stream a response"}],
            "stream": True
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        text = res.text
        assert "data: " in text
        assert "data: [DONE]" in text

    def test_f10_chat_completions_system_prompt(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "system", "content": "You are a concise AI architecture assistant."},
                {"role": "user", "content": "State your function."}
            ]
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["model"] == "qwen-2.5-72b-instruct"

    def test_f10_chat_completions_usage_tokens(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Check token metrics"}]
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        usage = res.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # --- Feature 11: Text Responses API ---
    def test_f11_text_responses_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "input": "Analyze pipeline failure reasons"
        }
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "response"
        assert "output" in data
        assert data["status"] == "completed"

    def test_f11_text_responses_reasoning_tokens(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "input": "Calculate optimal GPU memory partition"
        }
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "reasoning_tokens" in data
        assert data["reasoning_tokens"] > 0

    def test_f11_text_responses_id_format(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "input": "Verify response ID generation"
        }
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["id"].startswith("resp-")

    def test_f11_text_responses_model_reflection(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-vl-max",
            "input": "Describe the multimodal input diagram"
        }
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["model"] == "qwen-vl-max"

    def test_f11_text_responses_complex_prompt(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "input": "Provide step-by-step mathematical reasoning for batch sizing: B = (VRAM - Model) / Activation."
        }
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert len(res.json()["output"]) > 0
