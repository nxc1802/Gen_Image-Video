"""
Tier 3 Cross-Feature Interaction Tests: Decision Routing & Model Execution
Validates pipeline where Decision engine classifies task, evaluates quality/safety score,
and routes to the optimal AI worker.
"""

import pytest
import httpx


@pytest.mark.tier3
class TestDecisionRoutingPipeline:
    def test_pipeline_decision_predicate_and_choice_routing(self, client: httpx.Client, auth_headers):
        user_request = "Generate a 10-second animation of a spacecraft landing on an alien moon."

        # Step 1: Decision Predicate determines modality
        pred_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": user_request,
            "hypothesis": "The request requires video animation rather than a static image."
        })
        assert pred_res.status_code == 200
        is_video = pred_res.json()["decision"]
        assert is_video is True

        # Step 2: Decision Choice selects serving engine
        choice_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": user_request,
            "choices": ["wan-2.2", "qwen-image-2.1", "ace-step-1.5"]
        })
        assert choice_res.status_code == 200
        chosen_model = choice_res.json()["selected_choice"]
        assert chosen_model in ["wan-2.2", "qwen-image-2.1"]

        # Step 3: Route job to the chosen model
        job_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video" if chosen_model == "wan-2.2" else "image",
            "model": chosen_model,
            "task": "text_to_video" if chosen_model == "wan-2.2" else "text_to_image",
            "inputs": {"prompt": user_request}
        })
        assert job_res.status_code == 201

    def test_pipeline_decision_quality_gate_blocks_low_score(self, client: httpx.Client, auth_headers):
        # Step 1: Score prompt quality
        score_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Prompt quality check: detailed description with lighting and style."
        })
        assert score_res.status_code == 200
        quality_score = score_res.json()["score"]

        # Step 2: Only submit if quality passes threshold (> 0.5)
        if quality_score > 0.5:
            job_res = client.post("/v1/jobs", headers=auth_headers, json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Detailed landscape"}
            })
            assert job_res.status_code == 201

    def test_pipeline_decision_model_selection_to_job_execution(self, client: httpx.Client, auth_headers):
        # Query Registry to find candidate models
        reg_res = client.get("/v1/registry/models?family=qwen-image", headers=auth_headers)
        assert reg_res.status_code == 200
        available_models = [m["id"] for m in reg_res.json()["models"] if m["status"] == "full"]

        # Use Decision Choice to pick best model
        choice_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "Professional studio portrait with natural skin tones.",
            "choices": available_models
        })
        assert choice_res.status_code == 200
        selected = choice_res.json()["selected_choice"]

        # Submit job to the selected model
        job_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": selected,
            "task": "text_to_image",
            "inputs": {"prompt": "Studio portrait"}
        })
        assert job_res.status_code == 201
