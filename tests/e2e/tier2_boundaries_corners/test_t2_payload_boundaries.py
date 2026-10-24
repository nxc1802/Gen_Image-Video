"""
Tier 2 Boundary Value Analysis (BVA) & Payload Limits Tests
Validates extreme boundary conditions, limits enforcement, zero and negative inputs,
upper threshold cutoffs, and empty payload handling.
"""

import pytest
import httpx
from tests.e2e.conftest import assert_sanitized_error


@pytest.mark.tier2
class TestPayloadBoundaries:
    # --- Image Dimensions Boundary Analysis (Valid: 64 to 2048) ---
    @pytest.mark.parametrize("bad_width", [0, -1, -500, 63, 2049, 4096, 8192])
    def test_image_width_boundary_rejected(self, client: httpx.Client, auth_headers, bad_width):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Boundary test", "width": bad_width, "height": 1024}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    @pytest.mark.parametrize("bad_height", [0, -1, -500, 63, 2049, 4096, 8192])
    def test_image_height_boundary_rejected(self, client: httpx.Client, auth_headers, bad_height):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Boundary test", "width": 1024, "height": bad_height}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Video Frames Boundary Analysis (Valid: 1 to 161) ---
    @pytest.mark.parametrize("bad_frames", [0, -1, -49, 162, 300, 1000])
    def test_video_num_frames_boundary_rejected(self, client: httpx.Client, auth_headers, bad_frames):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Frame boundary test", "num_frames": bad_frames}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Music Duration Boundary Analysis (Valid: 1 to 300s) ---
    @pytest.mark.parametrize("bad_duration", [0, -1, -60, 301, 600, 3600])
    def test_music_duration_boundary_rejected(self, client: httpx.Client, auth_headers, bad_duration):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Duration boundary test", "duration_seconds": bad_duration}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Negative Seeds (Valid: seed >= 0) ---
    @pytest.mark.parametrize("bad_seed", [-1, -42, -999999])
    def test_negative_seed_rejected(self, client: httpx.Client, auth_headers, bad_seed):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Seed test", "seed": bad_seed}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Text / Context Window Upper Boundary (>128k characters) ---
    def test_text_context_window_overflow(self, client: httpx.Client, auth_headers):
        oversized_content = "Factory v4 " * 15000  # >150k characters
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": oversized_content}]
        }
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Empty Payloads Across Subsystems ---
    def test_chat_empty_messages(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-2.5-72b-instruct", "messages": []}
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_text_responses_empty_input(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-2.5-72b-instruct", "input": ""}
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_tts_empty_input(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen3-tts", "input": ""}
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_embeddings_empty_input(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-embedding-8b", "input": ""}
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_rerank_empty_query(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-rerank-8b", "query": "", "documents": ["doc"]}
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_rerank_empty_documents(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-rerank-8b", "query": "hello", "documents": []}
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # --- Valid Extremes (BVA Acceptance Checks) ---
    def test_valid_image_lower_bound_64x64(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Mini icon", "width": 64, "height": 64}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_valid_image_upper_bound_2048x2048(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Max HD landscape", "width": 2048, "height": 2048}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_valid_video_frames_upper_bound_161(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Max frame video clip", "num_frames": 161}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_valid_music_duration_upper_bound_300(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Full 5 minute track", "duration_seconds": 300}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_valid_zero_seed(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Zero seed", "seed": 0}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    # Additional boundary test cases to complete BVA matrix
    def test_negative_priority(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Negative priority check"}, "priority": -1
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201  # Priority can accept negative or default gracefully

    def test_jobs_missing_service_field(self, client: httpx.Client, auth_headers):
        payload = {"model": "qwen-image-2.1", "task": "text_to_image", "inputs": {}}
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_jobs_missing_model_field(self, client: httpx.Client, auth_headers):
        payload = {"service": "image", "task": "text_to_image", "inputs": {}}
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_jobs_missing_task_field(self, client: httpx.Client, auth_headers):
        payload = {"service": "image", "model": "qwen-image-2.1", "inputs": {}}
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_jobs_missing_inputs_field(self, client: httpx.Client, auth_headers):
        payload = {"service": "image", "model": "qwen-image-2.1", "task": "text_to_image"}
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")
