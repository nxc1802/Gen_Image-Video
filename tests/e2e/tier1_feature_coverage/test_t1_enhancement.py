"""
Tier 1 Feature Coverage: Enhancement Subsystem (Features 36 - 39)
Validates happy path and contract adherence for:
- Feature 36: Image Super-Resolution
- Feature 37: Video Super-Resolution
- Feature 38: Frame Interpolation
- Feature 39: Audio Enhancement
"""

import pytest
import httpx
from tests.e2e.conftest import assert_valid_job


@pytest.mark.tier1
class TestEnhancementSubsystem:
    # --- Feature 36: Image Super-Resolution ---
    def test_f36_image_super_resolution_2x(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/lowres.png",
                "scale_factor": 2
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "enhancement", "image_super_resolution")

    def test_f36_image_super_resolution_4x(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/render512.png",
                "scale_factor": 4
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f36_image_super_resolution_artifact(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/input.png", "scale_factor": 2}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        art = res.json()["artifacts"][0]
        assert art["mime_type"] == "image/png"
        assert art["url"].startswith("https://s3.ai.example.com/")

    def test_f36_image_super_resolution_denoise_option(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/grainy.png",
                "scale_factor": 2,
                "pre_denoise": True
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f36_image_super_resolution_state_queued(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/img.png"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["state"] == "QUEUED"

    # --- Feature 37: Video Super-Resolution ---
    def test_f37_video_super_resolution_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/720p_clip.mp4",
                "scale_factor": 2
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "enhancement", "video_super_resolution")

    def test_f37_video_super_resolution_artifact_mime(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/clip.mp4"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["mime_type"] == "image/png" or res.json()["artifacts"][0]["url"].endswith(".bin")

    def test_f37_video_super_resolution_priority(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/sample.mp4"},
            "priority": 5
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert res.json()["priority"] == 5

    def test_f37_video_super_resolution_retained_in_status(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/sample.mp4"}
        }
        sub = client.post("/v1/jobs", headers=auth_headers, json=payload)
        job_id = sub.json()["job_id"]
        status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status.status_code == 200
        assert status.json()["task"] == "video_super_resolution"

    def test_f37_video_super_resolution_idempotency(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/sample.mp4"},
            "idempotency_key": "video-upscale-idem-1"
        }
        r1 = client.post("/v1/jobs", headers=auth_headers, json=payload)
        r2 = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert r1.json()["job_id"] == r2.json()["job_id"]

    # --- Feature 38: Frame Interpolation ---
    def test_f38_frame_interpolation_2x(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/30fps_video.mp4",
                "multiplier": 2
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "enhancement", "frame_interpolation")

    def test_f38_frame_interpolation_4x(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/24fps_cinematic.mp4",
                "multiplier": 4
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f38_frame_interpolation_target_fps(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/clip.mp4",
                "target_fps": 60
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f38_frame_interpolation_artifact_url(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/clip.mp4"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["url"].startswith("https://s3.ai.example.com/")

    def test_f38_frame_interpolation_worker_assigned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/clip.mp4"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert "worker_id" in res.json()["worker"]

    # --- Feature 39: Audio Enhancement ---
    def test_f39_audio_enhancement_denoise(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/noisy_podcast.wav",
                "mode": "denoise"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "enhancement", "audio_enhancement")

    def test_f39_audio_enhancement_voice_clarity(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/muffled_dialogue.wav",
                "mode": "voice_clarity"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f39_audio_enhancement_level_balance(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/interview.wav",
                "auto_gain_control": True
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f39_audio_enhancement_artifact_returned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {"audio_url": "https://s3.ai.example.com/assets/audio.wav"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        art = res.json()["artifacts"][0]
        assert art["size_bytes"] > 0

    def test_f39_audio_enhancement_sse_events(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {"audio_url": "https://s3.ai.example.com/assets/audio.wav"}
        }
        sub = client.post("/v1/jobs", headers=auth_headers, json=payload)
        job_id = sub.json()["job_id"]
        events = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert "event: completed" in events.text
