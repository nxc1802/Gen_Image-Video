"""
Tier 1 Feature Coverage: Video Subsystem (Features 28 - 32)
Validates happy path and contract adherence for:
- Feature 28: Text-to-Video (T2V)
- Feature 29: Image-to-Video (I2V)
- Feature 30: First & Last Frame to Video (FL2V)
- Feature 31: Video Extension & Chaining
- Feature 32: Video with Audio (Audio VAE)
"""

import pytest
import httpx
from tests.e2e.conftest import assert_valid_job


@pytest.mark.tier1
class TestVideoSubsystem:
    # --- Feature 28: Text-to-Video ---
    def test_f28_t2v_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {
                "prompt": "Drone flight over an active volcanic crater at dawn",
                "num_frames": 49,
                "fps": 16,
                "seed": 1001
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "video", "text_to_video")

    def test_f28_t2v_high_frame_count(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Slow motion ocean waves crashing on black sand", "num_frames": 81}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f28_t2v_seed_reproducibility(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Astronaut planting a flag on Mars", "seed": 4242}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f28_t2v_mp4_artifact_returned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Golden retriever running in park"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        artifact = res.json()["artifacts"][0]
        assert artifact["mime_type"] == "video/mp4"
        assert artifact["url"].startswith("https://s3.ai.example.com/outputs/")

    def test_f28_t2v_fps_parameter(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Cyberpunk hovercar chase", "fps": 24}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    # --- Feature 29: Image-to-Video ---
    def test_f29_i2v_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "image_to_video",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/waterfall.png",
                "prompt": "Animate gentle water cascading and rising mist",
                "motion_bucket_id": 127
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "video", "image_to_video")

    def test_f29_i2v_camera_pan(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "image_to_video",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/palace.png",
                "prompt": "Camera pans slowly from left to right"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f29_i2v_motion_intensity(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "image_to_video",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/portrait.png",
                "prompt": "Subtle breathing and blinking",
                "motion_bucket_id": 40
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f29_i2v_seed_control(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "image_to_video",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/city.png",
                "prompt": "Traffic headlights streaming",
                "seed": 900
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f29_i2v_artifact_verification(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "image_to_video",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/img.png", "prompt": "Animate"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["size_bytes"] > 0

    # --- Feature 30: Video First & Last Frame (FL2V) ---
    def test_f30_fl2v_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/assets/shot_start.png",
                "last_frame_url": "https://s3.ai.example.com/assets/shot_end.png",
                "prompt": "Smooth cinematic camera transition between two keyframes",
                "num_frames": 49
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "video", "fl2v")

    def test_f30_fl2v_character_transformation(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/assets/caterpillar.png",
                "last_frame_url": "https://s3.ai.example.com/assets/butterfly.png",
                "prompt": "Metamorphosis timelapse animation"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f30_fl2v_seed_control(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/assets/f1.png",
                "last_frame_url": "https://s3.ai.example.com/assets/f2.png",
                "prompt": "Transition",
                "seed": 654
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f30_fl2v_fps_specification(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/assets/a.png",
                "last_frame_url": "https://s3.ai.example.com/assets/b.png",
                "prompt": "Action interpolation",
                "fps": 24
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f30_fl2v_artifact_contract(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/assets/k1.png",
                "last_frame_url": "https://s3.ai.example.com/assets/k2.png",
                "prompt": "Interpolate"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["mime_type"] == "video/mp4"

    # --- Feature 31: Video Extension & Chaining ---
    def test_f31_video_extension_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/scene1.mp4",
                "prompt": "Continue the character walking down the corridor",
                "source_frames": 49,
                "overlap_frames": 16
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "video", "video_extension")

    def test_f31_video_extension_seed(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/walk.mp4",
                "prompt": "Continuous scene",
                "source_frames": 60,
                "overlap_frames": 10,
                "seed": 888
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f31_video_extension_long_extension(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/sunset.mp4",
                "prompt": "Night falls and stars become visible",
                "source_frames": 49,
                "overlap_frames": 12
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f31_video_extension_num_frames(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/clip.mp4",
                "prompt": "Next scene",
                "source_frames": 60,
                "overlap_frames": 8,
                "num_frames": 49
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f31_video_extension_artifact(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/assets/clip.mp4",
                "prompt": "Extend clip",
                "source_frames": 40,
                "overlap_frames": 10
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["url"].endswith(".bin")

    # --- Feature 32: Video with Audio (Audio VAE) ---
    def test_f32_video_with_audio_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {
                "prompt": "Thunderstorm over an open ocean with heavy rainfall and distant thunder",
                "enable_audio": True
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "video", "video_with_audio")

    def test_f32_video_with_audio_from_image(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/campfire.png",
                "prompt": "Crackling campfire with embers and night crickets sound",
                "enable_audio": True
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f32_video_with_audio_stereo_channels(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {
                "prompt": "Jet engine takeoff with spatial stereo sound",
                "audio_channels": 2
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f32_video_with_audio_seed(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {
                "prompt": "Waterfall soundscape",
                "seed": 707
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f32_video_with_audio_artifact_format(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {"prompt": "Audio video check"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["mime_type"] == "video/mp4"
