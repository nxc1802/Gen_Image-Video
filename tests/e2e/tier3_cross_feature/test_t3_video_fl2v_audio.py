"""
Tier 3 Cross-Feature Interaction Tests: Video FL2V Keyframe Interpolation & Audio VAE Muxing
Validates multi-step video creation:
1. Image generation produces First Frame
2. Instruction Edit produces Last Frame
3. FL2V interpolates the intermediate scene
4. Audio VAE adds synchronized stereo sound
5. RIFE doubles the final frame rate
"""

import pytest
import httpx


@pytest.mark.tier3
class TestVideoFL2VAudioPipeline:
    def test_pipeline_image_pair_to_fl2v_video(self, client: httpx.Client, auth_headers):
        # Step 1: Generate Keyframe 1 (Start)
        k1_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Spacecraft on launchpad at dawn"}
        })
        k1_url = k1_res.json()["artifacts"][0]["url"]

        # Step 2: Generate Keyframe 2 (End)
        k2_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Spacecraft breaking through Earth upper atmosphere into orbit"}
        })
        k2_url = k2_res.json()["artifacts"][0]["url"]

        # Step 3: Run FL2V video interpolation
        fl2v_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": k1_url,
                "last_frame_url": k2_url,
                "prompt": "Rocket engines ignite, thrusting the spacecraft skyward"
            }
        })
        assert fl2v_res.status_code == 201
        video_url = fl2v_res.json()["artifacts"][0]["url"]
        assert video_url.startswith("https://s3.ai.example.com/")

    def test_pipeline_fl2v_to_audio_vae_mux(self, client: httpx.Client, auth_headers):
        # Submit FL2V with native audio VAE enabled
        job_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "video_with_audio",
            "inputs": {
                "prompt": "Volcanic eruption with thunderous rumbling and shockwaves",
                "enable_audio": True,
                "audio_channels": 2
            }
        })
        assert job_res.status_code == 201
        art = job_res.json()["artifacts"][0]
        assert art["mime_type"] == "video/mp4"

    def test_pipeline_generated_video_to_frame_interpolation(self, client: httpx.Client, auth_headers):
        # Step 1: Base video generation
        vid_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "High speed train crossing suspension bridge", "fps": 24}
        })
        base_video_url = vid_res.json()["artifacts"][0]["url"]

        # Step 2: Double frame rate using RIFE Frame Interpolation (24fps -> 48fps)
        interp_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {"video_url": base_video_url, "multiplier": 2}
        })
        assert interp_res.status_code == 201
        interp_job = interp_res.json()
        assert interp_job["service"] == "enhancement"
        assert interp_job["task"] == "frame_interpolation"

    def test_pipeline_video_super_resolution_after_interpolation(self, client: httpx.Client, auth_headers):
        # Frame interpolation
        interp_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "frame_interpolation",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/clip.mp4", "multiplier": 2}
        })
        interp_url = interp_res.json()["artifacts"][0]["url"]

        # Upscale video
        upscale_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "video_super_resolution",
            "inputs": {"video_url": interp_url, "scale_factor": 2}
        })
        assert upscale_res.status_code == 201
        assert upscale_res.json()["state"] == "QUEUED"

    def test_pipeline_video_extension_chain(self, client: httpx.Client, auth_headers):
        # Clip 1
        clip1_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Robot starts running"}
        })
        clip1_url = clip1_res.json()["artifacts"][0]["url"]

        # Extend Clip 1 -> Clip 2
        clip2_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {"video_url": clip1_url, "prompt": "Robot jumps over hurdle", "source_frames": 49, "overlap_frames": 12}
        })
        assert clip2_res.status_code == 201
