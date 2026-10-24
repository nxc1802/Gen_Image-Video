"""
Tier 3 Cross-Feature Interaction Tests: Text -> Image -> Enhancement Pipeline
Validates end-to-end data flow where LLM generates prompt concepts,
Image subsystem generates high-fidelity visuals, and Enhancement upscales the output.
"""

import pytest
import httpx


@pytest.mark.tier3
class TestTextImageEnhancementPipeline:
    def test_pipeline_text_to_image_to_super_res_2x(self, client: httpx.Client, auth_headers):
        # Step 1: Text LLM crafts the prompt
        chat_res = client.post("/v1/chat/completions", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "messages": [{"role": "user", "content": "Write a 1-sentence prompt for a photorealistic architectural glass pavilion."}]
        })
        assert chat_res.status_code == 200
        prompt = chat_res.json()["choices"][0]["message"]["content"]

        # Step 2: Submit Image T2I job
        img_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": prompt, "width": 1024, "height": 1024}
        })
        assert img_res.status_code == 201
        img_job = img_res.json()
        raw_artifact_url = img_job["artifacts"][0]["url"]
        assert raw_artifact_url.startswith("https://s3.ai.example.com/")

        # Step 3: Upscale 2x via Enhancement subsystem
        enhance_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": raw_artifact_url, "scale_factor": 2}
        })
        assert enhance_res.status_code == 201
        enhanced_job = enhance_res.json()
        assert enhanced_job["service"] == "enhancement"
        assert enhanced_job["task"] == "image_super_resolution"
        assert len(enhanced_job["artifacts"]) >= 1

    def test_pipeline_text_to_image_to_super_res_4x(self, client: httpx.Client, auth_headers):
        # Generate image
        img_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A medieval fantasy castle surrounded by waterfalls", "width": 1024, "height": 1024}
        })
        artifact_url = img_res.json()["artifacts"][0]["url"]

        # 4x Upscale
        enhance_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": artifact_url, "scale_factor": 4}
        })
        assert enhance_res.status_code == 201
        assert enhance_res.json()["state"] == "QUEUED"

    def test_pipeline_text_prompt_refinement_to_i2i(self, client: httpx.Client, auth_headers):
        # Step 1: LLM refines edit instructions
        resp = client.post("/v1/responses", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "input": "Suggest style modification for converting a daytime landscape to cyberpunk night"
        })
        instruction = resp.json()["output"]

        # Step 2: Ingest base image
        asset_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "image/png"})
        base_s3_url = asset_res.json()["s3_url"]

        # Step 3: Run Image-to-Image with the refined instruction
        i2i_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {"image_url": base_s3_url, "prompt": instruction, "denoise_strength": 0.7}
        })
        assert i2i_res.status_code == 201
        assert i2i_res.json()["task"] == "image_to_image"

    def test_pipeline_text_concept_to_controlnet_enhancement(self, client: httpx.Client, auth_headers):
        # Upload pose skeleton
        asset_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "image/png"})
        pose_url = asset_res.json()["s3_url"]

        # Generate via ControlNet
        cnet_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Martial arts master performing flying kick",
                "control_type": "openpose",
                "control_image_url": pose_url
            }
        })
        assert cnet_res.status_code == 201
        cnet_artifact = cnet_res.json()["artifacts"][0]["url"]

        # Super-resolve generated character
        enh_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": cnet_artifact, "scale_factor": 2}
        })
        assert enh_res.status_code == 201
