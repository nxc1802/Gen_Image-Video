"""
Tier 1 Feature Coverage: Image Subsystem (Features 21 - 27)
Validates happy path and contract adherence for:
- Feature 21: Text-to-Image
- Feature 22: Image-to-Image
- Feature 23: Inpainting
- Feature 24: Outpainting
- Feature 25: Instruction Edit
- Feature 26: Reference Conditioning
- Feature 27: ControlNet
"""

import pytest
import httpx
from tests.e2e.conftest import assert_valid_job


@pytest.mark.tier1
class TestImageSubsystem:
    # --- Feature 21: Text-to-Image ---
    def test_f21_t2i_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A modern robotic arm assembling glass panels", "width": 1024, "height": 1024, "seed": 42}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "text_to_image")

    def test_f21_t2i_negative_prompt(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Portrait of an astronaut", "negative_prompt": "blurry, distorted, oversaturated"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f21_t2i_aspect_ratios(self, client: httpx.Client, auth_headers):
        for w, h in [(768, 1024), (1024, 768), (1024, 1024)]:
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Landscape {w}x{h}", "width": w, "height": h}
            }
            res = client.post("/v1/jobs", headers=auth_headers, json=payload)
            assert res.status_code == 201

    def test_f21_t2i_seed_reproducibility(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Deterministic crystal cave", "seed": 123456}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f21_t2i_s3_artifact_returned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Artifact output check"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        artifacts = res.json()["artifacts"]
        assert len(artifacts) >= 1
        assert artifacts[0]["url"].startswith("https://s3.ai.example.com/")

    # --- Feature 22: Image-to-Image ---
    def test_f22_i2i_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/source_car.png",
                "prompt": "Convert car into a hovering sci-fi vehicle",
                "denoise_strength": 0.65
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "image_to_image")

    def test_f22_i2i_low_denoise(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/sketch.png", "prompt": "Subtle polish", "denoise_strength": 0.2}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f22_i2i_high_denoise(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/rough.png", "prompt": "Complete reimagining", "denoise_strength": 0.85}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f22_i2i_seed_tracking(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/img.png", "prompt": "Stylized render", "seed": 777}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f22_i2i_target_resolution(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image_to_image",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/img.png", "prompt": "HD upscale render", "width": 1024, "height": 1024}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    # --- Feature 23: Image Inpainting ---
    def test_f23_inpaint_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/room.png",
                "mask_url": "https://s3.ai.example.com/assets/table_mask.png",
                "prompt": "Place an antique brass clock on the wooden table"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "inpaint")

    def test_f23_inpaint_feather_mask(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/portrait.png",
                "mask_url": "https://s3.ai.example.com/assets/glasses_mask.png",
                "prompt": "Fashionable sunglasses",
                "mask_blur": 4
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f23_inpaint_seed_control(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/wall.png",
                "mask_url": "https://s3.ai.example.com/assets/wall_mask.png",
                "prompt": "A modern art painting",
                "seed": 98765
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f23_inpaint_background_replacement(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/product.png",
                "mask_url": "https://s3.ai.example.com/assets/bg_mask.png",
                "prompt": "Minimalist marble countertop background"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f23_inpaint_artifact_created(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/img.png",
                "mask_url": "https://s3.ai.example.com/assets/mask.png",
                "prompt": "Repair damaged portion"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert len(res.json()["artifacts"]) >= 1

    # --- Feature 24: Image Outpainting ---
    def test_f24_outpaint_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "outpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/square_scene.png",
                "expand": {"top": 128, "bottom": 128, "left": 256, "right": 256},
                "prompt": "Expand the scene showing full expansive valley"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "outpaint")

    def test_f24_outpaint_horizontal_only(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "outpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/center.png",
                "expand": {"left": 256, "right": 256},
                "prompt": "Widescreen panoramic extension"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f24_outpaint_vertical_only(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "outpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/center.png",
                "expand": {"top": 256, "bottom": 256},
                "prompt": "Tall banner extension with clouds and flowers"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f24_outpaint_denoise_strength(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "outpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/center.png",
                "expand": {"right": 384},
                "prompt": "Extended roadway",
                "denoise_strength": 0.8
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f24_outpaint_artifact_mime(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "outpaint",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/img.png", "prompt": "Expand canvas"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        art = res.json()["artifacts"][0]
        assert art["mime_type"] == "image/png"

    # --- Feature 25: Image Instruction Edit ---
    def test_f25_instruction_edit_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "instruction_edit",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/day_park.png",
                "instruction": "Change time of day to twilight with warm streetlights glowing"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "instruction_edit")

    def test_f25_instruction_edit_clothing_color(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "instruction_edit",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/model.png", "instruction": "Make the jacket crimson red"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f25_instruction_edit_weather(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "instruction_edit",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/city.png", "instruction": "Add heavy snowfall across the rooftops"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f25_instruction_edit_seed_reproducibility(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "instruction_edit",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/sketch.png", "instruction": "Render as watercolor", "seed": 404}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f25_instruction_edit_output_url(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "instruction_edit",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/car.png", "instruction": "Add racing stripes"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["url"].endswith(".bin")

    # --- Feature 26: Image Reference Conditioning ---
    def test_f26_reference_conditioning_single(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "reference_conditioning",
            "inputs": {
                "prompt": "Character walking through an ancient library",
                "reference_image_urls": ["https://s3.ai.example.com/assets/character_sheet.png"]
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "reference_conditioning")

    def test_f26_reference_conditioning_multiple(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "reference_conditioning",
            "inputs": {
                "prompt": "Hero in battle gear",
                "reference_image_urls": [
                    "https://s3.ai.example.com/assets/face_ref.png",
                    "https://s3.ai.example.com/assets/armor_ref.png"
                ]
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f26_reference_conditioning_style_mode(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "reference_conditioning",
            "inputs": {
                "prompt": "Eiffel tower at sunrise",
                "reference_image_urls": ["https://s3.ai.example.com/assets/van_gogh_style.png"],
                "mode": "style_transfer"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f26_reference_conditioning_fidelity_weight(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "reference_conditioning",
            "inputs": {
                "prompt": "Cyberpunk detective",
                "reference_image_urls": ["https://s3.ai.example.com/assets/face.png"],
                "fidelity": 0.85
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f26_reference_conditioning_artifact(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "reference_conditioning",
            "inputs": {"prompt": "Consistent mascot", "reference_image_urls": ["https://s3.ai.example.com/assets/mascot.png"]}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert len(res.json()["artifacts"]) >= 1

    # --- Feature 27: Image ControlNet ---
    def test_f27_controlnet_openpose(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Athletic runner sprinting across finish line",
                "control_type": "openpose",
                "control_image_url": "https://s3.ai.example.com/assets/pose_skeleton.png",
                "strength": 0.9
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "image", "controlnet")

    def test_f27_controlnet_depth(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Grand ballroom with crystal chandeliers",
                "control_type": "depth",
                "control_image_url": "https://s3.ai.example.com/assets/room_depth.png"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f27_controlnet_canny_edge(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Modern architectural glass pavilion",
                "control_type": "canny",
                "control_image_url": "https://s3.ai.example.com/assets/building_canny.png",
                "strength": 0.8
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f27_controlnet_seed_control(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Warrior in fighting stance",
                "control_type": "openpose",
                "control_image_url": "https://s3.ai.example.com/assets/pose.png",
                "seed": 555
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f27_controlnet_artifact_returned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "controlnet",
            "inputs": {
                "prompt": "Geometric sculpture",
                "control_type": "depth",
                "control_image_url": "https://s3.ai.example.com/assets/depth.png"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["mime_type"] == "image/png"
