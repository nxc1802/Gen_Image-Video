"""
Tier 1 Feature Coverage: Factory Studio Web (Features 40 - 42)
Validates happy path and contract adherence for:
- Feature 40: Dynamic Schema Form (GET /api/studio/schemas)
- Feature 41: Streaming Inspector (GET /api/studio/inspect/stream)
- Feature 42: Code Generator (POST /api/studio/codegen)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestStudioWeb:
    # --- Feature 40: Dynamic Schema Form ---
    def test_f40_studio_schemas_status(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/schemas", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "schemas" in data
        assert "services" in data["schemas"]
        assert "models" in data["schemas"]

    def test_f40_studio_schemas_services_inventory(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/schemas", headers=auth_headers)
        services = res.json()["schemas"]["services"]
        assert len(services) >= 10
        subsystems = {s["subsystem"] for s in services}
        for expected in ["text", "vlm", "decision", "speech", "image", "video", "music", "enhancement"]:
            assert expected in subsystems

    def test_f40_studio_schemas_model_parameters(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/schemas", headers=auth_headers)
        models = res.json()["schemas"]["models"]
        for m in models:
            assert "capabilities" in m
            assert "hardware" in m
            assert "status" in m

    def test_f40_studio_schemas_limits(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/schemas", headers=auth_headers)
        models = res.json()["schemas"]["models"]
        image_model = next((m for m in models if m["id"] == "qwen-image-2.1"), None)
        assert image_model is not None
        assert "limits" in image_model
        assert image_model["limits"]["max_width"] == 2048

    def test_f40_studio_schemas_hardware_requirements(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/schemas", headers=auth_headers)
        models = res.json()["schemas"]["models"]
        video_model = next((m for m in models if m["id"] == "wan-2.2"), None)
        assert video_model is not None
        assert video_model["hardware"]["min_vram_gb"] >= 14

    # --- Feature 41: Streaming Inspector ---
    def test_f41_streaming_inspector_status_code(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/inspect/stream", headers=auth_headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_f41_streaming_inspector_telemetry_event(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/inspect/stream", headers=auth_headers)
        text = res.text
        assert "event: telemetry" in text
        assert "active_jobs" in text

    def test_f41_streaming_inspector_vram_gauges(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/inspect/stream", headers=auth_headers)
        assert "gpu_0_vram_used_gb" in res.text
        assert "gpu_1_vram_used_gb" in res.text

    def test_f41_streaming_inspector_queue_depth(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/inspect/stream", headers=auth_headers)
        assert "queue_depth" in res.text

    def test_f41_streaming_inspector_ping_keepalive(self, client: httpx.Client, auth_headers):
        res = client.get("/api/studio/inspect/stream", headers=auth_headers)
        assert "event: ping" in res.text

    # --- Feature 42: Code Generator ---
    def test_f42_codegen_curl(self, client: httpx.Client, auth_headers):
        payload = {"service": "image", "model": "qwen-image-2.1", "task": "text_to_image"}
        res = client.post("/api/studio/codegen", headers=auth_headers, json=payload)
        assert res.status_code == 200
        curl = res.json()["curl"]
        assert curl.startswith("curl -X POST")
        assert "https://jobs.ai.example.com/v1/jobs" in curl
        assert "Authorization: Bearer" in curl

    def test_f42_codegen_python_httpx(self, client: httpx.Client, auth_headers):
        payload = {"service": "video", "model": "wan-2.2", "task": "text_to_video"}
        res = client.post("/api/studio/codegen", headers=auth_headers, json=payload)
        assert res.status_code == 200
        py_code = res.json()["python"]
        assert "import httpx" in py_code
        assert "httpx.post" in py_code
        assert "wan-2.2" in py_code

    def test_f42_codegen_typescript(self, client: httpx.Client, auth_headers):
        payload = {"service": "music", "model": "ace-step-1.5", "task": "text_to_music"}
        res = client.post("/api/studio/codegen", headers=auth_headers, json=payload)
        assert res.status_code == 200
        ts_code = res.json()["typescript"]
        assert "fetch(" in ts_code
        assert "Authorization" in ts_code
        assert "ace-step-1.5" in ts_code

    def test_f42_codegen_service_and_task_reflection(self, client: httpx.Client, auth_headers):
        payload = {"service": "enhancement", "model": "enhancement-suite-v1", "task": "image_super_resolution"}
        res = client.post("/api/studio/codegen", headers=auth_headers, json=payload)
        data = res.json()
        assert "enhancement" in data["curl"]
        assert "image_super_resolution" in data["python"]

    def test_f42_codegen_all_client_keys_present(self, client: httpx.Client, auth_headers):
        res = client.post("/api/studio/codegen", headers=auth_headers, json={})
        data = res.json()
        assert "curl" in data
        assert "python" in data
        assert "typescript" in data
