"""
Tier 1 Feature Coverage: Service & Model Registry (Features 7 - 8)
Validates happy path and contract adherence for:
- Feature 7: Service Manifest Discovery (GET /v1/registry/services)
- Feature 8: Model Manifest Registry (GET /v1/registry/models)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestRegistry:
    # --- Feature 7: Service Manifest Discovery ---
    def test_f07_service_discovery_all(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/services", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "services" in data
        assert len(data["services"]) >= 10
        assert data["total"] == len(data["services"])

    def test_f07_service_discovery_filter_by_subsystem(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/services?subsystem=image", headers=auth_headers)
        assert res.status_code == 200
        services = res.json()["services"]
        assert len(services) >= 1
        for s in services:
            assert s["subsystem"] == "image"

    def test_f07_service_discovery_ngrok_static_domains(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/services", headers=auth_headers)
        services = res.json()["services"]
        for s in services:
            domain = s["network"]["static_domain"]
            assert domain.endswith(".ai.example.com")
            assert len(domain.split(".")) >= 3

    def test_f07_service_discovery_framework_types(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/services", headers=auth_headers)
        services = res.json()["services"]
        valid_frameworks = {"vllm", "vllm-omni", "vllm/tei", "comfyui", "acestep", "fastapi", "nextjs", "custom"}
        for s in services:
            assert s["framework"] in valid_frameworks

    def test_f07_service_discovery_hardware_metadata(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/services", headers=auth_headers)
        services = res.json()["services"]
        for s in services:
            hw = s["hardware"]
            assert "gpu" in hw
            assert "min_vram_gb" in hw
            assert isinstance(hw["min_vram_gb"], (int, float))

    # --- Feature 8: Model Manifest Registry ---
    def test_f08_model_registry_list_all(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/models", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        assert len(data["models"]) >= 10
        assert data["total"] == len(data["models"])

    def test_f08_model_registry_filter_by_family(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/models?family=qwen-image", headers=auth_headers)
        assert res.status_code == 200
        models = res.json()["models"]
        assert len(models) >= 1
        for m in models:
            assert m["family"] == "qwen-image"

    def test_f08_model_registry_status_and_coverage(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/models", headers=auth_headers)
        models = res.json()["models"]
        for m in models:
            assert m["status"] in ["full", "partial"]
            assert 0.0 <= m["coverage_percent"] <= 100.0
            if m["status"] == "full":
                assert m["coverage_percent"] == 100.0

    def test_f08_model_registry_capabilities_map(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/models", headers=auth_headers)
        models = res.json()["models"]
        for m in models:
            caps = m["capabilities"]
            assert isinstance(caps, dict)
            assert len(caps) >= 1
            for cap_name, supported in caps.items():
                assert isinstance(supported, bool)

    def test_f08_model_registry_hardware_constraints(self, client: httpx.Client, auth_headers):
        res = client.get("/v1/registry/models", headers=auth_headers)
        models = res.json()["models"]
        for m in models:
            hw = m["hardware"]
            assert "min_vram_gb" in hw
            assert hw["min_vram_gb"] > 0
            assert "preferred_gpu" in hw
