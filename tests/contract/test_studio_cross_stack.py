"""
tests/contract/test_studio_cross_stack.py
Cross-Stack Isolation & Studio Contract Synchronization Test Suite (P1.7):
- Asserts presence and integrity of factory/apps/studio-web dependencies
- Enforces ZERO silent skips in CI runners
- Verifies Studio TypeScript build / typecheck via tsc --noEmit
- Verifies parity between capability manifest input types and Studio ParameterType
- Verifies OpenAPI route registry parity with StudioApiClient
"""
import os
from pathlib import Path
import subprocess
import pytest
import yaml

from factory.apps.control_api.src.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STUDIO_DIR = REPO_ROOT / "factory" / "apps" / "studio-web"
IS_CI = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


class TestStudioCrossStackIsolation:
    """Verifies that Studio node dependencies are present and compile with zero silent skips."""

    def test_studio_dependencies_available_zero_silent_skips(self):
        """Verifies studio-web package.json and node_modules exist. Fails loudly on CI if missing."""
        assert (STUDIO_DIR / "package.json").exists(), "Studio Web package.json missing"

        typescript_bin = STUDIO_DIR / "node_modules" / "typescript"
        if not typescript_bin.exists():
            if IS_CI:
                pytest.fail(
                    "Cross-Stack Failure: factory/apps/studio-web/node_modules is missing in CI! "
                    "The backend CI workflow must run 'npm ci' to guarantee zero silent skips."
                )
            else:
                pytest.skip("Studio node_modules not installed locally; run 'npm ci' in factory/apps/studio-web")

        assert typescript_bin.exists()

    def test_studio_typescript_typecheck(self):
        """Executes npm run typecheck (tsc --noEmit) in Studio Web directory."""
        if not (STUDIO_DIR / "node_modules" / "typescript").exists():
            if IS_CI:
                pytest.fail("Cannot run TypeScript typecheck in CI: node_modules missing!")
            pytest.skip("Studio node_modules not installed")

        res = subprocess.run(["npm", "run", "typecheck"], cwd=STUDIO_DIR, capture_output=True, text=True)
        assert res.returncode == 0, f"TypeScript typecheck failed:\n{res.stdout}\n{res.stderr}"

    def test_manifest_inputs_compatible_with_studio_types(self):
        """Verifies that all 12 capability manifests declare parameter types supported by Studio Web."""
        valid_studio_types = {
            "string", "integer", "float", "boolean", "image_url",
            "audio_url", "video_url", "mask_url", "choices", "json", "array"
        }
        caps_dir = REPO_ROOT / "manifests" / "capabilities"
        caps = list(caps_dir.glob("*.yaml"))
        assert len(caps) >= 12, f"Expected at least 12 capability manifests, found {len(caps)}"

        for cap_path in caps:
            data = yaml.safe_load(cap_path.read_text(encoding="utf-8"))
            cap_id = data.get("id")
            for param in data.get("inputs", []):
                ptype = param.get("type")
                pname = param.get("name")
                assert ptype in valid_studio_types, (
                    f"Capability '{cap_id}' parameter '{pname}' has type '{ptype}' "
                    f"not supported by Studio Web ParameterType"
                )

    def test_openapi_route_parity_with_studio_client(self):
        """Verifies that all routes called by StudioApiClient exist on the FastAPI Control Plane."""
        openapi_paths = app.openapi().get("paths", {}).keys()
        expected_studio_routes = [
            "/health",
            "/v1/services",
            "/v1/models",
            "/v1/capabilities",
            "/v1/jobs",
            "/v1/jobs/{job_id}",
            "/v1/jobs/{job_id}/result",
            "/v1/jobs/{job_id}/cancel",
            "/v1/jobs/{job_id}/retry",
            "/v1/jobs/{job_id}/events",
            "/v1/jobs/{job_id}/ws",
            "/v1/assets/upload",
            "/v1/memory",
            "/api/studio/schemas",
            "/api/studio/codegen",
            "/api/studio/inspect/stream",
        ]
        for route in expected_studio_routes:
            assert route in openapi_paths, f"Expected Studio route '{route}' missing from Control Plane OpenAPI"
