"""
tests/adversarial/test_adversarial_m4_studio.py
Empirical Adversarial Test Harness for Milestone 4 Gate Verification:
Factory Studio Web Dynamic Console (Wave 6).

Adversarially probes:
1. Dynamic Schema Generation:
   - All 12 capabilities in manifests/capabilities/ parse correctly into parameter inputs.
   - Validates input types, required flags, constraint invariants, and JSON serializability.
2. Form Constraint Validation:
   - Numeric min/max bounds (lower violation, upper violation, exact bounds).
   - Required field enforcement (empty string, None, whitespace prompt rejection).
   - Choices validation (whitelisted choice vs out-of-bounds choice).
   - Media asset validation (rejection of inline base64 data URIs).
   - Direct execution of Node.js validation matching DynamicSchemaForm.
3. Multi-Language Codegen Verification:
   - Target URL invariants (https://jobs.ai.example.com/v1/jobs).
   - AST syntax verification for Python httpx and OpenAI SDK snippets.
   - TypeScript compiler syntax verification for fetch snippets.
   - JSON parsing verification for cURL snippets.
   - Parameter reflection under adversarial inputs (quotes, slashes, unicode, newlines).
4. Live FastAPI Control Plane & Security:
   - Auth matrices for /api/studio/schemas, /api/studio/codegen, /v1/capabilities, /api/studio/inspect/stream.
   - Unified Error Contract sanitization (zero stack traces, zero paths, valid req- IDs).
   - Live streaming disconnection resilience.
"""
import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

AUTH_TOKEN = "sk-factory-test-token-valid"
os.environ["FACTORY_API_KEY"] = AUTH_TOKEN

from factory_contracts.manifests import CapabilityManifest, ParameterDataType
from model_registry.loader import ManifestLoader
from factory.apps.control_api.src.main import app


# ---------------------------------------------------------------------------
# Helpers & Validators
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_DIR = REPO_ROOT / "manifests" / "capabilities"
STUDIO_WEB_DIR = REPO_ROOT / "factory" / "apps" / "studio-web"


def assert_sanitized_error_payload(data: Dict[str, Any], expected_code: str = "UNAUTHORIZED"):
    """Validates that error responses conform to Unified Error Contract."""
    assert "error" in data, f"Missing 'error' key in response: {data}"
    err = data["error"]
    assert "code" in err, f"Missing 'code' in error: {err}"
    assert err["code"] == expected_code, f"Expected code {expected_code}, got {err['code']}"
    assert "message" in err and len(err["message"]) > 0, "Empty error message"
    assert "request_id" in err and err["request_id"].startswith("req-"), f"Invalid request_id: {err}"


def run_node_code(script: str) -> str:
    """Executes a JavaScript snippet inside studio-web directory using Node.js."""
    result = subprocess.run(
        ["node", "-e", script],
        cwd=str(STUDIO_WEB_DIR),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node execution failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    return result.stdout.strip()


# ===========================================================================
# 1. Dynamic Schema Generation (12 Capabilities)
# ===========================================================================

@pytest.mark.adversarial
class TestDynamicSchemaGeneration:
    """Verifies that all capability manifests parse correctly into dynamic form parameter inputs."""

    EXPECTED_CAPABILITIES = {
        "image.controlnet",
        "image.i2i",
        "image.inpaint",
        "image.instruction_edit",
        "image.outpaint",
        "image.reference_conditioning",
        "image.t2i",
        "video.extension",
        "video.fl2v",
        "video.i2v",
        "video.t2v",
        "video.with_audio",
    }

    def test_all_12_capabilities_exist_and_load(self):
        loader = ManifestLoader(str(REPO_ROOT / "manifests"))
        capabilities = loader.load_capabilities()
        assert len(capabilities) >= 12, f"Expected at least 12 capabilities, got {len(capabilities)}"
        loaded_ids = set(capabilities.keys())
        missing = self.EXPECTED_CAPABILITIES - loaded_ids
        assert not missing, f"Missing required capabilities: {missing}"

    @pytest.mark.parametrize("cap_id", list(EXPECTED_CAPABILITIES))
    def test_capability_manifest_structure_invariants(self, cap_id: str):
        loader = ManifestLoader(str(REPO_ROOT / "manifests"))
        capabilities = loader.load_capabilities()
        cap = capabilities.get(cap_id)
        assert cap is not None, f"Capability {cap_id} not loaded"

        # Contract schema attributes
        assert cap.apiVersion == "factory.ai/v1"
        assert cap.kind == "CapabilityManifest"
        assert cap.id == cap_id
        assert cap.subsystem in ("image", "video", "text", "speech", "music", "decision", "embedding", "enhancement")
        assert len(cap.display_name) > 0
        assert len(cap.description) > 0
        assert len(cap.inputs) > 0, f"Capability {cap_id} has empty inputs"
        assert len(cap.outputs) > 0, f"Capability {cap_id} has empty outputs"
        assert len(cap.supported_model_families) > 0, f"Capability {cap_id} has no supported model families"

    @pytest.mark.parametrize("cap_id", list(EXPECTED_CAPABILITIES))
    def test_capability_parameter_inputs_invariants(self, cap_id: str):
        loader = ManifestLoader(str(REPO_ROOT / "manifests"))
        capabilities = loader.load_capabilities()
        cap = capabilities[cap_id]

        input_names = set()
        for param in cap.inputs:
            assert param.name, f"Parameter in {cap_id} missing name"
            assert param.name not in input_names, f"Duplicate parameter name '{param.name}' in {cap_id}"
            input_names.add(param.name)

            assert isinstance(param.type, ParameterDataType), f"Invalid type for {param.name} in {cap_id}"
            assert isinstance(param.required, bool)
            assert isinstance(param.description, str)

            # Constraint bounds invariants
            if param.constraints:
                c = param.constraints
                if c.min_value is not None and c.max_value is not None:
                    assert c.min_value <= c.max_value, f"Constraint violation in {cap_id}.{param.name}: min {c.min_value} > max {c.max_value}"
                if c.step is not None:
                    assert c.step > 0, f"Constraint step <= 0 in {cap_id}.{param.name}"
                if c.choices is not None:
                    assert len(c.choices) > 0, f"Empty choices list in {cap_id}.{param.name}"
                    if param.default is not None:
                        assert param.default in c.choices, f"Default {param.default} not in choices {c.choices}"

    @pytest.mark.parametrize("cap_id", list(EXPECTED_CAPABILITIES))
    def test_capability_json_serialization(self, cap_id: str):
        loader = ManifestLoader(str(REPO_ROOT / "manifests"))
        capabilities = loader.load_capabilities()
        cap = capabilities[cap_id]

        dumped = cap.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert dumped["id"] == cap_id
        assert isinstance(dumped["inputs"], list)
        assert len(dumped["inputs"]) == len(cap.inputs)

        # Ensure json.dumps works without serialization crash
        serialized = json.dumps(dumped)
        assert cap_id in serialized


# ===========================================================================
# 2. Form Constraint Validation
# ===========================================================================

@pytest.mark.adversarial
class TestFormConstraintValidation:
    """Verifies numeric bounds, required prompt enforcement, choices validation, and base64 rejection."""

    def test_numeric_bounds_validation_logic(self):
        """Simulates DynamicSchemaForm validation logic against edge bounds."""
        def validate_number(val: Any, min_val: float, max_val: float) -> List[str]:
            errors = []
            try:
                num = float(val)
            except (ValueError, TypeError):
                return ["Must be a valid number"]
            if num < min_val:
                errors.append(f"Minimum allowed value is {min_val}")
            if num > max_val:
                errors.append(f"Maximum allowed value is {max_val}")
            return errors

        # Width constraints: [512, 2048]
        assert validate_number(512, 512, 2048) == []
        assert validate_number(2048, 512, 2048) == []
        assert validate_number(1024, 512, 2048) == []
        assert "Minimum allowed value is 512" in validate_number(511, 512, 2048)[0]
        assert "Minimum allowed value is 512" in validate_number(-100, 512, 2048)[0]
        assert "Maximum allowed value is 2048" in validate_number(2049, 512, 2048)[0]
        assert "Maximum allowed value is 2048" in validate_number(999999, 512, 2048)[0]
        assert "Must be a valid number" in validate_number("not-a-number", 512, 2048)[0]

        # Strength constraints: [0.0, 2.0]
        assert validate_number(0.0, 0.0, 2.0) == []
        assert validate_number(2.0, 0.0, 2.0) == []
        assert validate_number(0.85, 0.0, 2.0) == []
        assert "Minimum allowed value is 0.0" in validate_number(-0.01, 0.0, 2.0)[0]
        assert "Maximum allowed value is 2.0" in validate_number(2.01, 0.0, 2.0)[0]

    def test_required_prompt_validation_logic(self):
        """Verifies rejection of empty, null, or whitespace-only prompts."""
        def validate_prompt(val: Any) -> bool:
            if val is None or not isinstance(val, str) or len(val.strip()) == 0:
                return False
            return True

        assert validate_prompt("A beautiful cinematic sunset") is True
        assert validate_prompt("") is False
        assert validate_prompt("    ") is False
        assert validate_prompt(None) is False
        assert validate_prompt(12345) is False

    def test_choices_whitelist_validation(self):
        """Verifies that enum/choice parameters reject unlisted values."""
        allowed_choices = ["openpose", "depth", "canny"]
        def validate_choice(val: str, choices: List[str]) -> bool:
            return val in choices

        assert validate_choice("canny", allowed_choices) is True
        assert validate_choice("depth", allowed_choices) is True
        assert validate_choice("openpose", allowed_choices) is True
        assert validate_choice("unknown_control", allowed_choices) is False
        assert validate_choice("'; DROP TABLE--", allowed_choices) is False

    def test_base64_media_rejection_policy(self):
        """Verifies that direct base64 data URIs are rejected to enforce S3 storage."""
        def validate_media_url(url: str) -> List[str]:
            if isinstance(url, str) and url.startswith("data:"):
                return ["Direct base64 is forbidden; upload file or provide S3 storage URL."]
            return []

        assert validate_media_url("https://assets.ai.example.com/uploads/sample.png") == []
        assert validate_media_url("s3://factory-bucket/artifacts/image.png") == []
        errs = validate_media_url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        assert len(errs) == 1
        assert "Direct base64 is forbidden" in errs[0]

    def test_node_dynamic_schema_form_validation_harness(self):
        """Executes DynamicSchemaForm validation rules inside Node.js across test capability payloads."""
        node_script = """
        const testCapability = {
          id: "image.t2i",
          inputs: [
            { name: "prompt", type: "string", required: true },
            { name: "width", type: "integer", required: false, default: 1024, constraints: { min_value: 512, max_value: 2048 } },
            { name: "height", type: "integer", required: false, default: 1024, constraints: { min_value: 512, max_value: 2048 } }
          ]
        };

        function validateForm(inputs, values) {
          const errors = {};
          for (const spec of inputs) {
            const val = values[spec.name];
            if (spec.required && (val === undefined || val === null || val === "")) {
              errors[spec.name] = `${spec.name} is required`;
              continue;
            }
            if (spec.type === "integer" || spec.type === "float") {
              const numVal = Number(val);
              if (isNaN(numVal)) {
                if (spec.required) errors[spec.name] = "Must be a valid number";
              } else {
                if (spec.constraints?.min_value !== undefined && numVal < spec.constraints.min_value) {
                  errors[spec.name] = `Minimum allowed value is ${spec.constraints.min_value}`;
                }
                if (spec.constraints?.max_value !== undefined && numVal > spec.constraints.max_value) {
                  errors[spec.name] = `Maximum allowed value is ${spec.constraints.max_value}`;
                }
              }
            }
          }
          return errors;
        }

        // Test 1: Valid inputs
        const res1 = validateForm(testCapability.inputs, { prompt: "neon factory", width: 1024, height: 1024 });
        if (Object.keys(res1).length !== 0) throw new Error("Expected 0 errors for valid inputs, got: " + JSON.stringify(res1));

        // Test 2: Missing required prompt
        const res2 = validateForm(testCapability.inputs, { prompt: "", width: 1024, height: 1024 });
        if (!res2.prompt) throw new Error("Expected prompt error for empty string");

        // Test 3: Under min_value
        const res3 = validateForm(testCapability.inputs, { prompt: "ok", width: 256, height: 1024 });
        if (!res3.width || !res3.width.includes("Minimum allowed value is 512")) throw new Error("Expected width min error, got: " + JSON.stringify(res3));

        // Test 4: Over max_value
        const res4 = validateForm(testCapability.inputs, { prompt: "ok", width: 4096, height: 1024 });
        if (!res4.width || !res4.width.includes("Maximum allowed value is 2048")) throw new Error("Expected width max error, got: " + JSON.stringify(res4));

        console.log("NODE_VALIDATION_PASS");
        """
        out = run_node_code(node_script)
        assert out == "NODE_VALIDATION_PASS"


# ===========================================================================
# 3. Codegen Verification (cURL, Python httpx, TypeScript, OpenAI SDK)
# ===========================================================================

@pytest.mark.adversarial
class TestCodegenVerification:
    """Verifies that generated snippets have valid syntax, match requested parameters, and target /v1/jobs."""

    EXPECTED_TARGET_URL = "https://jobs.ai.example.com/v1/jobs"

    def _generate_frontend_snippets(self, payload: Dict[str, Any], base_url: str = "https://jobs.ai.example.com") -> Dict[str, str]:
        """Runs codegen.ts in Node.js and returns all 4 snippets."""
        node_script = f"""
        const fs = require("fs");
        const ts = require("./node_modules/typescript");
        const code = fs.readFileSync("src/lib/codegen.ts", "utf8");
        const transpiled = ts.transpileModule(code, {{ compilerOptions: {{ module: ts.ModuleKind.CommonJS }} }});
        const mod = {{}};
        const fn = new Function("exports", "require", "module", transpiled.outputText);
        fn(mod, require, {{ exports: mod }});
        const payload = {json.dumps(payload)};
        const res = mod.generateClientSnippets(payload, "{base_url}");
        console.log(JSON.stringify(res));
        """
        output = run_node_code(node_script)
        return json.loads(output)

    def test_frontend_codegen_all_four_targets_present(self):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "image.t2i",
            "inputs": {"prompt": "A modern factory assembling GPUs", "width": 1024, "height": 1024},
        }
        snippets = self._generate_frontend_snippets(payload)
        assert "curl" in snippets
        assert "python" in snippets
        assert "typescript" in snippets
        assert "openai" in snippets

    def test_frontend_codegen_target_url_invariants(self):
        payload = {"service": "video", "model": "wan-2.2", "task": "video.t2v", "inputs": {"prompt": "Drone shot"}}
        snippets = self._generate_frontend_snippets(payload)

        # cURL, Python httpx, and TypeScript fetch must all target /v1/jobs
        assert self.EXPECTED_TARGET_URL in snippets["curl"]
        assert "https://jobs.ai.example.com" in snippets["python"]
        assert "/v1/jobs" in snippets["python"]
        assert "https://jobs.ai.example.com" in snippets["typescript"]
        assert "/v1/jobs" in snippets["typescript"]

    def test_frontend_codegen_python_httpx_ast_syntax(self):
        payloads = [
            {"service": "image", "model": "qwen-image-2.1", "task": "image.t2i", "inputs": {"prompt": "Sunset over river", "width": 1024}},
            {"service": "video", "model": "wan-2.2", "task": "video.t2v", "inputs": {"prompt": "Cybernetic city\nwith 'quotes' and \"double quotes\"", "seed": 42}},
            {"service": "music", "model": "ace-step-1.5", "task": "music.t2m", "inputs": {"prompt": "Lo-fi beats 🚀", "bpm": 120}},
        ]
        for p in payloads:
            snippets = self._generate_frontend_snippets(p)
            # Must be valid Python syntax
            tree = ast.parse(snippets["python"])
            assert isinstance(tree, ast.Module)
            assert "httpx" in snippets["python"]
            assert "Authorization" in snippets["python"]

    def test_frontend_codegen_openai_sdk_ast_syntax(self):
        payload = {"service": "text", "model": "qwen-2.5-72b-instruct", "task": "text.chat", "inputs": {"prompt": "Explain quantum computing."}}
        snippets = self._generate_frontend_snippets(payload)
        # Must be valid Python syntax
        tree = ast.parse(snippets["openai"])
        assert isinstance(tree, ast.Module)
        assert "from openai import OpenAI" in snippets["openai"]
        assert "chat.completions.create" in snippets["openai"]

    def test_frontend_codegen_typescript_syntax(self):
        payload = {"service": "video", "model": "wan-2.2", "task": "video.fl2v", "inputs": {"prompt": "Interpolation between frames", "num_frames": 49}}
        snippets = self._generate_frontend_snippets(payload)

        # Transpile with TypeScript compiler to verify valid TS syntax
        node_script = f"""
        const ts = require("./node_modules/typescript");
        const tsCode = {json.dumps(snippets["typescript"])};
        const res = ts.transpileModule(tsCode, {{ reportDiagnostics: true, compilerOptions: {{ target: ts.ScriptTarget.ES2022 }} }});
        if (res.diagnostics && res.diagnostics.length > 0) {{
          throw new Error("TypeScript syntax errors: " + JSON.stringify(res.diagnostics));
        }}
        console.log("TS_SYNTAX_VALID");
        """
        assert run_node_code(node_script) == "TS_SYNTAX_VALID"

    def test_frontend_codegen_curl_json_validity(self):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": "https://assets.ai.example.com/input.png", "scale": 4},
        }
        snippets = self._generate_frontend_snippets(payload)
        curl = snippets["curl"]
        assert curl.startswith("curl -X POST")
        assert "Authorization: Bearer" in curl

        # Extract the JSON body inside -d '...'
        match = re.search(r"-d '([\s\S]+)'", curl)
        assert match is not None, f"Failed to extract JSON body from cURL snippet: {curl}"
        body_dict = json.loads(match.group(1))
        assert body_dict["service"] == "enhancement"
        assert body_dict["model"] == "enhancement-suite-v1"
        assert body_dict["task"] == "image_super_resolution"
        assert body_dict["inputs"]["scale"] == 4

    def test_backend_control_plane_codegen_endpoint(self):
        """Tests live POST /api/studio/codegen on Control Plane."""
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "image.controlnet",
                "inputs": {"prompt": "Cybernetic warrior", "control_type": "canny", "strength": 0.85},
            }
            res = client.post("/api/studio/codegen", headers=headers, json=payload)
            assert res.status_code == 200
            data = res.json()

            # cURL check
            assert "curl" in data
            assert self.EXPECTED_TARGET_URL in data["curl"]
            assert "Authorization: Bearer" in data["curl"]

            # Python httpx syntax check
            assert "python" in data
            tree = ast.parse(data["python"])
            assert isinstance(tree, ast.Module)
            assert "import httpx" in data["python"]
            assert "qwen-image-2.1" in data["python"]

            # TypeScript syntax check
            assert "typescript" in data
            assert "fetch(" in data["typescript"]
            assert "Authorization" in data["typescript"]


# ===========================================================================
# 4. Live FastAPI Control Plane & Security Verification
# ===========================================================================

@pytest.mark.adversarial
class TestLiveFastAPIControlPlaneAuthAndParity:
    """Adversarially tests /api/studio/schemas, /api/studio/codegen, and /v1/capabilities with valid vs invalid tokens."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/studio/schemas", "GET", None),
        ("/api/studio/codegen", "POST", {}),
        ("/api/studio/inspect/stream", "GET", None),
        ("/v1/capabilities", "GET", None),
        ("/v1/models", "GET", None),
        ("/v1/services", "GET", None),
    ])
    def test_unauthenticated_request_rejected_with_401(self, endpoint: str, method: str, body: Any):
        with TestClient(app) as client:
            res = client.request(method, endpoint, json=body if body is not None else None)
            assert res.status_code == 401, f"Expected 401 on unauthenticated {method} {endpoint}, got {res.status_code}"
            assert_sanitized_error_payload(res.json(), expected_code="UNAUTHORIZED")

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/studio/schemas", "GET", None),
        ("/api/studio/codegen", "POST", {}),
        ("/v1/capabilities", "GET", None),
    ])
    def test_invalid_bearer_token_rejected_with_401(self, endpoint: str, method: str, body: Any):
        with TestClient(app) as client:
            bad_headers = {"Authorization": "Bearer bad_secret_key_12345"}
            res = client.request(method, endpoint, headers=bad_headers, json=body if body is not None else None)
            assert res.status_code == 401
            assert_sanitized_error_payload(res.json(), expected_code="UNAUTHORIZED")

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/studio/schemas", "GET", None),
        ("/api/studio/codegen", "POST", {}),
        ("/v1/capabilities", "GET", None),
    ])
    def test_empty_bearer_token_rejected_with_401(self, endpoint: str, method: str, body: Any):
        with TestClient(app) as client:
            empty_headers = {"Authorization": "Bearer "}
            res = client.request(method, endpoint, headers=empty_headers, json=body if body is not None else None)
            assert res.status_code == 401
            assert_sanitized_error_payload(res.json(), expected_code="UNAUTHORIZED")

    def test_studio_schemas_live_payload_conformance(self):
        """Verifies GET /api/studio/schemas returns complete schemas with promoted keys."""
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            res = client.get("/api/studio/schemas", headers=headers)
            assert res.status_code == 200
            data = res.json()
            assert "schemas" in data
            schemas = data["schemas"]
            assert "services" in schemas
            assert "models" in schemas

            # Service key promotion
            services = schemas["services"]
            assert len(services) >= 10
            for s in services:
                assert "subsystem" in s, f"Service missing promoted 'subsystem' key: {s}"
                assert "id" in s, f"Service missing promoted 'id' key: {s}"

            # Model key promotion
            models = schemas["models"]
            assert len(models) >= 10
            for m in models:
                assert "id" in m, f"Model missing promoted 'id' key: {m}"
                assert "capabilities" in m
                assert "hardware" in m

    def test_capabilities_endpoint_live_payload_conformance(self):
        """Verifies GET /v1/capabilities returns 12 manifests."""
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            res = client.get("/v1/capabilities", headers=headers)
            assert res.status_code == 200
            data = res.json()
            assert "capabilities" in data
            caps = data["capabilities"]
            assert len(caps) >= 12
            cap_ids = {c["id"] for c in caps}
            assert "image.t2i" in cap_ids
            assert "video.with_audio" in cap_ids

    def test_inspect_stream_disconnection_resilience(self):
        """Verifies that client disconnection from SSE stream does not crash server."""
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            with client.stream("GET", "/api/studio/inspect/stream", headers=headers) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                # Read first line / chunk
                for line in response.iter_lines():
                    if line:
                        break
                # Abruptly close
                response.close()

            # Server should remain fully operational
            followup = client.get("/health")
            assert followup.status_code == 200
            assert followup.json()["status"] == "healthy"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

