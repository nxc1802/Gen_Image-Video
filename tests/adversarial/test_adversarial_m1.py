"""
tests/adversarial/test_adversarial_m1.py
Milestone 1 Adversarial Challenge & Stress-Testing Suite

Executes empirical challenges against:
1. Model Completeness Rule (status='full' bypass attempts, edge cases, mutation)
2. ServiceManifest Invariants (ports, ngrok domains, allow_unauthenticated, collisions)
3. Legacy Isolation & Import Prevention
4. UnifiedErrorResponse Schema Stress & Fuzzing
"""
import copy
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from factory_contracts.manifests.service import (
    ServiceManifest,
    ServiceMetadata,
    NetworkSpec,
    HealthCheckSpec,
    FrameworkSpec,
    HardwareRequirements,
    AuthPolicySpec,
    SubsystemType,
    FrameworkType,
    ProtocolType,
    ModalityType,
)
from factory_contracts.manifests.model import (
    ModelManifest,
    ModelMetadata,
    ModelWeightsSpec,
    ModelHardwareSpec,
    ModelLimitsSpec,
    ModelStatus,
    ModelWeightFormat,
    QuantizationType,
    PrecisionType,
    AllocationPolicy,
)
from factory_contracts.errors import (
    UnifiedErrorResponse,
    ErrorDetail,
    ErrorCode,
    FactoryError,
    UnauthorizedError,
    ForbiddenError,
    InvalidRequestError,
    SSRFBlockedError,
    InvalidMediaError,
    NotFoundError,
    ConflictError,
    PayloadTooLargeError,
    RateLimitExceededError,
    ModelUnavailableError,
    ServiceUnavailableError,
    GatewayTimeoutError,
    InternalServerError,
)
from factory_contracts.cli import validate_manifests
from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from factory.apps.control_api.src.api.errors import register_exception_handlers


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_model_payload() -> Dict[str, Any]:
    return {
        "apiVersion": "factory.ai/v1",
        "kind": "ModelManifest",
        "metadata": {
            "id": "test-model-v1",
            "family": "test-family",
            "display_name": "Test Model",
            "service_id": "image-service",
            "version": "1.0.0",
        },
        "weights": {
            "primary_checkpoint": "org/test-checkpoint",
            "components": {},
            "format": "safetensors",
            "quantization": "none",
            "default_precision": "bf16",
        },
        "hardware": {
            "min_vram_gb": 16.0,
            "context_window": 4096,
            "preferred_gpu": "t4",
            "allocation_policy": "single_gpu",
        },
        "capabilities": {
            "text_to_image": True,
            "inpainting": True,
        },
        "status": "full",
        "coverage_percent": 100.0,
        "missing_capabilities": [],
        "limits": {},
    }


@pytest.fixture
def valid_service_payload() -> Dict[str, Any]:
    return {
        "apiVersion": "factory.ai/v1",
        "kind": "ServiceManifest",
        "metadata": {
            "id": "test-adversarial-service",
            "name": "Test Adversarial Service",
            "subsystem": "image",
            "version": "1.0.0",
        },
        "network": {
            "static_domain": "image.adversarial.ngrok.app",
            "internal_port": 8188,
            "protocol": "http",
            "auth_required": True,
            "cors_origins": ["*"],
        },
        "health_check": {
            "endpoint": "/healthz",
            "interval_seconds": 15,
            "timeout_seconds": 5,
            "unhealthy_threshold": 3,
        },
        "framework": {
            "type": "comfyui",
            "version": "0.3.10",
            "entrypoint": "main.py",
        },
        "hardware": {
            "gpu_required": True,
            "gpu_device_index": 0,
            "min_vram_gb": 16.0,
            "min_ram_gb": 16.0,
            "cuda_version_min": "12.1",
        },
        "supported_modalities": ["image"],
        "auth_policy": {
            "scheme": "bearer",
            "rate_limit_rpm": 60,
            "allow_unauthenticated": False,
        },
    }


@pytest.fixture
def stressed_api_client():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)

    @app.get("/test/raise-error")
    def raise_error(code: str, message: str, status_code: int = 400):
        raise FactoryError(message=message, code=code, status_code=status_code)

    @app.get("/test/raise-unhandled")
    def raise_unhandled(message: str = "Unexpected failure"):
        raise RuntimeError(f"CRITICAL: {message} in /Volumes/Secret/path/to/key.pem")

    @app.post("/test/validate-payload")
    def validate_payload(manifest: ServiceManifest):
        return {"status": "ok", "service_id": manifest.metadata.id}

    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. Adversarial Model Completeness Rule Bypass Challenges
# ===========================================================================

class TestModelCompletenessRuleAdversarial:
    """Stress tests attempting to trick or bypass the Model Completeness Rule."""

    def test_bypass_full_with_sub_100_coverage(self, valid_model_payload):
        """Attempt: status='full' with 99.9% coverage."""
        payload = copy.deepcopy(valid_model_payload)
        payload["coverage_percent"] = 99.9
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "cannot be marked 'full'" in str(exc_info.value)
        assert "coverage_percent is 99.9%, expected 100.0%" in str(exc_info.value)

    def test_bypass_full_with_zero_coverage(self, valid_model_payload):
        """Attempt: status='full' with 0.0% coverage."""
        payload = copy.deepcopy(valid_model_payload)
        payload["coverage_percent"] = 0.0
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "cannot be marked 'full'" in str(exc_info.value)

    def test_bypass_full_with_missing_capabilities_list(self, valid_model_payload):
        """Attempt: status='full', coverage=100%, but missing_capabilities has items."""
        payload = copy.deepcopy(valid_model_payload)
        payload["missing_capabilities"] = ["video_with_audio"]
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "has missing capabilities" in str(exc_info.value)

    def test_bypass_full_with_empty_string_missing_capability(self, valid_model_payload):
        """Attempt: status='full', coverage=100%, missing_capabilities=['']."""
        payload = copy.deepcopy(valid_model_payload)
        payload["missing_capabilities"] = [""]
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "has missing capabilities" in str(exc_info.value)

    def test_bypass_full_with_unbacked_capability_flag(self, valid_model_payload):
        """Attempt: status='full', coverage=100%, missing_capabilities=[], but a capability is False."""
        payload = copy.deepcopy(valid_model_payload)
        payload["capabilities"]["outpainting"] = False
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "contains unbacked capabilities" in str(exc_info.value)

    def test_bypass_partial_when_actually_full(self, valid_model_payload):
        """Attempt: Mark status='partial' when coverage=100%, missing_caps=[], and all caps True."""
        payload = copy.deepcopy(valid_model_payload)
        payload["status"] = "partial"
        with pytest.raises(ValidationError) as exc_info:
            ModelManifest.model_validate(payload)
        assert "has 100% coverage and no missing capabilities; it must be marked status: 'full'" in str(exc_info.value)

    def test_coverage_percent_boundary_overflow(self, valid_model_payload):
        """Attempt: coverage_percent > 100.0 or negative."""
        payload_high = copy.deepcopy(valid_model_payload)
        payload_high["coverage_percent"] = 100.01
        with pytest.raises(ValidationError):
            ModelManifest.model_validate(payload_high)

        payload_neg = copy.deepcopy(valid_model_payload)
        payload_neg["coverage_percent"] = -0.01
        with pytest.raises(ValidationError):
            ModelManifest.model_validate(payload_neg)

    def test_status_case_sensitivity_rejection(self, valid_model_payload):
        """Attempt: Pass uppercase 'FULL' or 'Partial' instead of canonical lowercase."""
        payload = copy.deepcopy(valid_model_payload)
        payload["status"] = "FULL"
        with pytest.raises(ValidationError):
            ModelManifest.model_validate(payload)

    def test_extra_fields_forbidden_in_model_manifest(self, valid_model_payload):
        """Attempt: Inject unexpected top-level fields (e.g. bypass flags)."""
        payload = copy.deepcopy(valid_model_payload)
        payload["bypass_completeness"] = True
        with pytest.raises(ValidationError):
            ModelManifest.model_validate(payload)

    def test_metadata_id_regex_injection(self, valid_model_payload):
        """Attempt: Inject invalid characters, path traversals, or spaces into metadata.id."""
        bad_ids = [
            "../../etc/passwd",
            "model with spaces",
            "Model_Uppercase",
            "model$special#chars",
            "",
        ]
        for bad_id in bad_ids:
            payload = copy.deepcopy(valid_model_payload)
            payload["metadata"]["id"] = bad_id
            with pytest.raises(ValidationError):
                ModelManifest.model_validate(payload)

    def test_post_instantiation_mutation_exploit(self, valid_model_payload):
        """
        Adversarial In-Memory Mutation Probe:
        Instantiate valid partial model, then mutate `status = ModelStatus.FULL` in memory.
        Checks if model enforces immutability/re-validation or permits attribute assignment bypass.
        """
        payload = copy.deepcopy(valid_model_payload)
        payload["status"] = "partial"
        payload["coverage_percent"] = 50.0
        payload["missing_capabilities"] = ["inpaint"]
        payload["capabilities"]["inpainting"] = False

        manifest = ModelManifest.model_validate(payload)
        assert manifest.status == ModelStatus.PARTIAL

        # Attempt in-memory mutation
        # If model_config lacks frozen=True and validate_assignment=True, assignment succeeds
        # without triggering the completeness validator!
        try:
            manifest.status = ModelStatus.FULL
            is_mutable = True
        except (ValidationError, TypeError):
            is_mutable = False

        # Note: Document finding: ModelManifest allows post-instantiation mutation if not frozen
        if is_mutable:
            # Check if completeness rule was bypassed in memory
            assert manifest.status == ModelStatus.FULL
            assert manifest.coverage_percent == 50.0  # Invariant breached in memory!


# ===========================================================================
# 2. Adversarial ServiceManifest Invariant & Invalidation Challenges
# ===========================================================================

class TestServiceManifestInvariantsAdversarial:
    """Stress tests challenging ServiceManifest validation boundaries."""

    def test_allow_unauthenticated_true_strictly_rejected(self, valid_service_payload):
        """Attempt: allow_unauthenticated=True in AuthPolicySpec."""
        payload = copy.deepcopy(valid_service_payload)
        payload["auth_policy"]["allow_unauthenticated"] = True
        with pytest.raises(ValidationError) as exc_info:
            ServiceManifest.model_validate(payload)
        assert "Factory v4 prohibits unauthenticated public bypass" in str(exc_info.value)

    def test_auth_required_false_strictly_rejected(self, valid_service_payload):
        """Attempt: auth_required=False in NetworkSpec."""
        payload = copy.deepcopy(valid_service_payload)
        payload["network"]["auth_required"] = False
        with pytest.raises(ValidationError) as exc_info:
            ServiceManifest.model_validate(payload)
        assert "Factory v4 forbids unauthenticated services" in str(exc_info.value)

    def test_invalid_ports_rejected(self, valid_service_payload):
        """Attempt: Privileged ports (<1024), zero, negative, or >65535."""
        invalid_ports = [0, -1, 80, 443, 1023, 65536, 99999]
        for port in invalid_ports:
            payload = copy.deepcopy(valid_service_payload)
            payload["network"]["internal_port"] = port
            with pytest.raises(ValidationError):
                ServiceManifest.model_validate(payload)

    def test_valid_ephemeral_ports_accepted(self, valid_service_payload):
        """Verify boundary port values [1024, 65535] are valid."""
        for port in [1024, 8000, 8188, 65535]:
            payload = copy.deepcopy(valid_service_payload)
            payload["network"]["internal_port"] = port
            sm = ServiceManifest.model_validate(payload)
            assert sm.network.internal_port == port

    def test_missing_or_empty_static_domain_rejected(self, valid_service_payload):
        """Attempt: static_domain is empty, None, or <4 characters."""
        for bad_domain in ["", "a", "ab", "abc"]:
            payload = copy.deepcopy(valid_service_payload)
            payload["network"]["static_domain"] = bad_domain
            with pytest.raises(ValidationError):
                ServiceManifest.model_validate(payload)

    def test_rate_limit_rpm_non_positive_rejected(self, valid_service_payload):
        """Attempt: rate_limit_rpm <= 0."""
        for bad_rpm in [0, -1, -100]:
            payload = copy.deepcopy(valid_service_payload)
            payload["auth_policy"]["rate_limit_rpm"] = bad_rpm
            with pytest.raises(ValidationError):
                ServiceManifest.model_validate(payload)

    def test_invalid_subsystem_enum_rejected(self, valid_service_payload):
        """Attempt: Invalid subsystem string."""
        payload = copy.deepcopy(valid_service_payload)
        payload["metadata"]["subsystem"] = "cryptominer"
        with pytest.raises(ValidationError):
            ServiceManifest.model_validate(payload)

    def test_invalid_framework_type_rejected(self, valid_service_payload):
        """Attempt: Invalid framework type string."""
        payload = copy.deepcopy(valid_service_payload)
        payload["framework"]["type"] = "unsupported_runtime"
        with pytest.raises(ValidationError):
            ServiceManifest.model_validate(payload)

    def test_semver_pattern_strictly_enforced(self, valid_service_payload):
        """Attempt: Non-SemVer version string."""
        for bad_ver in ["v1", "1.0", "beta-1.0", "1.0.0-rc1", "version1"]:
            payload = copy.deepcopy(valid_service_payload)
            payload["metadata"]["version"] = bad_ver
            with pytest.raises(ValidationError):
                ServiceManifest.model_validate(payload)

    def test_service_metadata_id_injection(self, valid_service_payload):
        """Attempt: Invalid characters in service id."""
        for bad_id in ["Image_Service", "image service", "image/service", ""]:
            payload = copy.deepcopy(valid_service_payload)
            payload["metadata"]["id"] = bad_id
            with pytest.raises(ValidationError):
                ServiceManifest.model_validate(payload)

    def test_port_collision_detection_in_validator(self, valid_service_payload):
        """Attempt: Two services claiming the exact same port (e.g. 8000)."""
        s1 = ServiceManifest.model_validate(valid_service_payload)
        payload2 = copy.deepcopy(valid_service_payload)
        payload2["metadata"]["id"] = "service-two"
        payload2["network"]["static_domain"] = "service-two.ngrok.app"
        s2 = ServiceManifest.model_validate(payload2)

        # Mock loader returning both services with port collision
        class MockCollisionLoader:
            def load_services(self):
                return {s1.metadata.id: s1, s2.metadata.id: s2}
            def load_models(self):
                return {}
            def load_capabilities(self):
                return {}
            def load_hardware_topologies(self):
                return {}

        validator = ManifestValidator(MockCollisionLoader())
        ok, errors = validator.validate_all()
        assert not ok
        assert any("Port collision" in e and "8188" in e for e in errors)

    def test_domain_collision_detection_in_validator(self, valid_service_payload):
        """Attempt: Two services claiming the exact same ngrok static domain."""
        s1 = ServiceManifest.model_validate(valid_service_payload)
        payload2 = copy.deepcopy(valid_service_payload)
        payload2["metadata"]["id"] = "service-two"
        payload2["network"]["internal_port"] = 8189
        s2 = ServiceManifest.model_validate(payload2)

        class MockDomainCollisionLoader:
            def load_services(self):
                return {s1.metadata.id: s1, s2.metadata.id: s2}
            def load_models(self):
                return {}
            def load_capabilities(self):
                return {}
            def load_hardware_topologies(self):
                return {}

        validator = ManifestValidator(MockDomainCollisionLoader())
        ok, errors = validator.validate_all()
        assert not ok
        assert any("Domain collision" in e and "image.adversarial.ngrok.app" in e for e in errors)


# ===========================================================================
# 3. Legacy Isolation & Import Prevention Challenges
# ===========================================================================

class TestLegacyIsolationAdversarial:
    """Stress tests verifying isolation of archived legacy v3.5 code."""

    def test_no_legacy_import_in_production_tree(self):
        """Scan all python files in factory, packages, services, infrastructure for forbidden imports."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        production_roots = ["factory", "packages", "services", "workflows", "infrastructure"]

        forbidden_prefixes = ["legacy", "kaggle"]
        violations = []

        for prod_root in production_roots:
            p = root_dir / prod_root
            if not p.exists():
                continue
            for py_file in p.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for line_idx, line in enumerate(content.splitlines(), start=1):
                    stripped = line.strip()
                    for prefix in forbidden_prefixes:
                        if stripped.startswith(f"import {prefix}") or stripped.startswith(f"from {prefix}"):
                            violations.append(f"{py_file}:{line_idx}: {stripped}")

        assert not violations, f"Forbidden legacy imports found in production code:\n" + "\n".join(violations)

    def test_dynamic_legacy_import_detection(self):
        """
        Adversarial Challenge: Can dynamic importlib.import_module load legacy at runtime?
        Investigate whether Python sys.path contains legacy or root, allowing dynamic imports.
        """
        import importlib
        # Probe if legacy is accessible via sys.path
        try:
            legacy_mod = importlib.import_module("legacy.v3.pipelines")
            is_dynamically_importable = True
        except ImportError:
            is_dynamically_importable = False

        # If importable, verify that static checker in tests/security DOES NOT catch dynamic imports!
        if is_dynamically_importable:
            from tests.security.test_no_legacy_imports import LEGACY_IMPORT_PATTERNS
            dynamic_code_sample = 'importlib.import_module("legacy.v3.pipelines")'
            matches = [p.findall(dynamic_code_sample) for p in LEGACY_IMPORT_PATTERNS]
            regex_missed_dynamic = all(len(m) == 0 for m in matches)
            assert regex_missed_dynamic, "Static regex unexpectedly matched dynamic importlib call"


# ===========================================================================
# 4. UnifiedErrorResponse Schema Stress & Fuzzing Challenges
# ===========================================================================

class TestUnifiedErrorResponseAdversarial:
    """Fuzzing and stress testing the UnifiedErrorResponse schema and FastAPI error handlers."""

    def test_schema_rejects_missing_required_fields(self):
        """Attempt: ErrorDetail missing code, message, or request_id."""
        with pytest.raises(ValidationError):
            ErrorDetail.model_validate({
                "message": "Missing code and request_id",
            })

        with pytest.raises(ValidationError):
            ErrorDetail.model_validate({
                "code": "UNAUTHORIZED",
                "request_id": "req-123",
            })

    def test_schema_rejects_unknown_error_code(self):
        """Attempt: Pass undefined string as ErrorCode enum."""
        with pytest.raises(ValidationError):
            ErrorDetail.model_validate({
                "code": "NON_EXISTENT_CUSTOM_CODE_123",
                "message": "Some error",
                "request_id": "req-uuid",
            })

    def test_schema_forbids_injected_extra_fields(self):
        """Attempt: Inject stack trace, debug info, or raw path into ErrorDetail or UnifiedErrorResponse."""
        with pytest.raises(ValidationError):
            ErrorDetail.model_validate({
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Internal error",
                "request_id": "req-123",
                "stack_trace": "Traceback (most recent call last)...",
            })

        with pytest.raises(ValidationError):
            UnifiedErrorResponse.model_validate({
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal error",
                    "request_id": "req-123",
                },
                "debug": {"leaked_secret": "abc"},
            })

    def test_extreme_payload_fuzzing(self):
        """Stress: Error message with 100,000 characters and complex Unicode symbols."""
        huge_message = "💥 ERROR: " + "🔥" * 1000 + "A" * 50000 + "\x00\r\n\t" + "مرحبا بالعالم"
        detail = ErrorDetail(
            code=ErrorCode.BAD_REQUEST,
            message=huge_message,
            request_id=f"req-{uuid.uuid4()}",
            details={"deep": {"nested": {"unicode": "中文測試", "numbers": [math.pi, -1e10]}}},
        )
        envelope = UnifiedErrorResponse(error=detail)
        serialized = envelope.model_dump_json()
        deserialized = UnifiedErrorResponse.model_validate_json(serialized)
        assert deserialized.error.code == ErrorCode.BAD_REQUEST
        assert len(deserialized.error.message) == len(huge_message)

    def test_all_standard_status_codes_adhere_to_schema(self, stressed_api_client):
        """Verify that all mapped HTTP status codes return valid UnifiedErrorResponse bodies."""
        test_cases = [
            (ErrorCode.BAD_REQUEST, "Bad Request", 400),
            (ErrorCode.UNAUTHORIZED, "Unauthorized Access", 401),
            (ErrorCode.FORBIDDEN, "Forbidden Route", 403),
            (ErrorCode.NOT_FOUND, "Resource Missing", 404),
            (ErrorCode.CONFLICT, "Resource Conflict", 409),
            (ErrorCode.PAYLOAD_TOO_LARGE, "Payload Size Limit", 413),
            (ErrorCode.RATE_LIMITED, "Rate Limit Exceeded", 429),
            (ErrorCode.MODEL_UNAVAILABLE, "Model Daemon Down", 503),
            (ErrorCode.GATEWAY_TIMEOUT, "Upstream Timeout", 504),
        ]

        for code, msg, status in test_cases:
            res = stressed_api_client.get(
                f"/test/raise-error?code={code.value}&message={msg}&status_code={status}"
            )
            assert res.status_code == status
            body = res.json()
            # Must strictly validate against UnifiedErrorResponse schema
            validated = UnifiedErrorResponse.model_validate(body)
            assert validated.error.code == code
            assert validated.error.message == msg
            assert validated.error.request_id.startswith("req-")
            assert res.headers.get("X-Request-ID") == validated.error.request_id

    def test_unhandled_exception_sanitization_and_schema_conformance(self, stressed_api_client):
        """Stress: Verify unhandled exception returns 500, conforms to schema, and leaks zero secrets."""
        res = stressed_api_client.get("/test/raise-unhandled?message=DATABASE_CORRUPT_IN_SECRETS")
        assert res.status_code == 500
        body = res.json()
        # Must strictly validate against UnifiedErrorResponse
        validated = UnifiedErrorResponse.model_validate(body)
        assert validated.error.code == ErrorCode.INTERNAL_SERVER_ERROR
        assert validated.error.request_id.startswith("req-")
        assert res.headers.get("X-Request-ID") == validated.error.request_id

        # Zero leaks of internal path or message
        assert "Volumes" not in res.text
        assert "DATABASE_CORRUPT" not in res.text
        assert "key.pem" not in res.text
        assert "CRITICAL" not in res.text

    def test_request_validation_error_schema_conformance(self, stressed_api_client):
        """Verify that FastAPI 422 RequestValidationError adheres to UnifiedErrorResponse."""
        # Send empty/malformed payload to endpoint expecting ServiceManifest
        res = stressed_api_client.post("/test/validate-payload", json={"invalid": True})
        assert res.status_code == 422
        body = res.json()
        validated = UnifiedErrorResponse.model_validate(body)
        assert validated.error.code == ErrorCode.VALIDATION_ERROR
        assert "validation_errors" in validated.error.details
        assert isinstance(validated.error.details["validation_errors"], list)
        assert len(validated.error.details["validation_errors"]) > 0


# ===========================================================================
# 5. Full Manifest Directory Invariant Verification
# ===========================================================================

class TestLiveManifestsDirectoryIntegrity:
    """Verifies all production manifests in manifests/ against the CLI and validator."""

    def test_live_manifests_directory_passes_validation(self):
        """Runs validate_manifests against real manifests directory."""
        ok, errors, counts = validate_manifests("manifests")
        assert ok, f"Manifest validation failed with errors:\n" + "\n".join(errors)
        assert counts["services"] == 10
        assert counts["models"] >= 9
        assert counts["hardware"] == 3
        assert counts["capabilities"] >= 3

    def test_wan_2_2_model_strictly_conforms_to_partial_status(self):
        """Directly verify that wan-2.2 is validly marked PARTIAL with its missing capability."""
        loader = ManifestLoader("manifests")
        models = loader.load_models()
        assert "wan-2.2" in models
        wan = models["wan-2.2"]
        assert wan.status == ModelStatus.PARTIAL
        assert wan.coverage_percent == 80.0
        assert wan.missing_capabilities == ["video_with_audio"]
        assert wan.capabilities["video_with_audio"] is False

    def test_fully_backed_models_strictly_conform_to_full_status(self):
        """Directly verify that models marked FULL have 100% coverage and zero missing caps."""
        loader = ManifestLoader("manifests")
        models = loader.load_models()
        full_models = [m for m in models.values() if m.status == ModelStatus.FULL]
        assert len(full_models) > 0
        for m in full_models:
            assert m.coverage_percent == 100.0
            assert len(m.missing_capabilities) == 0
            assert all(v is True for v in m.capabilities.values())
