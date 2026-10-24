"""
tests/unit/contracts/test_schemas.py
Unit tests verifying Pydantic v2 schemas and the Model Completeness Rule
"""
import pytest
from pydantic import ValidationError

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
from factory_contracts.manifests.capability import (
    CapabilityManifest,
    ParameterSpec,
    ParameterDataType,
    OutputSpec,
    OutputDataType,
    ValidationSpec,
)
from factory_contracts.manifests.hardware import (
    HardwareTopologyManifest,
    GPUDeviceSpec,
    HostResourcesSpec,
)
from factory_contracts.errors.error_contract import (
    UnifiedErrorResponse,
    ErrorDetail,
    ErrorCode,
)
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobState,
    JobSubmissionRequest,
)


class TestServiceManifest:
    def test_valid_service_manifest(self):
        manifest = ServiceManifest(
            apiVersion="factory.ai/v1",
            kind="ServiceManifest",
            metadata=ServiceMetadata(
                id="test-service",
                name="Test Generation Service",
                subsystem=SubsystemType.IMAGE,
                version="1.0.0",
            ),
            network=NetworkSpec(
                static_domain="test.ai.example.com",
                internal_port=8188,
                protocol=ProtocolType.HTTP,
                auth_required=True,
                cors_origins=["*"],
            ),
            health_check=HealthCheckSpec(
                endpoint="/healthz",
                interval_seconds=15,
                timeout_seconds=5,
                unhealthy_threshold=3,
            ),
            framework=FrameworkSpec(
                type=FrameworkType.COMFYUI,
                version="0.3.0",
                entrypoint="python main.py",
            ),
            hardware=HardwareRequirements(
                gpu_required=True,
                gpu_device_index=0,
                min_vram_gb=14.0,
                min_ram_gb=16.0,
            ),
            supported_modalities=[ModalityType.IMAGE],
            auth_policy=AuthPolicySpec(
                scheme="bearer",
                rate_limit_rpm=60,
                allow_unauthenticated=False,
            ),
        )
        assert manifest.metadata.id == "test-service"
        assert manifest.network.auth_required is True

    def test_service_manifest_rejects_unauthenticated_bypass(self):
        with pytest.raises(ValidationError):
            AuthPolicySpec(
                scheme="bearer",
                rate_limit_rpm=60,
                allow_unauthenticated=True,  # Forbidden in v4
            )

    def test_service_manifest_rejects_auth_required_false(self):
        with pytest.raises(ValidationError):
            NetworkSpec(
                static_domain="test.ai.example.com",
                internal_port=8188,
                protocol=ProtocolType.HTTP,
                auth_required=False,  # Forbidden in v4
            )


class TestModelCompletenessRule:
    def test_full_model_family_valid(self):
        manifest = ModelManifest(
            apiVersion="factory.ai/v1",
            kind="ModelManifest",
            metadata=ModelMetadata(
                id="qwen-image-2.1",
                family="qwen-image",
                display_name="Qwen-Image 2.1 Family",
                service_id="image-service",
                version="2.1.0",
            ),
            weights=ModelWeightsSpec(
                primary_checkpoint="abenzerps/Qwen-Image-2.1-GGUF",
                format=ModelWeightFormat.GGUF,
                quantization=QuantizationType.Q4_K_M,
                default_precision=PrecisionType.BF16,
            ),
            hardware=ModelHardwareSpec(
                min_vram_gb=14.0,
                preferred_gpu="t4",
                allocation_policy=AllocationPolicy.SINGLE_GPU,
            ),
            capabilities={
                "text_to_image": True,
                "image_to_image": True,
                "inpaint": True,
            },
            status=ModelStatus.FULL,
            coverage_percent=100.0,
            missing_capabilities=[],
        )
        assert manifest.status == ModelStatus.FULL
        assert manifest.coverage_percent == 100.0

    def test_full_model_rejected_if_coverage_under_100(self):
        with pytest.raises(ValidationError, match="cannot be marked 'full'"):
            ModelManifest(
                apiVersion="factory.ai/v1",
                kind="ModelManifest",
                metadata=ModelMetadata(
                    id="incomplete-model",
                    family="incomplete",
                    display_name="Incomplete Model",
                    service_id="test-service",
                    version="1.0.0",
                ),
                weights=ModelWeightsSpec(
                    primary_checkpoint="repo/checkpoint",
                ),
                hardware=ModelHardwareSpec(min_vram_gb=8.0),
                capabilities={
                    "text_to_image": True,
                    "inpaint": False,
                },
                status=ModelStatus.FULL,  # Invariant violation: coverage is 50%
                coverage_percent=50.0,
                missing_capabilities=["inpaint"],
            )

    def test_full_model_rejected_if_missing_capabilities_present(self):
        with pytest.raises(ValidationError, match="has missing capabilities"):
            ModelManifest(
                apiVersion="factory.ai/v1",
                kind="ModelManifest",
                metadata=ModelMetadata(
                    id="fake-full-model",
                    family="fake",
                    display_name="Fake Full Model",
                    service_id="test-service",
                    version="1.0.0",
                ),
                weights=ModelWeightsSpec(
                    primary_checkpoint="repo/checkpoint",
                ),
                hardware=ModelHardwareSpec(min_vram_gb=8.0),
                capabilities={
                    "text_to_image": True,
                },
                status=ModelStatus.FULL,
                coverage_percent=100.0,
                missing_capabilities=["unbacked_feature"],  # Invariant violation
            )

    def test_partial_model_valid(self):
        manifest = ModelManifest(
            apiVersion="factory.ai/v1",
            kind="ModelManifest",
            metadata=ModelMetadata(
                id="wan-2.2",
                family="wan-video",
                display_name="Wan 2.2 Video Family",
                service_id="video-service",
                version="2.2.0",
            ),
            weights=ModelWeightsSpec(
                primary_checkpoint="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                format=ModelWeightFormat.DIFFUSERS,
                quantization=QuantizationType.FOUR_BIT,
            ),
            hardware=ModelHardwareSpec(min_vram_gb=14.0),
            capabilities={
                "text_to_video": True,
                "video_with_audio": False,
            },
            status=ModelStatus.PARTIAL,
            coverage_percent=50.0,
            missing_capabilities=["video_with_audio"],
        )
        assert manifest.status == ModelStatus.PARTIAL
        assert manifest.coverage_percent == 50.0

    def test_partial_model_rejected_if_it_is_actually_100_percent_full(self):
        with pytest.raises(ValidationError, match="must be marked status: 'full'"):
            ModelManifest(
                apiVersion="factory.ai/v1",
                kind="ModelManifest",
                metadata=ModelMetadata(
                    id="wan-complete",
                    family="wan-video",
                    display_name="Wan Complete",
                    service_id="video-service",
                    version="2.2.0",
                ),
                weights=ModelWeightsSpec(
                    primary_checkpoint="Wan-AI/Wan2.2",
                ),
                hardware=ModelHardwareSpec(min_vram_gb=14.0),
                capabilities={
                    "text_to_video": True,
                },
                status=ModelStatus.PARTIAL,  # Invariant violation: 100% coverage, must be FULL
                coverage_percent=100.0,
                missing_capabilities=[],
            )


class TestUnifiedErrorResponse:
    def test_unified_error_response_valid(self):
        err = UnifiedErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.UNAUTHORIZED,
                message="Missing Authorization header",
                request_id="req-12345",
                details={"reason": "token_missing"},
            )
        )
        assert err.error.code == ErrorCode.UNAUTHORIZED
        assert err.error.request_id == "req-12345"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            UnifiedErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Missing header",
                    request_id="req-12345",
                ),
                extra_field="disallowed",
            )


class TestFactoryJobContract:
    def test_job_submission_and_factory_job(self):
        sub = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "A scenic mountain"},
            priority=10,
        )
        assert sub.service == "image"
        assert sub.priority == 10

        job = FactoryJob(
            job_id="job-11111111-1111-1111-1111-111111111111",
            request_id="req-22222222-2222-2222-2222-222222222222",
            service=sub.service,
            model=sub.model,
            task=sub.task,
            inputs=sub.inputs,
            state=JobState.CREATED,
        )
        assert job.state == JobState.CREATED
        assert job.progress.stage == "queued"
        assert job.retry_count == 0
