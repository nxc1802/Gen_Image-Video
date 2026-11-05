"""
tests/adversarial/test_adversarial_gate4_challenger.py
Adversarial Challenger Test Suite for Gate 4 & Gate 5:
1. P2.1 & Gate 4: Model Registry GGUF Variant Specifications
   - ModelVariantSpec schema validation, strict extra="forbid", validation constraints.
   - Manifest loading of manifests/models/qwen-image-2.1.yaml with variants.turbo.
   - GGUF format, q4_k_m quantization, 8 Euler steps, linked dataset cuongnguyen1802/qwen-image-2-1-turbo-gguf.
   - factory_contracts.models.qwen_image authoritative constants.
2. Canonical Model Normalization:
   - "qwen-image-2.1-turbo", "qwen-image-turbo", "qwen_image_2.1_turbo" -> "qwen-image-2.1".
   - Model to family resolution -> "qwen-image".
   - get_model_aliases includes all turbo variants.
3. Live Kaggle Dataset Pruning:
   - Pruning of stale 404 datasets in session1_visual_pipeline.py and monitor_both_sessions.py.
   - Active mount of cuongnguyen1802/qwen-image-2-1-turbo-gguf.
4. Physical Artifact Attestation (Gate 5 / R5):
   - Cryptographic SHA256 verification of actual visual outputs on disk against visual_pipeline_report.json.
   - Authentic non-synthetic execution verification (fallback_used == False).
5. Scratch Disk Cleanup Isolation & Regression Prevention:
   - scratch_disk.config.cleanup_on_download default integrity.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Dict

from pydantic import ValidationError
import pytest

from factory_contracts import (
    ModelManifest,
    ModelVariantSpec,
    ModelWeightFormat,
    ModelWeightsSpec,
    PrecisionType,
    QuantizationType,
)
from factory_contracts.models import (
    QWEN_IMAGE_2_1_TURBO_GGUF,
    QWEN_IMAGE_VARIANTS,
)
from factory_contracts.tasks import (
    CANONICAL_MODEL_MAP,
    get_model_aliases,
    model_to_family,
    normalize_model,
)
from model_registry.loader import ManifestLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = REPO_ROOT / "manifests"
VISUAL_OUTPUTS_DIR = REPO_ROOT / "deployments" / "kaggle" / "visual_outputs"


# ==============================================================================
# 1. P2.1 & Gate 4: Model Registry GGUF Variant Specifications
# ==============================================================================

class TestModelRegistryGGUFVariants:
    """Verifies ModelVariantSpec schema contracts, validation rules, and manifest registration."""

    def test_model_variant_spec_valid_construction(self):
        """Constructs a valid ModelVariantSpec and asserts all fields."""
        variant = ModelVariantSpec(
            id="turbo",
            display_name="Qwen-Image 2.1 Turbo GGUF (8 Euler steps)",
            checkpoint="qwen-image-2.1-turbo-Q4_K_M.gguf",
            format=ModelWeightFormat.GGUF,
            quantization=QuantizationType.Q4_K_M,
            default_precision=PrecisionType.BF16,
            recommended_steps=8,
            kaggle_dataset="cuongnguyen1802/qwen-image-2-1-turbo-gguf",
            huggingface_repo="unsloth/Qwen-Image-2.1-Turbo-GGUF",
            file_name="qwen-image-2.1-turbo-Q4_K_M.gguf",
            sha256="a" * 64,
            size_bytes=4199565120,
            min_vram_gb=14.0,
        )
        assert variant.id == "turbo"
        assert variant.format == ModelWeightFormat.GGUF
        assert variant.quantization == QuantizationType.Q4_K_M
        assert variant.recommended_steps == 8
        assert variant.kaggle_dataset == "cuongnguyen1802/qwen-image-2-1-turbo-gguf"
        assert variant.min_vram_gb == 14.0

    def test_model_variant_spec_forbids_extra_fields(self):
        """ModelVariantSpec must forbid extra attributes."""
        with pytest.raises(ValidationError) as exc_info:
            ModelVariantSpec(
                id="turbo",
                display_name="Turbo",
                checkpoint="turbo.gguf",
                unsupported_extra_attribute="illegal_value",
            )
        assert "extra_forbidden" in str(exc_info.value)

    def test_model_variant_spec_id_pattern_validation(self):
        """Variant ID must match ^[a-z0-9-._]+$."""
        # Invalid uppercase or spaces
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="TURBO UPPERCASE",
                display_name="Turbo",
                checkpoint="turbo.gguf",
            )
        # Invalid special chars
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="turbo@v1",
                display_name="Turbo",
                checkpoint="turbo.gguf",
            )

    def test_model_variant_spec_recommended_steps_range(self):
        """Recommended steps must be between 1 and 100."""
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="turbo",
                display_name="Turbo",
                checkpoint="turbo.gguf",
                recommended_steps=0,
            )
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="turbo",
                display_name="Turbo",
                checkpoint="turbo.gguf",
                recommended_steps=101,
            )

    def test_model_variant_spec_sha256_pattern(self):
        """SHA256 must match ^[a-f0-9]{64}$."""
        # Too short
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="turbo",
                display_name="Turbo",
                checkpoint="turbo.gguf",
                sha256="abc123",
            )
        # Invalid chars
        with pytest.raises(ValidationError):
            ModelVariantSpec(
                id="turbo",
                display_name="Turbo",
                checkpoint="turbo.gguf",
                sha256="z" * 64,
            )

    def test_model_weights_spec_variants_dictionary(self):
        """ModelWeightsSpec accepts valid variants dictionary and defaults to empty."""
        weights = ModelWeightsSpec(primary_checkpoint="test/repo")
        assert weights.variants == {}

        variant = ModelVariantSpec(
            id="q4_k_m",
            display_name="Q4_K_M GGUF",
            checkpoint="model-q4.gguf",
        )
        weights_with_variants = ModelWeightsSpec(
            primary_checkpoint="test/repo",
            variants={"q4_k_m": variant},
        )
        assert "q4_k_m" in weights_with_variants.variants
        assert weights_with_variants.variants["q4_k_m"].checkpoint == "model-q4.gguf"

    def test_manifest_loads_qwen_image_turbo_variant(self):
        """ManifestLoader successfully loads manifests/models/qwen-image-2.1.yaml with turbo variant."""
        loader = ManifestLoader(MANIFESTS_DIR)
        models = loader.load_models()
        assert "qwen-image-2.1" in models, "qwen-image-2.1 model manifest not found"

        qwen = models["qwen-image-2.1"]
        assert hasattr(qwen.weights, "variants"), "weights.variants missing from ModelManifest"
        assert "turbo" in qwen.weights.variants, "'turbo' variant not declared in weights.variants"

        turbo = qwen.weights.variants["turbo"]
        assert turbo.id == "turbo"
        assert turbo.format == ModelWeightFormat.GGUF
        assert turbo.quantization == QuantizationType.Q4_K_M
        assert turbo.recommended_steps == 8
        assert turbo.kaggle_dataset == "cuongnguyen1802/qwen-image-2-1-turbo-gguf"
        assert turbo.huggingface_repo == "unsloth/Qwen-Image-2.1-Turbo-GGUF"
        assert turbo.file_name == "qwen-image-2.1-turbo-Q4_K_M.gguf"
        assert turbo.size_bytes == 4199565120
        assert turbo.min_vram_gb == 14.0

    def test_authoritative_qwen_image_python_constants(self):
        """Validates QWEN_IMAGE_2_1_TURBO_GGUF and QWEN_IMAGE_VARIANTS in factory_contracts.models."""
        assert QWEN_IMAGE_2_1_TURBO_GGUF.id == "turbo"
        assert QWEN_IMAGE_2_1_TURBO_GGUF.format == ModelWeightFormat.GGUF
        assert QWEN_IMAGE_2_1_TURBO_GGUF.quantization == QuantizationType.Q4_K_M
        assert QWEN_IMAGE_2_1_TURBO_GGUF.recommended_steps == 8
        assert QWEN_IMAGE_2_1_TURBO_GGUF.kaggle_dataset == "cuongnguyen1802/qwen-image-2-1-turbo-gguf"
        assert "turbo" in QWEN_IMAGE_VARIANTS
        assert QWEN_IMAGE_VARIANTS["turbo"] is QWEN_IMAGE_2_1_TURBO_GGUF


# ==============================================================================
# 2. Canonical Model Normalization & Alias Mapping
# ==============================================================================

class TestCanonicalModelNormalization:
    """Verifies that turbo model aliases resolve canonically and map to family."""

    @pytest.mark.parametrize(
        "alias,expected_canonical",
        [
            ("qwen-image-2.1-turbo", "qwen-image-2.1"),
            ("qwen-image-turbo", "qwen-image-2.1"),
            ("qwen_image_2.1_turbo", "qwen-image-2.1"),
            ("  QWEN-IMAGE-2.1-TURBO  ", "qwen-image-2.1"),
            ("QWEN_IMAGE_2.1_TURBO", "qwen-image-2.1"),
            ("qwen-image", "qwen-image-2.1"),
            ("qwen-image-2.1", "qwen-image-2.1"),
        ],
    )
    def test_normalize_model_turbo_aliases(self, alias: str, expected_canonical: str):
        """All turbo alias variants normalize to 'qwen-image-2.1'."""
        assert normalize_model(alias) == expected_canonical

    @pytest.mark.parametrize(
        "alias",
        [
            "qwen-image-2.1-turbo",
            "qwen-image-turbo",
            "qwen_image_2.1_turbo",
            "qwen-image",
            "qwen-image-2.1",
        ],
    )
    def test_model_to_family_turbo_aliases(self, alias: str):
        """All turbo aliases resolve to the 'qwen-image' family."""
        assert model_to_family(alias) == "qwen-image"

    def test_get_model_aliases_contains_turbo(self):
        """get_model_aliases('qwen-image-2.1') includes all turbo variants."""
        aliases = get_model_aliases("qwen-image-2.1")
        assert "qwen-image-2.1-turbo" in aliases
        assert "qwen-image-turbo" in aliases
        assert "qwen_image_2.1_turbo" in aliases


# ==============================================================================
# 3. Live Kaggle Dataset Pruning & Active Configuration
# ==============================================================================

class TestKaggleDatasetPruningAndActiveConfig:
    """Verifies that the 4 stale 404 datasets were pruned from dataset_sources."""

    def test_session1_visual_pipeline_dataset_sources_pruned(self):
        """Asserts that session1_visual_pipeline.py dataset_sources does not contain 404 datasets."""
        pipeline_file = REPO_ROOT / "deployments" / "kaggle" / "session1_visual_pipeline.py"
        assert pipeline_file.exists(), f"{pipeline_file} does not exist"

        content = pipeline_file.read_text(encoding="utf-8")
        stale_datasets = [
            "nguynxuncngde180528/qwen-image-2-1-gguf",
            "asdwddjksa/minimax-h3-diffusion",
            "asdwddjksa/minimax-h3-text-encoder",
            "asdwddjksa/minimax-h3-assets",
        ]

        # Extract dataset_sources list definition in run_session_1
        match = re.search(r"dataset_sources\s*=\s*\[(.*?)\]", content, re.DOTALL)
        assert match is not None, "dataset_sources list not found in session1_visual_pipeline.py"
        dataset_sources_text = match.group(1)

        for stale in stale_datasets:
            assert stale not in dataset_sources_text, f"Stale 404 dataset '{stale}' still present in dataset_sources"

        # Assert active turbo dataset is present
        assert "qwen-image-2-1-turbo-gguf" in dataset_sources_text, (
            "qwen-image-2-1-turbo-gguf dataset missing from dataset_sources"
        )

    def test_monitor_both_sessions_dataset_sources_pruned(self):
        """Asserts that monitor_both_sessions.py visual dataset_sources does not contain 404 datasets."""
        monitor_file = REPO_ROOT / "deployments" / "kaggle" / "monitor_both_sessions.py"
        assert monitor_file.exists()

        content = monitor_file.read_text(encoding="utf-8")
        stale_datasets = [
            "nguynxuncngde180528/qwen-image-2-1-gguf",
            "asdwddjksa/minimax-h3-diffusion",
            "asdwddjksa/minimax-h3-text-encoder",
            "asdwddjksa/minimax-h3-assets",
        ]

        match = re.search(r'"visual":\s*\{.*?"dataset_sources":\s*\[(.*?)\]', content, re.DOTALL)
        assert match is not None, "visual dataset_sources not found in monitor_both_sessions.py"
        visual_sources_text = match.group(1)

        for stale in stale_datasets:
            assert stale not in visual_sources_text, f"Stale 404 dataset '{stale}' found in monitor_both_sessions.py"

        assert "qwen-image-2-1-turbo-gguf" in visual_sources_text


# ==============================================================================
# 4. Gate 5: Physical Artifact Attestation & Non-Synthetic Verification
# ==============================================================================

class TestPhysicalVisualOutputsAttestation:
    """Verifies that physical artifacts match cryptographic hashes in visual_pipeline_report.json."""

    @pytest.fixture
    def pipeline_report(self) -> Dict:
        report_file = VISUAL_OUTPUTS_DIR / "visual_pipeline_report.json"
        assert report_file.exists(), f"{report_file} does not exist"
        return json.loads(report_file.read_text(encoding="utf-8"))

    def test_physical_artifacts_exist_on_disk(self):
        """Asserts that all three core visual outputs physically exist on disk."""
        image_path = VISUAL_OUTPUTS_DIR / "studio_qwen_image_master.png"
        video_path = VISUAL_OUTPUTS_DIR / "studio_minimax_h3_dragon_video.mp4"
        prompt_path = VISUAL_OUTPUTS_DIR / "qwen38_generated_prompt.txt"

        assert image_path.exists(), "studio_qwen_image_master.png missing from visual_outputs"
        assert video_path.exists(), "studio_minimax_h3_dragon_video.mp4 missing from visual_outputs"
        assert prompt_path.exists(), "qwen38_generated_prompt.txt missing from visual_outputs"

        # Check substantial file sizes (not 0 bytes or mock strings)
        assert image_path.stat().st_size > 1_000_000, "studio_qwen_image_master.png is smaller than 1MB"
        assert video_path.stat().st_size > 700_000, "studio_minimax_h3_dragon_video.mp4 is smaller than 700KB"
        assert prompt_path.stat().st_size > 50, "qwen38_generated_prompt.txt is smaller than 50 bytes"

    def test_sha256_checksums_match_attestation_report(self, pipeline_report: Dict):
        """Verifies physical file SHA256 matches exact hash in visual_pipeline_report.json."""
        image_path = VISUAL_OUTPUTS_DIR / "studio_qwen_image_master.png"
        video_path = VISUAL_OUTPUTS_DIR / "studio_minimax_h3_dragon_video.mp4"
        prompt_path = VISUAL_OUTPUTS_DIR / "qwen38_generated_prompt.txt"

        img_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        vid_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
        prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()

        results = pipeline_report["results"]
        expected_img_sha = results["qwen_image_2.1"]["artifact_sha256"]
        expected_vid_sha = results["minimax_h3"]["artifact_sha256"]
        expected_prompt_sha = results["qwen38_27b_vlm"]["artifact_sha256"]

        assert img_sha == expected_img_sha, f"Image SHA mismatch: {img_sha} vs {expected_img_sha}"
        assert vid_sha == expected_vid_sha, f"Video SHA mismatch: {vid_sha} vs {expected_vid_sha}"
        assert prompt_sha == expected_prompt_sha, f"Prompt SHA mismatch: {prompt_sha} vs {expected_prompt_sha}"

    def test_report_attests_fallback_integrity_and_zero_vlm_false_positives(self, pipeline_report: Dict):
        """Verifies report honestly attributes fallbacks: image/video are authentic, and VLM fallback is not falsely declared as passed."""
        results = pipeline_report["results"]

        # Visual generators must be authentic without fallback
        for stage_name in ["qwen_image_2.1", "minimax_h3"]:
            stage_data = results[stage_name]
            assert stage_data.get("status") == "passed", f"Stage '{stage_name}' status is not 'passed'"
            assert stage_data.get("fallback_used") is False, f"Stage '{stage_name}' used an unexpected fallback!"

        # VLM: Prompt generation was choked by unclosed <think> reasoning; must be honestly declared as fallback
        vlm_data = results["qwen38_27b_vlm"]
        assert vlm_data.get("status") == "fallback", "VLM stage must report 'fallback' when reasoning choked"
        assert vlm_data.get("fallback_used") is True, "VLM stage must declare fallback_used == True"
        assert not (vlm_data.get("status") == "passed" and vlm_data.get("fallback_used") is False), (
            "Forbidden regression: VLM fallback artifact declared as genuine passed inference!"
        )

        # Top-level status must reflect overall fallback state
        assert pipeline_report.get("status") == "fallback", "Top-level status must be 'fallback'"
        assert pipeline_report.get("fallback_used") is True, "Top-level report must declare fallback_used == True"


# ==============================================================================
# 5. Scratch Disk Isolation & Regression Prevention
# ==============================================================================

class TestScratchDiskIsolationRegressionPrevention:
    """Verifies scratch disk cleanup does not leak global state to concurrent downloads."""

    def test_scratch_disk_default_cleanup_is_false(self):
        """Verifies ScratchDiskConfig defaults cleanup_on_download to False."""
        from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig
        config = ScratchDiskConfig()
        assert config.cleanup_on_download is False, (
            "ScratchDiskConfig.cleanup_on_download must default to False to protect concurrent downloads"
        )


class TestGateRegressionPrevention:
    """Verifies that key invariants across Gates 1-3 remain completely unregressed."""

    def test_gate1_boto3_lazy_import_integrity(self):
        """Verifies boto3 and botocore are not imported at top-level of s3.py."""
        s3_file = REPO_ROOT / "factory" / "apps" / "control-api" / "src" / "storage" / "s3.py"
        assert s3_file.exists()
        tree = ast.parse(s3_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "boto" not in alias.name, f"Top-level import '{alias.name}' violates Gate 1"
            elif isinstance(node, ast.ImportFrom):
                assert "boto" not in (node.module or ""), f"Top-level import from '{node.module}' violates Gate 1"

    def test_gate2_task_canonicalization_integrity(self):
        """Verifies canonical tasks resolve correctly without regress."""
        from factory_contracts.tasks import normalize_task, task_to_capability_id
        assert normalize_task("t2i") == "text_to_image"
        assert normalize_task("text_to_image") == "text_to_image"
        assert task_to_capability_id("text_to_image") == "image.t2i"

    def test_gate3_required_phase_keys_integrity(self):
        """Verifies required 6-phase keys remain standard."""
        from deployments.kaggle.certify_kernel import REQUIRED_PHASE_KEYS
        expected_phases = {
            "model_loading_ms",
            "prompt_processing_ms",
            "decode_ms",
            "sampling_ms",
            "vae_ms",
            "artifact_encoding_ms",
        }
        assert set(REQUIRED_PHASE_KEYS) == expected_phases
