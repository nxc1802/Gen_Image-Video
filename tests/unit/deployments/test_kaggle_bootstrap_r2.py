"""
tests/unit/deployments/test_kaggle_bootstrap_r2.py
Comprehensive unit tests for Milestone R2: Kaggle Bootstrap & Model Truth.

Verifies:
1. Manifest reconciliation across all 11 models, hardware topology, and launcher scripts.
2. Weight shard parsing from index JSON and rejection of marker-only downloads.
3. Component validation for multi-component models.
4. Rejection of zero-byte / corrupt files.
5. Model path namespacing under models_dir / model_id / ... preventing collisions.
6. Pure dry-run behavior for --check-only (zero filesystem or environment mutations).
7. Active backend health probing gating readiness on live responsive endpoints.
8. Tesla T4 hardware constraints (14.2 GB allocatable limit, YuE2 Gate Y0).
"""

import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict
import pytest
import yaml

from deployments.kaggle.bootstrap import (
    DATASET_SPECS,
    MAX_ALLOCATABLE_VRAM_PER_GPU_MB,
    SERVICE_BACKEND_MAP,
    HardwareReport,
    KaggleBootstrap,
)
from deployments.kaggle.prepare_datasets import MODEL_DATASETS


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_manifest_reconciliation_all_11_models():
    """Validates that all 11 model manifests exist and match hardware & service topology."""
    models_dir = WORKSPACE_ROOT / "manifests" / "models"
    model_manifests = list(models_dir.glob("*.yaml"))
    assert len(model_manifests) == 11, f"Expected 11 model manifests, found {len(model_manifests)}"

    # Load hardware topology
    hw_path = WORKSPACE_ROOT / "manifests" / "hardware" / "kaggle-dual-t4.yaml"
    with open(hw_path, "r", encoding="utf-8") as f:
        hw_data = yaml.safe_load(f)

    gpu0_services = hw_data["gpus"][0]["assigned_services"]
    gpu1_services = hw_data["gpus"][1]["assigned_services"]

    # Verify each model
    for mf in model_manifests:
        with open(mf, "r", encoding="utf-8") as f:
            m = yaml.safe_load(f)
        mid = m["metadata"]["id"]
        sid = m["metadata"]["service_id"]
        pref_gpu = m["hardware"]["preferred_gpu_index"]

        if pref_gpu == 0:
            assert sid in gpu0_services, f"Model {mid} preferred GPU 0 but {sid} not in GPU 0 services"
        elif pref_gpu == 1:
            assert sid in gpu1_services, f"Model {mid} preferred GPU 1 but {sid} not in GPU 1 services"

    # Verify VLM launcher uses GPU 0
    vlm_sh_path = WORKSPACE_ROOT / "services" / "vlm" / "launcher" / "run_kaggle.sh"
    content = vlm_sh_path.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0" in content, "VLM launcher run_kaggle.sh must use CUDA_VISIBLE_DEVICES=0"
    assert "CUDA_VISIBLE_DEVICES=1" not in content

    # Verify MODEL_ENGINE_MAP documents YuE2 on port 8009 and ACE-Step on port 8008
    engine_map_path = WORKSPACE_ROOT / "MODEL_ENGINE_MAP.md"
    engine_map_text = engine_map_path.read_text(encoding="utf-8")
    assert "8009" in engine_map_text, "MODEL_ENGINE_MAP.md must document YuE2 port 8009"
    assert "8008" in engine_map_text, "MODEL_ENGINE_MAP.md must document ACE-Step port 8008"


def test_prepare_datasets_manifest_alignment():
    """Validates that prepare_datasets.py contains all 11 models aligned with manifests."""
    model_ids = [item["id"] for item in MODEL_DATASETS]
    assert "yue2" in model_ids, "prepare_datasets.py must include yue2"
    assert "qwen-text" in model_ids
    assert "qwen-vlm" in model_ids

    # Verify AWQ checkpoints (not stale 27B GGUF)
    qwen_text = next(item for item in MODEL_DATASETS if item["id"] == "qwen-text")
    assert "AWQ" in qwen_text["repo"], "qwen-text in prepare_datasets must use AWQ checkpoint"
    assert "27b" not in qwen_text["slug"].lower(), "qwen-text must not use stale 27B GGUF"

    qwen_vlm = next(item for item in MODEL_DATASETS if item["id"] == "qwen-vlm")
    assert "AWQ" in qwen_vlm["repo"], "qwen-vlm in prepare_datasets must use AWQ checkpoint"


def test_bootstrap_model_path_namespacing():
    """Verifies that discover_and_mount_datasets creates namespaced paths to prevent collisions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        models_dir = Path(tmp_dir) / "models"
        input_dir.mkdir(parents=True)

        # Setup 2 datasets sharing filename "model.safetensors"
        # 1. qwen-decision
        dec_dir = input_dir / "qwen25-3b-instruct"
        dec_dir.mkdir(parents=True)
        (dec_dir / "model.safetensors").write_bytes(b"DECISION_WEIGHTS_MOCK")

        # 2. ace-step-1.5
        ace_dir = input_dir / "acestep-1-5"
        ace_dir.mkdir(parents=True)
        (ace_dir / "model.safetensors").write_bytes(b"ACESTEP_WEIGHTS_MOCK")

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir, models_dir=models_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=True)

        # Check namespaced destination paths
        dec_link = models_dir / "qwen-decision" / "model.safetensors"
        ace_link = models_dir / "ace-step-1.5" / "model.safetensors"

        assert dec_link.exists(), "qwen-decision weight link must exist under models/qwen-decision/"
        assert ace_link.exists(), "ace-step-1.5 weight link must exist under models/ace-step-1.5/"
        assert dec_link.read_bytes() == b"DECISION_WEIGHTS_MOCK"
        assert ace_link.read_bytes() == b"ACESTEP_WEIGHTS_MOCK"


def test_bootstrap_rejects_marker_only_safetensors_index():
    """Asserts that index.json alone without shard files is flagged incomplete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        qwen_text_dir = input_dir / "qwen25-7b-instruct-awq"
        qwen_text_dir.mkdir(parents=True)

        # Write index JSON with 2 declared shards, but do NOT create shard files
        index_data = {
            "metadata": {"format": "pt"},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            },
        }
        (qwen_text_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=False)

        res = info["discovered"]["qwen-text"]
        assert res["complete"] is False, "Dataset must NOT be complete when shards are missing"
        assert len(res["corrupted_files"]) > 0 or len(res["missing_files"]) > 0


def test_bootstrap_sharded_safetensors_validation_success():
    """Verifies that when all constituent safetensors shards exist, complete is True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        qwen_text_dir = input_dir / "qwen25-7b-instruct-awq"
        qwen_text_dir.mkdir(parents=True)

        index_data = {
            "metadata": {"format": "pt"},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            },
        }
        (qwen_text_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))
        (qwen_text_dir / "model-00001-of-00002.safetensors").write_bytes(b"SHARD1_BYTES")
        (qwen_text_dir / "model-00002-of-00002.safetensors").write_bytes(b"SHARD2_BYTES")

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=False)

        res = info["discovered"]["qwen-text"]
        assert res["complete"] is True, "Dataset must be complete when all shards exist and > 0 bytes"
        assert "model-00001-of-00002.safetensors" in res["found_files"]
        assert "model-00002-of-00002.safetensors" in res["found_files"]


def test_bootstrap_rejects_zero_byte_files():
    """Asserts that 0-byte weight files are rejected as corrupt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        ace_dir = input_dir / "acestep-1-5"
        ace_dir.mkdir(parents=True)

        # Create 0-byte file
        (ace_dir / "model.safetensors").write_bytes(b"")

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=False)

        res = info["discovered"]["ace-step-1.5"]
        assert res["complete"] is False, "0-byte weight file must cause complete=False"
        assert any("zero_bytes" in err for err in res["corrupted_files"])


def test_bootstrap_diffusers_rejects_marker_only():
    """Asserts that diffusers model with model_index.json but no weight files fails validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        wan_dir = input_dir / "wan22-ti2v-5b-diffusers"
        wan_dir.mkdir(parents=True)

        # Only model_index.json
        (wan_dir / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=False)

        res = info["discovered"]["wan-2.2"]
        assert res["complete"] is False, "Diffusers model must not be complete without actual weights"


def test_bootstrap_validates_declared_components():
    """Asserts that models with declared components verify all constituent component weights."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "kaggle_input"
        img_dir = input_dir / "qwen-image-2-1-gguf"
        img_dir.mkdir(parents=True)

        # Create primary GGUF but omit components (clip, vae, pose, depth)
        (img_dir / "qwen-image-2.1-Q4_K_M.gguf").write_bytes(b"GGUF_MOCK_BYTES")

        bootstrapper = KaggleBootstrap(kaggle_input_dir=input_dir)
        info = bootstrapper.discover_and_mount_datasets(create_symlinks=False)

        res = info["discovered"]["qwen-image-2.1"]
        assert res["complete"] is False, "Model must be incomplete when declared components are missing"
        assert "clip" in res["missing_components"]
        assert "vae" in res["missing_components"]

        # Now add the component weights
        (img_dir / "qwen3vl_8b_fp8_scaled.safetensors").write_bytes(b"CLIP_BYTES")
        (img_dir / "qwen_image_2.1_vae_bf16.safetensors").write_bytes(b"VAE_BYTES")
        (img_dir / "qwen_image_control_pose.safetensors").write_bytes(b"POSE_BYTES")
        (img_dir / "qwen_image_control_depth.safetensors").write_bytes(b"DEPTH_BYTES")

        info2 = bootstrapper.discover_and_mount_datasets(create_symlinks=False)
        res2 = info2["discovered"]["qwen-image-2.1"]
        assert res2["complete"] is True, "Model must be complete when all components are present and non-empty"


def test_bootstrap_check_only_pure_dry_run_zero_mutation():
    """Asserts that check_only is a pure dry-run that creates no files, no dirs, and mutates no env vars."""
    orig_env = dict(os.environ)

    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch_dir = Path(tmp_dir) / "scratch_test"
        models_dir = Path(tmp_dir) / "models_test"
        input_dir = Path(tmp_dir) / "input_test"

        # Ensure cwd does not create bootstrap_summary.json
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            bootstrapper = KaggleBootstrap(
                kaggle_input_dir=input_dir,
                scratch_artifacts_dir=scratch_dir,
                models_dir=models_dir,
            )

            res = bootstrapper.run_bootstrap(check_only=True, install_missing=True)
            assert res.storage.get("dry_run") is True

            # Verify no files or dirs created
            assert not scratch_dir.exists(), "Scratch dir must not be created during check_only"
            assert not models_dir.exists(), "Models dir must not be created during check_only"
            assert not (Path(tmp_dir) / "bootstrap_summary.json").exists(), (
                "bootstrap_summary.json must not be written during check_only unless explicitly specified"
            )

            # Verify os.environ was not mutated
            for key in ["STORAGE_BACKEND", "SCRATCH_DIR", "FACTORY_DB_BACKEND"]:
                if key not in orig_env:
                    assert key not in os.environ, f"check_only must not mutate os.environ['{key}']"
        finally:
            os.chdir(orig_cwd)


def test_kaggle_dual_t4_vram_and_yue2_feasibility():
    """Verifies that unquantized YuE2 7B BF16 is marked HARDWARE_BLOCKED on Tesla T4."""
    bootstrapper = KaggleBootstrap()
    hw = HardwareReport(
        platform_system="Linux",
        platform_release="6.1.0",
        python_version="3.12.0",
        cpu_count=4,
        system_ram_gb=32.0,
        scratch_disk_free_gb=50.0,
        cuda_available=True,
        gpu_count=2,
        gpu_names=["Tesla T4", "Tesla T4"],
        vram_per_gpu_mb=[15360.0, 15360.0],
        dual_t4_detected=True,
        max_allocatable_vram_mb=MAX_ALLOCATABLE_VRAM_PER_GPU_MB,
    )

    # 1. Unquantized YuE2 (Track A)
    res_unquant = bootstrapper.check_hardware_feasibility(hw, model_id="yue2", quantization="none")
    assert res_unquant["status"] == "HARDWARE_BLOCKED"
    assert res_unquant["feasible"] is False
    assert "HARDWARE_BLOCKED" in res_unquant["reason"]

    # 2. Track B Quantized YuE2
    res_quant = bootstrapper.check_hardware_feasibility(hw, model_id="yue2", quantization="awq")
    assert res_quant["status"] == "FEASIBLE"
    assert res_quant["feasible"] is True


def test_bootstrap_backend_health_probing():
    """Verifies that model readiness is gated on native backend response."""
    bootstrapper = KaggleBootstrap()

    # When backend is offline on random port
    readiness_offline = bootstrapper.check_model_readiness(
        model_id="qwen-decision",
        dataset_entry={"complete": True},
        timeout=0.1,
    )
    assert readiness_offline["ready"] is False
    assert readiness_offline["status"] == "BACKEND_UNAVAILABLE"

    # Spin up a temporary mock HTTP server on an available port
    class MockHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), MockHandler)
    mock_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Probe our active mock server
        probe = bootstrapper.probe_backend("127.0.0.1", mock_port, endpoint="/health", timeout=1.0)
        assert probe["online"] is True
        assert probe["status_code"] == 200

        # Now pass this probe to check_model_readiness
        readiness_online = bootstrapper.check_model_readiness(
            model_id="qwen-decision",
            dataset_entry={"complete": True},
            backend_probe=probe,
        )
        assert readiness_online["ready"] is True
        assert readiness_online["status"] == "VERIFIED"
    finally:
        server.shutdown()
