"""Phase 1 model truth gates. Software-only; never native inference evidence."""
import json
import hashlib
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from factory_contracts.errors import FactoryError
from factory_contracts.jobs.job_contract import JobSubmissionRequest
from factory_contracts.manifests.catalog import load_model_manifests
from factory_contracts.manifests.model import ModelManifest
from factory_contracts.tasks import normalize_model
from model_registry.cli import factory_main
from model_registry.loader import ManifestLoader

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "manifests"


def test_all_model_truth_is_pinned_without_static_readiness():
    for model in load_model_manifests(CATALOG).values():
        assert model.apiVersion == "factory.ai/v2"
        assert model.engine and model.backend and model.runtime_profiles
        assert model.weights.files and "primary" in model.weights.files
        assert not model.weights.variants and not model.task_variant_matrix
        assert not model.installed and not model.backend_ready and not model.checkpoint_valid
        assert all(not c.live_verified for c in model.capabilities_truth.values())
        # Implementation progresses after Phase 1; static code never grants live
        # verification or production readiness.
        assert {cid for cid, truth in model.capabilities_truth.items() if truth.implemented} == set(model.validation.implemented_capabilities)
        assert model.validation.state.value != "READY"


def test_inspector_provides_complete_static_truth_for_entire_catalog(capsys):
    for mid in load_model_manifests(CATALOG):
        assert factory_main(["--manifests-dir", str(CATALOG), "models", "inspect", mid]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["canonical_truth"]["complete"]
        assert result["checkpoint"]["files"]["primary"]["sha256"]
        assert result["engine"] and result["backend"] and result["validation"]
        assert "kaggle-dual-t4" in result["runtime_profiles"]
        assert not result["current_session"]["attested"]
        assert not result["current_session"]["production_ready"]


def test_different_checkpoint_bundles_have_distinct_ids():
    models = load_model_manifests(CATALOG)
    assert "qwen-speech" not in models
    assert models["qwen-asr"].weights.repository != models["qwen-tts"].weights.repository
    assert models["qwen-embedding"].weights.repository != models["qwen-reranker"].weights.repository
    assert len({models[mid].weights.sha256 for mid in ["qwen-image-2.1", "qwen-image-2.1-turbo", "qwen-image-2.1-uncensored"]}) == 3
    assert models["minimax-h3"].weights.sha256 == models["minimax-h3-turbo"].weights.sha256
    assert len(models["minimax-h3-turbo"].weights.files) > len(models["minimax-h3"].weights.files)
    assert len({models[mid].weights.sha256 for mid in ["ace-step-1.5", "ace-step-1.5-sft", "ace-step-1.5-turbo", "ace-step-1.5-xl-base"]}) == 4
    assert models["yue2"].weights.repository == "m-a-p/YuE2-3B"


def test_aliases_do_not_choose_another_checkpoint():
    assert normalize_model("qwen-image-turbo") == "qwen-image-2.1-turbo"
    assert normalize_model("qwen3-asr") == "qwen-asr"
    assert normalize_model("qwen3-tts") == "qwen-tts"
    assert normalize_model("qwen-rerank") == "qwen-reranker"
    assert normalize_model("qwen-text-2.5") == "qwen-text-2.5"


def test_new_identity_is_manifest_only_and_ambiguous_aliases_fail(tmp_path, monkeypatch):
    data = yaml.safe_load((CATALOG / "models/qwen-text.yaml").read_text())
    data["metadata"]["id"] = "new-model"
    data["aliases"] = ["new-model-alias"]
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "one.yaml").write_text(yaml.safe_dump(data))
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(tmp_path))
    assert normalize_model("new-model-alias") == "new-model"
    data["metadata"]["id"] = "another-model"
    (directory / "two.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_model("new-model-alias")


def test_v2_rejects_legacy_flags_checkpoint_variants_and_ready_label():
    data = yaml.safe_load((CATALOG / "models/qwen-text.yaml").read_text())
    for field, value in [("installed", True), ("task_variant_matrix", {"chat_completion": ["turbo"]}), ("backend_ready", True)]:
        with pytest.raises(ValueError):
            ModelManifest.model_validate({**data, field: value})
    data["validation"]["state"] = "READY"
    with pytest.raises(ValueError, match="cannot grant READY"):
        ModelManifest.model_validate(data)


def test_runtime_profile_cannot_choose_an_undeclared_backend_or_share_dual_gpu():
    data = yaml.safe_load((CATALOG / "models/qwen-text.yaml").read_text())
    profile = data["runtime_profiles"]["kaggle-dual-t4"]
    profile["preferred_backend"] = "undeclared_backend"
    with pytest.raises(ValueError, match="preferred backend"):
        ModelManifest.model_validate(data)
    profile["preferred_backend"] = data["backend"]["supported"][0]
    profile["exclusive"] = False
    with pytest.raises(ValueError, match="exclusive"):
        ModelManifest.model_validate(data)


def test_canonical_admission_fails_closed_without_deployment_attestation():
    from factory.apps.control_api.src.api.routers.jobs import _validate_job_capability
    request = MagicMock()
    request.app.state.manifest_loader = ManifestLoader(CATALOG)
    job = JobSubmissionRequest(service="image", model="qwen-image-2.1", task="text_to_image", inputs={"prompt": "a cat"})
    with pytest.raises(FactoryError) as error:
        _validate_job_capability(job, request)
    assert error.value.code == "MODEL_NOT_LOADED"
    job.inputs["variant"] = "turbo"
    with pytest.raises(FactoryError) as error:
        _validate_job_capability(job, request)
    assert error.value.code == "UNSUPPORTED_CAPABILITY"


@pytest.mark.parametrize("contents,verified", [(b"weights", True), (b"garbage", False)])
def test_bootstrap_checks_pinned_bytes_not_existence(tmp_path, monkeypatch, contents, verified):
    from deployments.kaggle.bootstrap import KaggleBootstrap
    data = yaml.safe_load((CATALOG / "models/qwen-text.yaml").read_text())
    data["metadata"]["id"] = "file-integrity-test"
    data["aliases"] = []
    primary = data["weights"]["files"]["primary"]
    primary.update(sha256=hashlib.sha256(b"weights").hexdigest(), size_bytes=7, destination="weights.gguf")
    catalog = tmp_path / "catalog/models"
    catalog.mkdir(parents=True)
    (catalog / "test.yaml").write_text(yaml.safe_dump(data))
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(catalog.parent))
    dataset = tmp_path / "input/file-integrity-test"
    dataset.mkdir(parents=True)
    (dataset / "weights.gguf").write_bytes(contents)
    bootstrap = KaggleBootstrap(kaggle_input_dir=dataset.parent, models_dir=tmp_path / "model-store")
    report = bootstrap.discover_and_mount_datasets(create_symlinks=True)["discovered"]["file-integrity-test"]
    assert report["complete"] is verified
    assert report["weights_verified"] is verified
    assert not report["model_inference_verified"]
    if not verified:
        assert not (tmp_path / "model-store/file-integrity-test/weights.gguf").exists()


def test_canonical_bootstrap_cannot_certify_model_from_shared_health_or_variant(monkeypatch):
    from deployments.kaggle.bootstrap import KaggleBootstrap
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(CATALOG))
    bootstrap = KaggleBootstrap()
    hardware = MagicMock()
    for mid in ["qwen-text", "yue2", "unknown-model"]:
        readiness = bootstrap.check_model_readiness(mid, dataset_entry={"complete": True, "weights_verified": True},
            hw_report=hardware, backend_probe={"online": True})
        assert not readiness["ready"] and not readiness["production_ready"]
        assert not readiness["model_inference_verified"]
    feasibility = bootstrap.check_hardware_feasibility(hardware, "yue2", variant="track_b", quantization="awq")
    assert not feasibility["feasible"]
    assert feasibility["requirements"]["preferred_backend"] == "yue2_native"


@pytest.mark.parametrize("module_name", [
    "services.text.launcher.run_text_service",
    "services.speech.launcher.run_speech_stack",
    "services.music.launcher.run_music_stack",
])
def test_legacy_launchers_do_not_select_another_checkpoint_or_backend(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr("sys.argv", ["legacy-launcher"])
    popen = MagicMock(side_effect=AssertionError("Unexpected process launch"))
    monkeypatch.setattr("subprocess.Popen", popen)
    with pytest.raises(RuntimeError, match="legacy|Legacy"):
        module.main()
    popen.assert_not_called()


def test_retired_composite_identity_requires_explicit_model_choice(capsys):
    assert factory_main(["--manifests-dir", str(CATALOG), "models", "inspect", "qwen-speech"]) == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "model_retired"
    assert error["replacements"] == ["qwen-asr", "qwen-tts"]


@pytest.mark.parametrize("corruption", ["stale_checkpoint", "missing_component"])
def test_phase1_gate_rejects_stale_or_incomplete_probe(tmp_path, corruption):
    from benchmarks.model_truth_gate import evaluate_catalog
    probe = json.loads((ROOT / "validation/checkpoint_metadata_current.json").read_text())
    observed = probe["models"][0]
    if corruption == "stale_checkpoint":
        observed["expected_sha256"] = "0" * 64
        expected_error = "probe_inventory_matches_model"
    else:
        observed["file_identities"].pop()
        expected_error = "complete_bundle_identity"
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    gate = evaluate_catalog(CATALOG, path)
    assert gate["exit_gate"] == "NOT_MET"
    assert f"{observed['model']}: {expected_error}" in gate["errors"]
    assert not gate["production_ready"]
