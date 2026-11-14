"""Truth guards: labels and unrelated capabilities must never grant evidence."""
from pathlib import Path

import yaml

from factory_contracts.manifests.model import ModelManifest

ROOT = Path(__file__).resolve().parents[2]


def manifest_payload():
    return yaml.safe_load((ROOT / "tests/fixtures/legacy_manifests/models/qwen-text.yaml").read_text())


def test_model_label_cannot_certify_capability_or_attest_runtime():
    payload = manifest_payload()
    payload.update(validation_status="live_verified", capabilities_truth={},
                   evidence_run_id="unrelated-historical-run", installed=False,
                   backend_ready=False, checkpoint_valid=False)
    model = ModelManifest.model_validate(payload)
    assert not model.installed
    assert not model.backend_ready
    assert not model.checkpoint_valid
    for capability in payload["capabilities"]:
        truth = model.get_capability_truth(capability)
        assert not truth.live_verified
        assert truth.evidence_run_id is None


def test_verified_capability_cannot_certify_new_capability():
    payload = manifest_payload()
    payload.update(validation_status="live_verified", evidence_run_id="one-capability-only")
    payload["capabilities_truth"]["chat_completion"].update(
        live_verified=True, evidence_run_id="one-capability-only")
    model = ModelManifest.model_validate(payload)
    assert model.get_capability_truth("chat_completion").live_verified
    assert not model.get_capability_truth("tool_calling").live_verified
    # Also exercise the fallback accessor, without invoking the validator again.
    model.capabilities_truth.pop("structured_output")
    assert not model.get_capability_truth("structured_output").live_verified
    assert model.get_capability_truth("structured_output").evidence_run_id is None


def test_checked_in_claims_do_not_certify_unproven_inference():
    for path in sorted((ROOT / "manifests/models").glob("*.yaml")):
        model = ModelManifest.model_validate(yaml.safe_load(path.read_text()))
        assert model.validation_status.value == "pending_live", path
        assert not model.validated_hardware_profiles, path
        assert model.evidence_run_id is None, path
        assert all(not truth.live_verified for truth in model.capabilities_truth.values()), path


def test_baseline_accounts_for_every_claim_and_rejects_non_inference_evidence():
    from benchmarks.truth_baseline import build_baseline

    report = build_baseline()
    pairs = [cap for model in report["models"] for cap in model["capabilities"]]
    assert len(pairs) == report["summary"]["model_capability_pairs"]
    assert all(cap["disposition"] in {"BLOCKED", "DISABLED"} for cap in pairs)
    assert all(cap["reason"] and cap["implementation_sources"] for cap in pairs)
    assert all(not cap["production_ready"] and not cap["accepted_evidence"] for cap in pairs)
    assert all(not e["accepted_for_V3"] and not e["accepted_for_V4"] for e in report["evidence_audit"])
    assert any(e["classification"] == "SYNTHETIC_TENSOR" for e in report["evidence_audit"])
    assert report["api_inventory"]["routes"]
    assert report["tests"]["files"]
