"""Phase 6 official SDK wire acceptance. Explicit mocks, never inference evidence."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.engine_refactor_gate import evaluate as engine_gate
from deployments.kaggle.backend_lifecycle_probe import source_bundle
from model_registry.deployment import ModelCatalog, DeploymentRegistry
from factory_contracts.deployment import DeploymentConfig

SDK_VERSION = "3.49.0"
REQUIRED = {
    "/v1/responses": {"post"},
    "/v1/chat/completions": {"post"},
    "/v1/embeddings": {"post"},
    "/v1/audio/transcriptions": {"post"},
    "/v1/audio/speech": {"post"},
    "/v1/images/generations": {"post"},
    "/v1/images/edits": {"post"},
    "/v1/videos": {"post"},
    "/v1/videos/edits": {"post"},
    "/v1/videos/extensions": {"post"},
    "/v1/videos/{identifier}": {"get"},
    "/v1/videos/{identifier}/content": {"get"},
    "/v1/models": {"get"},
    "/v1/decisions": {"post"},
    "/v1/rerank": {"post"},
    "/v1/music": {"post"},
    "/v1/music/{identifier}": {"get"},
    "/v1/music/{identifier}/content": {"get"},
    "/v1/enhancements": {"post"},
    "/v1/enhancements/{identifier}": {"get"},
}
SUITES = ["tests/contract/test_openai_sdk_api.py", "tests/contract/test_phase6_native_contracts.py"]


def evaluate():
    findings = []
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            findings.append(name)

    prior = engine_gate()
    check("prior_phases_current_source", prior["exit_gate"] == "PASS")
    if prior["exit_gate"] != "PASS":
        findings.extend(prior["findings"])
    sdk = None
    receipt = {}
    routes = []
    try:
        import openai

        sdk = importlib.metadata.version("openai")
        check("pinned_official_sdk", sdk == SDK_VERSION)
        from factory.apps.control_api.src.main import app

        paths = app.openapi()["paths"]
        for path, methods in REQUIRED.items():
            passed = path in paths and methods <= set(paths[path])
            routes.append({"path": path, "methods": sorted(methods), "registered": passed})
        check("all_master_plan_routes_mounted", all(row["registered"] for row in routes))
        catalog = ModelCatalog(ROOT / "manifests")
        registry = DeploymentRegistry(DeploymentConfig(), catalog)
        check(
            "zero_static_ready_and_zero_public_activation",
            registry.active_models() == ()
            and all(
                m.validation.state.value != "READY"
                and not any(c.live_verified for c in m.capabilities_truth.values())
                for m in catalog.models().values()
            ),
        )
        with tempfile.TemporaryDirectory(prefix="factory-sdk-gate-") as temp:
            junit = Path(temp) / "sdk.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                *SUITES,
                "-q",
                "-o",
                "addopts=",
                f"--junitxml={junit}",
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
            counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
            cases = []
            if junit.exists():
                tree = ET.parse(junit)
                for suite in tree.iter("testsuite"):
                    for key in counts:
                        counts[key] += int(suite.attrib.get(key, 0))
                cases = [case.attrib["name"] for case in tree.iter("testcase")]
            check(
                "official_sdk_and_native_contracts_executed",
                result.returncode == 0
                and counts["tests"] >= 54
                and not any(counts[k] for k in ("failures", "errors", "skipped")),
            )
            needed = [
                "test_sdk_chat_and_responses",
                "test_sdk_chat_and_responses_streams",
                "test_sdk_embeddings_float_and_default_base64",
                "test_sdk_audio_upload_and_waveform",
                "test_sdk_image_generate_and_multipart_edits",
                "test_sdk_video_create_retrieve_content_extension",
                "test_sdk_decisions_and_models",
                "test_async_sdk",
                "test_public_decision_batch_preserves_question_semantics_and_one_usage",
                "test_custom_music_jobs_and_disabled_enhancement",
                "test_custom_rerank_uses_canonical_engine",
            ]
            check("all_supported_families_exercised", set(needed) <= set(cases))
            check(
                "unvalidated_sessions_exercised",
                len(
                    [
                        c
                        for c in cases
                        if c.startswith("test_unvalidated_session_fails_closed_through_sdk[")
                    ]
                )
                == 6,
            )
            receipt = {
                "classification": "SOFTWARE_ONLY_EXPLICIT_MOCK_AUTHORIZATION_AND_ADAPTER_OUTPUTS",
                "command": command,
                "exit_code": result.returncode,
                "counts": counts,
                "cases": cases,
                "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest()
                if junit.is_file()
                else None,
                "sources": [
                    {
                        "file": str(path),
                        "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                    }
                    for path in SUITES
                ],
                "diagnostics": None
                if result.returncode == 0
                else (result.stdout + result.stderr)[-6000:],
            }
    except Exception as exc:
        findings.append(f"{type(exc).__name__}: {exc}")
    return {
        "phase": 6,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_gate": "PASS" if not findings else "NOT_MET",
        "classification": "OFFICIAL_PYTHON_SDK_SOFTWARE_COMPATIBILITY_ONLY",
        "sdk_version": sdk,
        "source_bundle_sha256": hashlib.sha256(source_bundle()).hexdigest(),
        "checks": checks,
        "routes": routes,
        "acceptance": receipt,
        "findings": findings,
        "prior_engine_gate": prior["exit_gate"],
        "scope": "Unmodified official sync/async Python SDK against actual localhost HTTP routers and canonical Engines with explicit test-only authorization/adapter doubles. Existing video IDs are tested over HTTP because SDK 3.49.0 fails dictionary video extraction client-side. Video editing remains unsupported by current manifests.",
        "production_ready": False,
        "public_models": [],
        "native_inference_executed_by_this_gate": False,
        "semantic_validation_executed": False,
        "full_factory_e2e_executed": False,
        "limitations": [
            "Supported API subset only; unsupported options return explicit errors",
            "Hosted Videos API is shut down; compatibility targets retained historical SDK wire contract",
            "All public model capabilities remain evidence-locked pending Native V3 + semantic + Full Factory E2E authorization",
            "No OpenAI-hosted inference/API-key use; SDK uses Factory local credentials",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/openai_api_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("exit_gate", "classification", "sdk_version", "findings")}
        )
    )
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
