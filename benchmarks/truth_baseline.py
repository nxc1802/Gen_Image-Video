"""Phase 0 audit snapshot, never a runtime catalog or an inference certifier.

Read the frozen Git tree, not today's working tree. Markdown is derived from JSON.
Run: python benchmarks/truth_baseline.py [--check]
New source commits/models require a new explicit audit; they are not auto-certified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "e74f674fbfb6876b7b5899a7b64458801e3a683d"
OUTPUT = ROOT / "validation"

# Temporary migration audit findings, scoped to SOURCE_COMMIT. These are code
# observations, not model support declarations. Production must read manifests.
AUDIT = {
    "qwen-text": ("vllm", ["services/text/src/engine.py", "services/text/src/responses_handler.py"],
        "HTTP/streaming adapters exist. Manifest is Qwen3.8 GGUF; service config launches Qwen2.5 AWQ via vLLM. Shard filename omits UD. Exact checkpoint/revision/integrity and semantic evidence are missing."),
    "qwen-vlm": ("vllm", ["services/vlm/src/engine.py", "services/vlm/src/app.py"],
        "Multimodal HTTP/streaming boundary exists. Shard used Qwen3.8 GGUF, not declared Qwen2-VL AWQ. Responses uses shared text handler. Grounding/video/document semantics have no exact-model proof."),
    "qwen-decision": ("vllm", ["services/decision/src/engine.py", "services/decision/src/vllm_adapter.py"],
        "Predicate/choice/score delegated to upstream. In-memory golden path and contract PASS do not prove native decision inference."),
    "qwen-embedding": ("vllm / TEI (configured externally)", ["services/embedding/src/text_embed.py", "services/embedding/src/reranker.py", "services/embedding/src/multimodal_embed.py"],
        "HTTP embedding/rerank adapters exist. Config uses Qwen-Embedding/Qwen-Rerank while manifest uses Qwen3. Reranker component has no own revision/integrity. Multimodal is disabled; wrapper alone does not implement it."),
    "qwen-speech": ("Qwen ASR / vllm-omni TTS / external aligner", ["services/speech/asr/engine.py", "services/speech/tts/engine.py", "services/speech/alignment/aligner.py", "deployments/kaggle/session2_audio_pipeline.py"],
        "ASR/TTS HTTP boundaries exist; alignment disabled. Revoke TTS/cloning verification: historical runner catches errors and writes sine-wave WAV plus unrelated CUDA allocations. XTTS is not Qwen TTS. No admissible exact-model ASR/TTS/cloning evidence."),
    "qwen-image-2.1": ("comfyui", ["services/image/worker.py", "workflows/image/registry.py", "workflows/image/parameter_mappings.py"],
        "Worker/template path exists. Manifest reference_image/multi_reference/control_pose/control_depth do not have matching registry task keys. Standard manifest also embeds a different Turbo checkpoint. Turbo output does not certify Standard."),
    "qwen-image-2.1-turbo": ("comfyui", ["services/image/worker.py", "deployments/kaggle/live_factory_e2e_runner.py", "workflows/image/parameter_mappings.py"],
        "Turbo T2I runner/workflow exists; worker advertises Standard ID. Historical report claims native execution but lacks pinned verified weight/input/semantic evidence. No V4 through target deployment registry/adapter path."),
    "wan-2.2": ("comfyui", ["services/video/worker.py", "workflows/video/registry.py", "workflows/video/templates/wan_t2v_a14b.json"],
        "Worker/templates exist. Manifest targets TI2V-5B Diffusers, registry selects A14B T2V/I2V templates. Mounted checkpoint and MP4 container checks cannot certify inference."),
    "minimax-h3": ("comfyui", ["services/video/worker.py", "workflows/video/registry.py", "deployments/kaggle/session1_visual_pipeline.py"],
        "Worker/templates exist; historical FL2V report is a candidate only. Revision, component integrity, real inputs and semantic validation not bound in an admissible evidence record; other capabilities cannot inherit FL2V evidence."),
    "ace-step-1.5": ("ACE-Step native HTTP", ["services/music/acestep/adapter.py", "services/music/acestep/client.py", "services/music/worker.py"],
        "Task mapper/client/worker exist. Base/SFT/Turbo/XL are different weights behind variants. Specialized tasks require Base/XL. Tensor projections and contract checks do not certify music. Deprecated gateway returns 410."),
    "yue2": ("external YuE runtime HTTP", ["services/music/yue2/adapter.py", "services/music/yue2/client.py", "services/music/yue2/runtime.py", "services/music/worker.py"],
        "Task mapper/client/runtime wrapper exist. Score parsing/edit helpers are not neural song inference. Native exact checkpoint/GPU/backend/semantic evidence absent."),
    "enhancement-suite-v1": ("external enhancement HTTP", ["services/enhancement/backend.py", "services/enhancement/worker.py"],
        "Fail-closed HTTP scaffolding exists, but no complete shipped native enhancement runtime. All four capabilities disabled. Tensor resize/interpolation is not Real-ESRGAN/RIFE/DeepFilterNet inference."),
}

EVIDENCE = {
    "deployments/kaggle/certification_results.json": ("CONTRACT_SIMULATION", "Top-level execution_mode is pre-flight-contract-simulation, live_pass=false and telemetry UNAVAILABLE; nested certified=true is rejected."),
    "deployments/kaggle/live_hardware_attested_report.json": ("SYNTHETIC_TENSOR", "Top-level synthetic_tensor_benchmark / SYNTHETIC_PASS; hardware presence does not prove weight inference."),
    "benchmarks/reports/MASTER_DUAL_T4_BENCHMARK_REPORT.md": ("HISTORICAL_CLAIM", "Markdown claims are not §38 evidence records. Native L4 claims are not automatically V3; lifecycle/L5 claims are not automatically V4. Speech fallback invalidates certification."),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def source(path: str) -> str:
    return git("show", f"{SOURCE_COMMIT}:{path}")


def fingerprint(path: str) -> dict:
    data = source(path).encode()
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


def route_inventory(paths: list[str]) -> list[dict]:
    """Static routes, including WS, grouped by process; never start backends."""
    routes = []
    for path in paths:
        tree = ast.parse(source(path), filename=path)
        prefixes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = getattr(node.value.func, "id", None)
                if name in {"APIRouter", "FastAPI"}:
                    prefix = next((ast.literal_eval(k.value) for k in node.value.keywords if k.arg == "prefix"), "")
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            prefixes[target.id] = prefix
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                method = decorator.func.attr
                owner = getattr(decorator.func.value, "id", "")
                if owner not in prefixes or method not in {"get", "post", "put", "patch", "delete", "head", "options", "websocket", "api_route"}:
                    continue
                route = ast.literal_eval(decorator.args[0])
                methods = [method.upper()]
                if method == "api_route":
                    methods = next((ast.literal_eval(k.value) for k in decorator.keywords if k.arg == "methods"), ["GET"])
                for verb in methods:
                    routes.append({"method": verb, "path": prefixes[owner] + route, "source": path, "line": decorator.lineno, "handler": node.name})
    return sorted(routes, key=lambda r: (r["source"], r["path"], r["method"]))


def build_baseline() -> dict:
    files = git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines()
    model_paths = [p for p in files if p.startswith("manifests/models/") and p.endswith(".yaml")]
    models = []
    for path in model_paths:
        m = yaml.safe_load(source(path))
        model_id = m["metadata"]["id"]
        backend, implementation_sources, finding = AUDIT[model_id]
        caps = sorted(set(m["capabilities"]) | set(m.get("capabilities_truth", {})) | set(m.get("task_variant_matrix", {})))
        pairs = []
        for cap in caps:
            supported = m["capabilities"].get(cap, False)
            truth = m.get("capabilities_truth", {}).get(cap, {})
            # Explicitly describe code reachability; support flags alone do not
            # prove a working implementation, much less real inference.
            implementation = "ADAPTER_OR_WORKFLOW_PRESENT" if supported else "DISABLED_OR_UNBACKED"
            if cap not in m["capabilities"]:
                implementation = "MATRIX_ONLY_DRIFT"
            if model_id == "qwen-image-2.1" and cap in {"reference_image", "multi_reference", "control_pose", "control_depth"}:
                implementation = "TASK_REGISTRY_MISMATCH"
            pairs.append({"capability": cap, "claimed_support": supported,
                "claimed_truth": truth, "claimed_variants": m.get("task_variant_matrix", {}).get(cap, []),
                "implementation": implementation, "implementation_sources": implementation_sources,
                "V3": "BLOCKED", "V4": "BLOCKED", "semantic": "BLOCKED", "production_ready": False,
                "disposition": "BLOCKED" if supported else "DISABLED",
                "reason": finding if supported else "Not enabled by frozen manifest; wrappers/matrix entries do not establish support.",
                "accepted_evidence": []})
        models.append({"id": model_id, "manifest": fingerprint(path), "checkpoint_claim": m["weights"],
            "checkpoint_verification": "BLOCKED", "backend_observed_in_code": backend,
            "hardware_requirement_claim": m["hardware"], "hardware_verified": False,
            "frozen_manifest_claims": {k: m.get(k) for k in ("status", "coverage_percent", "validation_status", "validation_notes", "validated_hardware_profiles", "installed", "checkpoint_valid", "backend_ready", "evidence_run_id")},
            "audit_state": "EXPERIMENTAL" if any(m["capabilities"].values()) else "DISABLED",
            "finding": finding, "capabilities": pairs})
    if {m["id"] for m in models} != set(AUDIT):
        raise ValueError("Model catalog changed: explicit audit required")

    evidence = []
    for path in files:
        if path in EVIDENCE:
            classification, reason = EVIDENCE[path]
        elif path.startswith("benchmarks/results/") and path.endswith((".json", ".jsonl")):
            classification, reason = "LEGACY_NON_CERTIFYING", "Legacy contract/smoke/local results; no admissible §38 record binding exact weights, real GPU, real input/output and semantic validation."
        else:
            continue
        evidence.append({**fingerprint(path), "classification": classification,
            "accepted_for_V3": False, "accepted_for_V4": False, "reason": reason})

    test_paths = [p for p in files if p.endswith(".py") and (Path(p).name.startswith("test_") or Path(p).name == "conftest.py")]
    route_paths = [p for p in files if p.endswith(".py") and p.startswith(("factory/apps/control-api/src/", "services/")) and "/tests/" not in p]
    test_inventory = []
    for path in test_paths:
        tree = ast.parse(source(path), filename=path)
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")]
        test_inventory.append({**fingerprint(path), "test_functions": funcs})
    pairs = [c for m in models for c in m["capabilities"]]
    supplement_path = OUTPUT / "local_evidence_audit.json"
    supplement = json.loads(supplement_path.read_text())
    return {"schema_version": 1, "phase": 0, "source_commit": SOURCE_COMMIT,
        "audit_date": "2026-11-12", "master_plan": fingerprint("docs/MASTER_PLAN.md"),
        "scope": "Immutable source-tree/code/evidence audit, not a GPU run. BLOCKED means required proof missing, not inference failure.",
        "legacy_runtime_flags": "Frozen installed/checkpoint_valid/backend_ready flags are unverified static declarations, never current deployment attestation. Correct replacement belongs to Phases 1/3/4.",
        "summary": {"models": len(models), "model_capability_pairs": len(pairs),
            "dispositions": dict(sorted(Counter(p["disposition"] for p in pairs).items())), "production_ready_pairs": 0},
        "models": models, "evidence_audit": evidence,
        "supplemental_local_evidence_audit": {"path": str(supplement_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(supplement_path.read_bytes()).hexdigest(),
            "scope": supplement["scope"], "evidence": supplement["evidence"]},
        "api_inventory": {"method": "Static AST route decorators with local router prefixes; grouped by source/process, not a claim of unified public reachability. FastAPI generated documentation routes excluded.", "routes": route_inventory(route_paths)},
        "tests": {"scope": "All tracked Python test/conftest files, including suites outside default pytest testpaths. Parametrized cases are not inferred from function counts.", "configuration": [fingerprint(p) for p in ("pytest.ini", "pyproject.toml")], "files": test_inventory},
        "exit_gate": {"status": "PASS", "meaning": "Every frozen model/capability has explicit audited disposition and evidence rejection reasons. No pair certified. Does not grant Phase 1 or release readiness."}}


def render_matrix(report: dict) -> str:
    lines = ["# Phase 0 — Truth Baseline", "", f"Frozen main: `{report['source_commit']}`. Master Plan §27; audit date 2026-11-12.", "",
        "Generated by `python benchmarks/truth_baseline.py`; `baseline.json` contains source hashes, original claims, API/test inventories and evidence decisions.", "",
        "**No production-ready pair.** BLOCKED means missing required evidence, not a measured inference failure. All V3/V4/semantic proofs remain BLOCKED; DISABLED means not enabled in the frozen manifest. Code presence is not inference proof.", "",
        "Legacy full/partial coverage and static installed/checkpoint_valid/backend_ready flags are unverified implementation/configuration claims. They do not attest to a live session. No historical L-level is automatically translated into V3/V4.", "",
        "| Model | Capability | Claimed | Code observation | V3 | V4 | Semantic | Disposition | Ready |", "|---|---|---|---|---|---|---|---|---|"]
    for model in report["models"]:
        for cap in model["capabilities"]:
            lines.append(f"| {model['id']} | {cap['capability']} | {str(cap['claimed_support']).lower()} | {cap['implementation']} | {cap['V3']} | {cap['V4']} | {cap['semantic']} | {cap['disposition']} | false |")
    lines += ["", "## Model findings", ""]
    for m in report["models"]:
        w = m["checkpoint_claim"]
        lines += [f"### {m['id']}", "", f"- Checkpoint claim (unverified): `{w.get('repository')}` @ `{w.get('checkpoint_revision')}` / `{w.get('weights_file')}`.",
            f"- Backend observed: {m['backend_observed_in_code']}. GPU requirement claim: `{json.dumps(m['hardware_requirement_claim'], sort_keys=True)}`.",
            f"- Audit: {m['finding']}", f"- Code: {', '.join('`' + s + '`' for s in m['capabilities'][0]['implementation_sources'])}.", ""]
    lines += ["## Evidence decisions", ""]
    for e in report["evidence_audit"]:
        lines.append(f"- `{e['path']}` — **{e['classification']}**: {e['reason']}")
    lines += ["", "## Supplement: previously ignored local evidence", "",
        "These artifacts were found in the local workspace during Kaggle source investigation. They are outside the frozen Git tree; `local_evidence_audit.json` freezes their hashes and review results without committing historical media.", ""]
    for e in report["supplemental_local_evidence_audit"]["evidence"]:
        lines.append(f"- `{e['path']}` — **{e['classification']}**: {e['reason']}")
    lines += ["", "## Exit Gate", "", "PASS for Phase 0 truth inventory only. All model/capability pairs are accounted for; no inference certification is inferred from reports, synthetic tensors, mocked workers, artifact existence or passing software tests.", "", "Phase 1 must resolve checkpoint/backend/variant identity drift before capability canonicalization. Architecture, Studio and cleanup are deferred to their dependency gates.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check checked-in JSON/Markdown against frozen source without writing")
    args = parser.parse_args()
    report = build_baseline()
    outputs = {OUTPUT / "baseline.json": json.dumps(report, indent=2, ensure_ascii=False) + "\n", OUTPUT / "baseline_matrix.md": render_matrix(report)}
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, value in outputs.items() if not p.exists() or p.read_text() != value]
        if stale:
            parser.exit(1, "Stale or missing baseline: " + ", ".join(stale) + "\n")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for path, value in outputs.items():
            path.write_text(value)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
