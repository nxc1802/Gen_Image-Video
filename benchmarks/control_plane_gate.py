"""Phase 7 single-host composition gate; no model inference certification."""

from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.openai_api_gate import evaluate as api_gate
from deployments.kaggle.backend_lifecycle_probe import source_bundle

OPTIONAL = {
    "redis",
    "boto3",
    "botocore",
    "sqlalchemy",
    "alembic",
    "aiosqlite",
    "asyncpg",
    "psycopg",
    "psycopg2",
    "jwt",
    "cryptography",
}


def evaluate():
    checks = []
    findings = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            findings.append(name)

    prior = api_gate()
    check("previous_phases_current_source", prior["exit_gate"] == "PASS")
    if prior["exit_gate"] != "PASS":
        findings.extend(prior["findings"])
    source = hashlib.sha256(source_bundle()).hexdigest()
    minimal = {}
    contracts = {}
    try:
        from factory.apps.control_api.src.main import create_app
        import tomllib

        paths = create_app().openapi()["paths"]
        check(
            "single_production_gateway_no_worker_leases",
            not any("/workers" in p for p in paths) and "post" not in paths["/v1/jobs"],
        )
        check(
            "local_job_management_available",
            all(
                p in paths
                for p in [
                    "/v1/jobs",
                    "/v1/jobs/{identifier}",
                    "/v1/jobs/{identifier}/cancel",
                    "/v1/jobs/{identifier}/events",
                    "/v1/jobs/{identifier}/content",
                ]
            ),
        )
        main = ROOT / "factory/apps/control-api/src/main.py"
        tree = ast.parse(main.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        check(
            "core_startup_has_no_legacy_composition",
            not any(
                ".queue" in name
                or ".legacy" in name
                or ".storage.s3" in name
                or ".storage.router" in name
                for name in imports
            ),
        )
        dependencies = {}
        for relative in [
            "pyproject.toml",
            "factory/apps/control-api/pyproject.toml",
            "packages/auth/pyproject.toml",
        ]:
            data = tomllib.loads((ROOT / relative).read_text())
            values = data["project"].get("dependencies", [])
            dependencies[relative] = values
            check(
                "no_required_infrastructure_" + relative,
                not any(
                    any(value.lower().startswith(name) for name in OPTIONAL | {"pyjwt"})
                    for value in values
                ),
            )
        with tempfile.TemporaryDirectory(prefix="factory-p7-gate-") as directory:
            temp = Path(directory)
            venv = temp / "core"
            create = subprocess.run(
                ["uv", "venv", str(venv)], cwd=ROOT, capture_output=True, text=True, timeout=60
            )
            check("fresh_core_environment_created", create.returncode == 0)
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            command = ["uv", "pip", "install", "--python", str(python)]
            for package in ["contracts", "auth", "model-registry"]:
                command.extend(["-e", str(ROOT / "packages" / package)])
            command.extend(["-e", str(ROOT / "factory/apps/control-api")])
            install = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            check("core_only_dependencies_installed", install.returncode == 0)
            if install.returncode != 0:
                raise RuntimeError(
                    "Minimal install failed: " + (install.stderr + install.stdout)[-2000:]
                )
            output = temp / "observation.json"
            probe = subprocess.run(
                [
                    str(python),
                    str(ROOT / "benchmarks/core_runtime_probe.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            check("real_local_http_probe_without_infrastructure", probe.returncode == 0)
            if probe.returncode != 0:
                raise RuntimeError(
                    "Minimal core probe failed: " + (probe.stderr + probe.stdout)[-3000:]
                )
            minimal = json.loads(output.read_text())
            minimal.update(
                source_bundle_sha256=source,
                probe_source_sha256=hashlib.sha256(
                    (ROOT / "benchmarks/core_runtime_probe.py").read_bytes()
                ).hexdigest(),
                classification="SOFTWARE_ONLY_CORE_RUNTIME_AND_EXPLICIT_ADAPTER_DOUBLES",
            )
            check(
                "every_minimal_runtime_check_passed",
                all(minimal["checks"].values())
                and not minimal["native_inference_executed"]
                and not minimal["production_ready"],
            )
            check(
                "optional_infrastructure_not_installed",
                not (OPTIONAL | {"pyjwt"}).intersection(minimal["installed_packages"]),
            )
            (ROOT / "validation/core_runtime_probe.json").write_text(
                json.dumps(minimal, indent=2) + "\n"
            )
            junit = temp / "contracts.xml"
            suite = ROOT / "tests/contract/test_local_control_plane.py"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(suite),
                    "-q",
                    "-o",
                    "addopts=",
                    f"--junitxml={junit}",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            counts = {k: 0 for k in ["tests", "failures", "errors", "skipped"]}
            cases = []
            if junit.is_file():
                root = ET.parse(junit)
                for entry in root.iter("testsuite"):
                    for key in counts:
                        counts[key] += int(entry.attrib.get(key, 0))
                cases = [case.attrib["name"] for case in root.iter("testcase")]
            check(
                "cancellation_locks_shutdown_artifact_contracts",
                run.returncode == 0
                and counts["tests"] >= 11
                and all(counts[k] == 0 for k in ["failures", "errors", "skipped"]),
            )
            contracts = {
                "counts": counts,
                "cases": cases,
                "source_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
                "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest()
                if junit.is_file()
                else None,
                "diagnostics": None if run.returncode == 0 else (run.stdout + run.stderr)[-4000:],
            }
        run = json.loads((ROOT / "validation/backend_lifecycle_run.json").read_text())
        version = run["response"]["versionNumber"]
        observed = json.loads(
            (ROOT / f"validation/backend_lifecycle_observation_v{version}.json").read_text()
        )
        live = observed["control_plane"]
        check(
            "real_kaggle_current_core_composition",
            observed["source_bundle_sha256"] == source
            and live["mode"] == "single-host"
            and live["job_storage"] == "memory"
            and live["artifact_storage"] == "local_scratch"
            and live["external_services_required"]
            == live["legacy_components"]
            == live["loaded_optional_infrastructure"]
            == []
            and live["jobs_accepting"] is True
            and live["pid"] > 0
            and live["production_ready"] is False,
        )
        minimal["real_kaggle_core_observation"] = (
            f"validation/backend_lifecycle_observation_v{version}.json"
        )
    except Exception as exc:
        findings.append(f"{type(exc).__name__}: {exc}")
    return {
        "phase": 7,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_gate": "PASS" if not findings else "NOT_MET",
        "classification": "SINGLE_HOST_COMPOSITION_AND_SOFTWARE_ONLY",
        "source_bundle_sha256": source,
        "checks": checks,
        "findings": findings,
        "prior_api_gate": prior["exit_gate"],
        "minimal_runtime": minimal,
        "contracts": contracts,
        "scope": "Core-only installation with no optional infrastructure packages; actual local HTTP/core Engine path uses explicit test authorization/Adapter doubles. Current-source Kaggle observes core startup plus ASR lifecycle, not ASR inference. Legacy code remains isolated for migration.",
        "production_ready": False,
        "native_inference_executed_by_this_gate": False,
        "semantic_validation_executed": False,
        "full_factory_e2e_executed": False,
        "retained_legacy_host": "factory/apps/control-api/src/legacy.py (rejects production)",
        "external_services_required": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/control_plane_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["exit_gate", "classification", "findings"]}))
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
