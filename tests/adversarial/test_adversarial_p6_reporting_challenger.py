"""
tests/adversarial/test_adversarial_p6_reporting_challenger.py

Phase 6 Adversarial Challenger Test Suite:
Empirical stress testing of the Phase 6 benchmark reporting pipeline,
CLI fault injection, input corruption tolerance, mathematical reconciliation
against results.jsonl, zero-mock enforcement, and truth model fidelity.

Adheres strictly to:
- Factory_v4_Lite_Full_Implementation_and_Benchmark_Plan.md (§10.5, §10.6, §10.7)
- Truth Model Mandate (§5.1: Declared vs Implemented vs Live Verified)
- Canonical 163-case matrix results (benchmarks/results/p5_matrix_live_04/)
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

import pytest
import yaml

from benchmarks.runners.report_builder import (
    WORKSPACE_ROOT,
    BenchmarkReportBuilder,
    main as report_builder_main,
)

CANONICAL_RUN_DIR = WORKSPACE_ROOT / "benchmarks" / "results" / "p5_matrix_live_04"
CANONICAL_CONFIG_DIR = WORKSPACE_ROOT / "benchmarks" / "configs"
CANONICAL_KAGGLE_REPORT = WORKSPACE_ROOT / "deployments" / "kaggle" / "live_hardware_attested_report.json"
CANONICAL_REPORTS_DIR = WORKSPACE_ROOT / "benchmarks" / "reports"

EXPECTED_REPORT_FILES = [
    "executive_summary.md",
    "capability_matrix.md",
    "model_comparison.md",
    "latency_vram.md",
    "failure_analysis.md",
    "final_report.md",
]


# =============================================================================
# 1. Fault Injection & Edge-Case Probes
# =============================================================================

class TestReportBuilderFaultInjectionAndEdgeCases:
    """Adversarial fault injection testing CLI, inputs, corruption, and edge cases."""

    def test_cli_missing_required_run_dir_fails(self, tmp_path: Path):
        """CLI with non-existent --run-dir must fail with non-zero exit code."""
        bad_run_dir = tmp_path / "non_existent_run_dir"
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.runners.report_builder",
            "--run-dir",
            str(bad_run_dir),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        assert res.returncode != 0
        assert "FileNotFoundError" in res.stderr or "metadata.json not found" in res.stderr

    def test_cli_rejects_unknown_arguments(self):
        """CLI must reject unsupported arguments with exit code 2 (argparse error)."""
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.runners.report_builder",
            "--unsupported-adversarial-flag-xyz",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        assert res.returncode == 2
        assert "unrecognized arguments" in res.stderr

    def test_missing_metadata_json_raises_filenotfound(self, tmp_path: Path):
        """Builder raises FileNotFoundError when metadata.json is missing."""
        (tmp_path / "results.jsonl").write_text("", encoding="utf-8")
        builder = BenchmarkReportBuilder(run_dir=tmp_path, output_dir=tmp_path / "out")
        with pytest.raises(FileNotFoundError, match="metadata.json not found"):
            builder.load_data()

    def test_missing_results_jsonl_raises_filenotfound(self, tmp_path: Path):
        """Builder raises FileNotFoundError when results.jsonl is missing."""
        (tmp_path / "metadata.json").write_text(json.dumps({"run_id": "test"}), encoding="utf-8")
        builder = BenchmarkReportBuilder(run_dir=tmp_path, output_dir=tmp_path / "out")
        with pytest.raises(FileNotFoundError, match="results.jsonl not found"):
            builder.load_data()

    def test_corrupted_jsonl_syntax_raises_json_decode_error(self, tmp_path: Path):
        """Malformed JSON line in results.jsonl must raise json.JSONDecodeError."""
        (tmp_path / "metadata.json").write_text(json.dumps({"run_id": "test"}), encoding="utf-8")
        corrupted_jsonl = tmp_path / "results.jsonl"
        corrupted_jsonl.write_text(
            '{"test_id": "R01", "status": "PASS"}\n'
            '{"test_id": "R02", "status": BROKEN_SYNTAX_WITHOUT_QUOTES}\n',
            encoding="utf-8",
        )
        builder = BenchmarkReportBuilder(run_dir=tmp_path, output_dir=tmp_path / "out")
        with pytest.raises(json.JSONDecodeError):
            builder.load_data()

    def test_missing_evaluation_cases_yaml_raises_filenotfound(self, tmp_path: Path):
        """Builder raises FileNotFoundError when evaluation_cases.yaml is missing in config dir."""
        builder = BenchmarkReportBuilder(
            run_dir=CANONICAL_RUN_DIR,
            output_dir=tmp_path / "out",
            config_dir=tmp_path / "empty_configs",
        )
        with pytest.raises(FileNotFoundError, match="evaluation_cases.yaml not found"):
            builder.load_data()

    def test_missing_optional_configs_survives(self, tmp_path: Path):
        """Builder survives when optional hardware_profiles.yaml and music_comparison.yaml are absent."""
        cfg_dir = tmp_path / "minimal_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        # Copy only evaluation_cases.yaml
        eval_cases = (CANONICAL_CONFIG_DIR / "evaluation_cases.yaml").read_text(encoding="utf-8")
        (cfg_dir / "evaluation_cases.yaml").write_text(eval_cases, encoding="utf-8")

        builder = BenchmarkReportBuilder(
            run_dir=CANONICAL_RUN_DIR,
            output_dir=tmp_path / "out",
            config_dir=cfg_dir,
        )
        builder.load_data()
        assert builder.hardware_profiles == {}
        assert builder.music_config == {}

    def test_missing_failures_json_and_metrics_csv_survives(self, tmp_path: Path):
        """Builder gracefully handles absence of failures.json and metrics.csv without crash."""
        run_dir = tmp_path / "partial_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Copy metadata.json and results.jsonl only
        (run_dir / "metadata.json").write_text(
            (CANONICAL_RUN_DIR / "metadata.json").read_text(encoding="utf-8")
        )
        (run_dir / "results.jsonl").write_text(
            (CANONICAL_RUN_DIR / "results.jsonl").read_text(encoding="utf-8")
        )

        builder = BenchmarkReportBuilder(run_dir=run_dir, output_dir=tmp_path / "out")
        builder.load_data()
        assert builder.failures == []
        assert builder.metrics == []
        assert builder.pass_inference_latencies == []

    def test_missing_kaggle_report_falls_back_gracefully(self, tmp_path: Path):
        """Non-existent kaggle_report path does not crash builder and defaults to empty dict."""
        bad_kaggle = tmp_path / "non_existent_kaggle.json"
        builder = BenchmarkReportBuilder(
            run_dir=CANONICAL_RUN_DIR,
            output_dir=tmp_path / "out",
            config_dir=CANONICAL_CONFIG_DIR,
            kaggle_report=bad_kaggle,
        )
        builder.load_data()
        assert builder.kaggle_report == {}
        # Verify latency_vram report generates without crash
        vram_doc = builder.build_latency_vram()
        assert "Kaggle Dual Tesla T4 Attested Hardware Telemetry" in vram_doc

    def test_deep_nested_output_directory_creation(self, tmp_path: Path):
        """Builder creates deeply nested non-existent output directory recursively."""
        deep_out = tmp_path / "a" / "b" / "c" / "deep_reports"
        assert not deep_out.exists()
        builder = BenchmarkReportBuilder(
            run_dir=CANONICAL_RUN_DIR,
            output_dir=deep_out,
            config_dir=CANONICAL_CONFIG_DIR,
        )
        files = builder.generate_all_reports()
        assert deep_out.exists()
        assert len(files) == 6

    def test_empty_results_jsonl_adversarial_vulnerability_probe(self, tmp_path: Path):
        """Empirically probes builder behavior when results.jsonl contains 0 evaluated cases.

        Documented finding: build_final_report raises ZeroDivisionError when total_cases == 0
        due to missing total_cases guard in lines 794-796.
        """
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": "test_empty_matrix",
            "timestamp": "2026-11-08T00:00:00Z",
            "hardware_profile": "darwin-apple-silicon-local",
            "environment": {"system": "Darwin", "machine": "arm64", "cuda_available": False},
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        (run_dir / "results.jsonl").write_text("", encoding="utf-8")

        builder = BenchmarkReportBuilder(run_dir=run_dir, output_dir=tmp_path / "out")
        builder.load_data()
        assert len(builder.results) == 0

        # Executive summary handles total_cases == 0 safely:
        exec_doc = builder.build_executive_summary()
        assert "0" in exec_doc

        # build_final_report encounters ZeroDivisionError when total_cases is 0
        with pytest.raises(ZeroDivisionError):
            builder.build_final_report()

    def test_zero_timed_passes_adversarial_format_probe(self, tmp_path: Path):
        """Empirically probes builder behavior when results has non-zero cases but 0 timed passes.

        Documented finding: build_final_report encounters TypeError at line 876 when
        self.pass_inference_latencies is empty due to `[] and mean(...)` evaluating to `[]`.
        """
        run_dir = tmp_path / "hb_only_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": "test_hb_only",
            "timestamp": "2026-11-08T00:00:00Z",
            "hardware_profile": "darwin-apple-silicon-local",
            "environment": {"system": "Darwin", "machine": "arm64", "cuda_available": False},
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        # One valid case with HARDWARE_BLOCKED (no timed pass)
        res = {
            "test_id": "T01",
            "suite": "live_model",
            "status": "HARDWARE_BLOCKED",
            "model": "qwen-text",
            "variant": "default",
            "task": "text_chat",
        }
        (run_dir / "results.jsonl").write_text(json.dumps(res) + "\n", encoding="utf-8")

        builder = BenchmarkReportBuilder(run_dir=run_dir, output_dir=tmp_path / "out")
        builder.load_data()
        assert len(builder.results) == 1
        assert len(builder.pass_inference_latencies) == 0

        with pytest.raises(TypeError, match="unsupported format string"):
            builder.build_final_report()

    def test_unknown_test_id_mapped_gracefully(self, tmp_path: Path):
        """Result with an unknown test ID not in evaluation_cases.yaml maps to 'unknown' group."""
        run_dir = tmp_path / "ghost_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": "test_ghost", "timestamp": "2026-11-08T00:00:00Z"}
        (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        res = {
            "test_id": "GHOST-999",
            "suite": "synthetic",
            "status": "PASS",
            "model": "ghost_model",
            "task": "ghost_task",
        }
        (run_dir / "results.jsonl").write_text(json.dumps(res) + "\n", encoding="utf-8")

        builder = BenchmarkReportBuilder(run_dir=run_dir, output_dir=tmp_path / "out")
        builder.load_data()
        assert "unknown" in builder.group_counts
        assert builder.group_counts["unknown"]["PASS"] == 1


# =============================================================================
# 2. Mathematical Consistency & Reconciliation Audit
# =============================================================================

class TestMathematicalConsistencyAndReconciliationAudit:
    """Rigorous mathematical audit reconciling results.jsonl against all generated markdown reports."""

    @pytest.fixture(scope="class")
    @classmethod
    def loaded_data(cls):
        """Loads canonical evaluation artifacts."""
        with open(CANONICAL_RUN_DIR / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        results: List[Dict[str, Any]] = []
        with open(CANONICAL_RUN_DIR / "results.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        with open(CANONICAL_RUN_DIR / "failures.json", encoding="utf-8") as f:
            failures = json.load(f)

        with open(CANONICAL_RUN_DIR / "metrics.csv", encoding="utf-8") as f:
            metrics = list(csv.DictReader(f))

        with open(CANONICAL_CONFIG_DIR / "evaluation_cases.yaml", encoding="utf-8") as f:
            cases_config = yaml.safe_load(f)

        reports = {}
        for fname in EXPECTED_REPORT_FILES:
            rpath = CANONICAL_REPORTS_DIR / fname
            assert rpath.exists(), f"Report file missing: {rpath}"
            reports[fname] = rpath.read_text(encoding="utf-8")

        return {
            "metadata": metadata,
            "results": results,
            "failures": failures,
            "metrics": metrics,
            "cases_config": cases_config,
            "reports": reports,
        }

    def test_audit_total_cases_and_status_breakdown(self, loaded_data):
        """Verifies exactly 163 records: 51 PASS, 111 HARDWARE_BLOCKED, 1 NOT_IMPLEMENTED, 0 FAIL, 0 ERROR."""
        results = loaded_data["results"]
        metadata = loaded_data["metadata"]

        assert len(results) == 163
        assert metadata["cases_evaluated"] == 163
        assert metadata["total_planned_cases"] == 163

        status_counts = Counter(r["status"] for r in results)
        assert status_counts["PASS"] == 51
        assert status_counts["HARDWARE_BLOCKED"] == 111
        assert status_counts["NOT_IMPLEMENTED"] == 1
        assert status_counts.get("FAIL", 0) == 0
        assert status_counts.get("ERROR", 0) == 0
        assert status_counts.get("BLOCKED", 0) == 0
        assert status_counts.get("SKIPPED", 0) == 0
        assert status_counts.get("NOT_RUN", 0) == 0

        # Failures list must match non-PASS cases exactly (111 + 1 = 112)
        failures = loaded_data["failures"]
        assert len(failures) == 112
        failure_statuses = Counter(f["status"] for f in failures)
        assert failure_statuses["HARDWARE_BLOCKED"] == 111
        assert failure_statuses["NOT_IMPLEMENTED"] == 1

    def test_audit_executive_summary_mathematics(self, loaded_data):
        """Verifies exact numbers and percentages in executive_summary.md."""
        doc = loaded_data["reports"]["executive_summary.md"]

        # Exact percentages
        assert "**51**" in doc
        assert "31.29%" in doc
        assert "**111**" in doc
        assert "68.10%" in doc
        assert "**1**" in doc
        assert "0.61%" in doc
        assert "**163**" in doc
        assert "100.00%" in doc

        # Zero violations
        assert "| **FAIL** | **0** | **0.00%** |" in doc
        assert "| **ERROR** | **0** | **0.00%** |" in doc
        assert "| **SKIPPED** | **0** | **0.00%** |" in doc
        assert "| **NOT_RUN** | **0** | **0.00%** |" in doc

    def test_audit_15_subsystems_group_reconciliation(self, loaded_data):
        """Verifies exact counts for all 15 operational subsystems across reports."""
        cases = loaded_data["cases_config"]["cases"]
        case_map = {c["id"]: c for c in cases}
        results = loaded_data["results"]

        expected_group_breakdown = {
            "acestep": {"total": 24, "PASS": 5, "HARDWARE_BLOCKED": 18, "NOT_IMPLEMENTED": 1},
            "cross_feature": {"total": 12, "PASS": 0, "HARDWARE_BLOCKED": 12, "NOT_IMPLEMENTED": 0},
            "decision": {"total": 3, "PASS": 0, "HARDWARE_BLOCKED": 3, "NOT_IMPLEMENTED": 0},
            "embedding": {"total": 4, "PASS": 0, "HARDWARE_BLOCKED": 4, "NOT_IMPLEMENTED": 0},
            "enhancement": {"total": 6, "PASS": 0, "HARDWARE_BLOCKED": 6, "NOT_IMPLEMENTED": 0},
            "gpu_runtime": {"total": 10, "PASS": 0, "HARDWARE_BLOCKED": 10, "NOT_IMPLEMENTED": 0},
            "image": {"total": 15, "PASS": 3, "HARDWARE_BLOCKED": 12, "NOT_IMPLEMENTED": 0},
            "minimax_video": {"total": 10, "PASS": 2, "HARDWARE_BLOCKED": 8, "NOT_IMPLEMENTED": 0},
            "router_infrastructure": {"total": 28, "PASS": 28, "HARDWARE_BLOCKED": 0, "NOT_IMPLEMENTED": 0},
            "speech": {"total": 11, "PASS": 2, "HARDWARE_BLOCKED": 9, "NOT_IMPLEMENTED": 0},
            "text": {"total": 5, "PASS": 0, "HARDWARE_BLOCKED": 5, "NOT_IMPLEMENTED": 0},
            "video_integration": {"total": 2, "PASS": 2, "HARDWARE_BLOCKED": 0, "NOT_IMPLEMENTED": 0},
            "vlm": {"total": 5, "PASS": 0, "HARDWARE_BLOCKED": 5, "NOT_IMPLEMENTED": 0},
            "wan_video": {"total": 8, "PASS": 2, "HARDWARE_BLOCKED": 6, "NOT_IMPLEMENTED": 0},
            "yue2": {"total": 20, "PASS": 7, "HARDWARE_BLOCKED": 13, "NOT_IMPLEMENTED": 0},
        }

        # Calculate actual group counts
        actual_group_breakdown = {}
        for r in results:
            tid = r["test_id"]
            grp = case_map[tid]["group"]
            st = r["status"]
            if grp not in actual_group_breakdown:
                actual_group_breakdown[grp] = Counter()
            actual_group_breakdown[grp][st] += 1

        for grp, exp in expected_group_breakdown.items():
            act = actual_group_breakdown[grp]
            assert sum(act.values()) == exp["total"], f"Group {grp} total mismatch"
            assert act.get("PASS", 0) == exp["PASS"], f"Group {grp} PASS mismatch"
            assert act.get("HARDWARE_BLOCKED", 0) == exp["HARDWARE_BLOCKED"], f"Group {grp} HB mismatch"
            assert act.get("NOT_IMPLEMENTED", 0) == exp["NOT_IMPLEMENTED"], f"Group {grp} NI mismatch"

        # Verify executive_summary table rows match exactly
        exec_doc = loaded_data["reports"]["executive_summary.md"]
        for grp, exp in expected_group_breakdown.items():
            tot = exp["total"]
            p = exp["PASS"]
            hb = exp["HARDWARE_BLOCKED"]
            ni = exp["NOT_IMPLEMENTED"]
            expected_row = f"| `{grp}` | {tot} | {p} | {hb} | {ni} | 0 |"
            assert expected_row in exec_doc, f"Row missing or mismatch in executive_summary: {expected_row}"

    def test_audit_all_163_test_ids_present_in_final_report_and_matrix(self, loaded_data):
        """Verifies that every single test ID (163 of 163) appears in final_report.md and capability_matrix.md."""
        results = loaded_data["results"]
        final_doc = loaded_data["reports"]["final_report.md"]
        cap_doc = loaded_data["reports"]["capability_matrix.md"]

        for r in results:
            tid = r["test_id"]
            assert f"`{tid}`" in final_doc, f"Test ID {tid} missing in final_report.md"
            assert f"`{tid}`" in cap_doc, f"Test ID {tid} missing in capability_matrix.md"

    def test_audit_model_comparison_prompts_b_m01_to_b_m12(self, loaded_data):
        """Verifies all B-M01 to B-M12 prompts and 8 listening rubric dimensions in model_comparison.md."""
        comp_doc = loaded_data["reports"]["model_comparison.md"]

        for i in range(1, 13):
            pid = f"B-M{i:02d}"
            assert f"`{pid}`" in comp_doc, f"Prompt ID {pid} missing from model_comparison.md"

        rubric_dimensions = [
            "Lyric Intelligibility",
            "Melody & Harmony",
            "Arrangement & Instrumentation",
            "Vocal Quality & Timbre",
            "Style & Genre Adherence",
            "Song Coherence",
            "Absence of Artifacts",
            "Overall Preference",
        ]
        for dim in rubric_dimensions:
            assert dim in comp_doc, f"Rubric dimension {dim} missing from model_comparison.md"

    def test_audit_historical_fixes_in_failure_analysis(self, loaded_data):
        """Verifies FIX-01..20, VULN-P3-01..05, and VULN-P4-01..04 in failure_analysis.md."""
        fail_doc = loaded_data["reports"]["failure_analysis.md"]

        for i in range(1, 21):
            assert f"**FIX-{i:02d}**" in fail_doc, f"FIX-{i:02d} missing from failure_analysis.md"

        for i in range(1, 6):
            assert f"**VULN-P3-0{i}**" in fail_doc, f"VULN-P3-0{i} missing from failure_analysis.md"

        for i in range(1, 5):
            assert f"**VULN-P4-0{i}**" in fail_doc, f"VULN-P4-0{i} missing from failure_analysis.md"


# =============================================================================
# 3. Zero-Mock & Truth Model Integrity
# =============================================================================

class TestZeroMockAndTruthModelIntegrity:
    """Verifies that no mock inferences pass as live certification and hardware blockers are honest."""

    def test_zero_mock_and_zero_fallback_across_all_163_records(self):
        """Strict verification: mock_used must be False and fallback_used must be False for 100% of records."""
        results_file = CANONICAL_RUN_DIR / "results.jsonl"
        with open(results_file, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                rec = json.loads(line)
                tid = rec.get("test_id", f"record_{idx}")
                integrity = rec.get("integrity", {})
                assert integrity.get("mock_used") is False, f"Test {tid} violated integrity: mock_used is True"
                assert integrity.get("fallback_used") is False, f"Test {tid} violated integrity: fallback_used is True"

    def test_darwin_host_claims_no_cuda_inferences(self):
        """Verifies that Darwin host has 0 CUDA GPUs and all heavy neural inferences are HARDWARE_BLOCKED."""
        meta_file = CANONICAL_RUN_DIR / "metadata.json"
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        env = meta.get("environment", {})
        assert env.get("system") == "Darwin"
        assert env.get("cuda_available") is False
        assert env.get("gpu_count") == 0

        # Neural test cases requiring CUDA must never be PASS on Darwin
        neural_groups = ["text", "vlm", "decision", "embedding", "cross_feature"]
        with open(CANONICAL_CONFIG_DIR / "evaluation_cases.yaml", encoding="utf-8") as f:
            cases = yaml.safe_load(f)["cases"]
        neural_tids = {c["id"] for c in cases if c.get("group") in neural_groups}

        with open(CANONICAL_RUN_DIR / "results.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["test_id"] in neural_tids:
                    assert rec["status"] == "HARDWARE_BLOCKED", (
                        f"Neural test {rec['test_id']} in group {rec.get('group')} falsely passed without CUDA"
                    )

    def test_truth_model_separation_in_capability_matrix(self):
        """Verifies that capability_matrix.md enforces Declared vs Implemented vs Live Verified."""
        cap_file = CANONICAL_REPORTS_DIR / "capability_matrix.md"
        content = cap_file.read_text(encoding="utf-8")

        assert "## 1. Truth Model Architecture" in content
        assert "Declared" in content
        assert "Implemented" in content
        assert "Live Verified (Darwin)" in content
        assert "Live Verified (Dual T4)" in content

        # ACE-23 must be explicitly NOT_IMPLEMENTED
        assert "`ACE-23`" in content
        assert "Workflow contract absence explicitly recorded" in content

    def test_kaggle_telemetry_isolation(self):
        """Kaggle Dual Tesla T4 telemetry is explicitly cited from live_hardware_attested_report.json."""
        vram_file = CANONICAL_REPORTS_DIR / "latency_vram.md"
        content = vram_file.read_text(encoding="utf-8")

        assert "deployments/kaggle/live_hardware_attested_report.json" in content
        assert "2 × NVIDIA Tesla T4 (15,360 MB GDDR6 per card)" in content
        assert "580.178.04" in content
        assert "CUDA version `13.0`" in content
