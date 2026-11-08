"""
tests/unit/test_report_builder.py
Unit and regression test suite for Phase 6 Benchmark Report Builder.

Verifies:
1. BenchmarkReportBuilder Python API and CLI execution.
2. Generation, existence, and non-empty size (>500 bytes) of all 6 acceptance reports.
3. Strict mathematical consistency with results.jsonl (163 cases: 51 PASS, 111 HARDWARE_BLOCKED, 1 NOT_IMPLEMENTED, 0 FAIL, 0 ERROR).
4. Conformance to Plan §10.5 and §10.7 required sections and truth model criteria.
5. Ingestion of Kaggle Dual Tesla T4 hardware attested telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import pytest

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


@pytest.fixture
def canonical_builder(tmp_path: Path) -> BenchmarkReportBuilder:
    """Fixture providing a builder configured to write into an isolated temporary directory."""
    return BenchmarkReportBuilder(
        run_dir=CANONICAL_RUN_DIR,
        output_dir=tmp_path,
        config_dir=CANONICAL_CONFIG_DIR,
        kaggle_report=CANONICAL_KAGGLE_REPORT,
    )


class TestBenchmarkReportBuilderCore:
    """Core tests for data ingestion, mathematical fidelity, and report generation."""

    def test_builder_load_data_integrity(self, canonical_builder: BenchmarkReportBuilder):
        """Verifies that all 163 matrix results and configs load with mathematical precision."""
        canonical_builder.load_data()

        assert len(canonical_builder.results) == 163, "Expected exactly 163 evaluated results"
        assert len(canonical_builder.cases_by_id) == 163, "Expected 163 unique configured cases"

        # Mathematical breakdown
        assert canonical_builder.status_counts["PASS"] == 51
        assert canonical_builder.status_counts["HARDWARE_BLOCKED"] == 111
        assert canonical_builder.status_counts["NOT_IMPLEMENTED"] == 1
        assert canonical_builder.status_counts.get("FAIL", 0) == 0
        assert canonical_builder.status_counts.get("ERROR", 0) == 0

        # Subsystems verification (15 operational groups)
        assert len(canonical_builder.group_counts) == 15
        assert sum(sum(cnts.values()) for cnts in canonical_builder.group_counts.values()) == 163

    def test_builder_generates_all_six_reports_in_output_dir(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Verifies that all 6 specified markdown reports are generated and exceed size thresholds."""
        files: Dict[str, Path] = canonical_builder.generate_all_reports()

        assert len(files) == 6
        for expected_name in EXPECTED_REPORT_FILES:
            assert expected_name in files, f"Missing generated file: {expected_name}"
            target_path = tmp_path / expected_name
            assert target_path.exists(), f"File does not exist on disk: {target_path}"
            file_size = target_path.stat().st_size
            assert file_size > 500, f"Report {expected_name} is unexpectedly small: {file_size} bytes"
            # In practice, each comprehensive report is > 5,000 bytes
            assert file_size > 4000, f"Report {expected_name} lacks full depth: {file_size} bytes"

    def test_executive_summary_schema_and_math(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates executive_summary.md contents, headers, and exact metrics."""
        canonical_builder.generate_all_reports()
        exec_summary = (tmp_path / "executive_summary.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Benchmark Executive Summary" in exec_summary
        assert "## 1. High-Level Outcome & Zero-Hidden-Cases Guarantee" in exec_summary
        assert "## 2. Subsystem Matrix Summary" in exec_summary
        assert "## 3. Platform Comparison & Execution Integrity" in exec_summary

        # Check exact mathematical numbers
        assert "163" in exec_summary
        assert "**51**" in exec_summary
        assert "**111**" in exec_summary
        assert "**1**" in exec_summary
        assert "31.29%" in exec_summary
        assert "68.10%" in exec_summary

        # Integrity guarantees
        assert "mock_used: False" in exec_summary
        assert "fallback_used: False" in exec_summary

    def test_capability_matrix_truth_model_and_subsystems(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates capability_matrix.md for Truth Model representation and all 15 subsystems."""
        canonical_builder.generate_all_reports()
        cap_matrix = (tmp_path / "capability_matrix.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Capability Matrix & Truth Model" in cap_matrix
        assert "## 1. Truth Model Architecture" in cap_matrix
        assert "## 2. Capability Matrix by Subsystem" in cap_matrix
        assert "## 3. Variant-Safe Routing & Model Independence" in cap_matrix

        # Check all 15 operational groups are present in headers
        expected_groups = [
            "acestep", "cross_feature", "decision", "embedding", "enhancement",
            "gpu_runtime", "image", "minimax_video", "router_infrastructure",
            "speech", "text", "video_integration", "vlm", "wan_video", "yue2",
        ]
        for grp in expected_groups:
            assert f"`{grp}`" in cap_matrix

        # Truth Model column validation
        assert "Declared" in cap_matrix
        assert "Implemented" in cap_matrix
        assert "Live Verified (Darwin)" in cap_matrix
        assert "Live Verified (Dual T4)" in cap_matrix

        # Check ACE-23 is explicitly marked NOT_IMPLEMENTED
        assert "`ACE-23`" in cap_matrix
        assert "Workflow contract absence explicitly recorded" in cap_matrix

    def test_model_comparison_prompts_and_rubric(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates model_comparison.md for B-M01..B-M12 prompts and 8 listening dimensions."""
        canonical_builder.generate_all_reports()
        comparison = (tmp_path / "model_comparison.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Comparative Model Analysis: ACE-Step 1.5 vs. YuE2" in comparison
        assert "## 1. Architectural Foundations & Design Paradigms" in comparison
        assert "## 2. Master Evaluation Prompt Matrix (B-M01 to B-M12)" in comparison
        assert "## 3. Head-to-Head Prompt Evaluation & Qualitative Analysis" in comparison
        assert "## 4. Human Listening Evaluation Rubric (Plan §10.4)" in comparison
        assert "## 5. Benchmarking Fairness & Compute Accounting" in comparison

        # Check all prompts B-M01 to B-M12
        for i in range(1, 13):
            pid = f"B-M{i:02d}"
            assert f"`{pid}`" in comparison

        # Check listening rubric criteria
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
            assert dim in comparison

    def test_latency_vram_telemetry_and_attestation(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates latency_vram.md for statistical measures and Dual Tesla T4 attestation."""
        canonical_builder.generate_all_reports()
        telemetry = (tmp_path / "latency_vram.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Latency & VRAM Telemetry Analysis" in telemetry
        assert "## 1. Local Execution Latency Telemetry (Darwin Apple Silicon)" in telemetry
        assert "## 2. Subsystem Latency & Execution Distribution" in telemetry
        assert "## 3. Kaggle Dual Tesla T4 Attested Hardware Telemetry" in telemetry
        assert "## 4. VRAM Budget & Memory Safety Margin Analysis" in telemetry

        # Check attested Tesla T4 references
        assert "Tesla T4" in telemetry
        assert "15,360.0 MB" in telemetry
        assert "14,540.0 MB" in telemetry
        assert "820 MB" in telemetry
        assert "CUDA version `13.0`" in telemetry

    def test_failure_analysis_and_historical_fixes(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates failure_analysis.md for taxonomy, FIX-01..20, and VULN audits."""
        canonical_builder.generate_all_reports()
        failures_doc = (tmp_path / "failure_analysis.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Failure Analysis & Historical Fix Audit" in failures_doc
        assert "## 1. Transparent Root-Cause Taxonomy" in failures_doc
        assert "## 2. Hardware Blocker Breakdown by Subsystem" in failures_doc
        assert "## 3. Unimplemented Feature Analysis: ACE-23" in failures_doc
        assert "## 4. Comprehensive Historical Fix Audit Log" in failures_doc

        # Check all Phase 0 fixes FIX-01 to FIX-20
        for i in range(1, 21):
            fix_id = f"FIX-{i:02d}"
            assert f"**{fix_id}**" in failures_doc, f"Missing {fix_id} in failure audit log"

        # Check Phase 3 vulnerabilities VULN-P3-01 to VULN-P3-05
        for i in range(1, 6):
            vuln_id = f"VULN-P3-0{i}"
            assert f"**{vuln_id}**" in failures_doc, f"Missing {vuln_id} in Phase 3 audit log"

        # Check Phase 4 vulnerabilities VULN-P4-01 to VULN-P4-04
        for i in range(1, 5):
            vuln_id = f"VULN-P4-0{i}"
            assert f"**{vuln_id}**" in failures_doc, f"Missing {vuln_id} in Phase 4 audit log"

    def test_final_report_eleven_sections_and_163_cases(
        self, canonical_builder: BenchmarkReportBuilder, tmp_path: Path
    ):
        """Validates final_report.md synthesizing sections 1-11 and itemizing all 163 cases."""
        canonical_builder.generate_all_reports()
        final_doc = (tmp_path / "final_report.md").read_text(encoding="utf-8")

        assert "# Factory v4 Lite — Master Benchmark & Acceptance Report" in final_doc

        # Verify all 11 sections per Plan §10.7
        for sec_num in range(1, 12):
            assert f"## {sec_num}." in final_doc, f"Section {sec_num} missing from final_report.md"

        # Verify that all 163 case IDs appear in the Master Functional Test Results table
        for r in canonical_builder.results:
            tid = r["test_id"]
            assert f"`{tid}`" in final_doc, f"Test case ID {tid} missing from final_report.md"


class TestBenchmarkReportBuilderCLI:
    """Verifies CLI execution, argument parsing, and error handling."""

    def test_cli_execution_via_python_module(self, tmp_path: Path):
        """Runs python -m benchmarks.runners.report_builder with custom output dir."""
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.runners.report_builder",
            "--run-dir",
            str(CANONICAL_RUN_DIR),
            "--output-dir",
            str(tmp_path),
            "--config-dir",
            str(CANONICAL_CONFIG_DIR),
            "--kaggle-report",
            str(CANONICAL_KAGGLE_REPORT),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        assert res.returncode == 0, f"CLI execution failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "Successfully generated 6 reports" in res.stdout

        for filename in EXPECTED_REPORT_FILES:
            assert (tmp_path / filename).exists()

    def test_cli_main_function(self, tmp_path: Path):
        """Direct call to main() function with argument vector."""
        ret = report_builder_main([
            "--run-dir", str(CANONICAL_RUN_DIR),
            "--output-dir", str(tmp_path),
            "--config-dir", str(CANONICAL_CONFIG_DIR),
            "--kaggle-report", str(CANONICAL_KAGGLE_REPORT),
        ])
        assert ret == 0
        for filename in EXPECTED_REPORT_FILES:
            assert (tmp_path / filename).exists()

    def test_builder_raises_file_not_found_on_missing_run_dir(self, tmp_path: Path):
        """Builder raises FileNotFoundError when metadata.json is not present."""
        bad_run_dir = tmp_path / "non_existent_run"
        builder = BenchmarkReportBuilder(run_dir=bad_run_dir, output_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            builder.load_data()

    def test_builder_raises_file_not_found_on_missing_config_dir(self, tmp_path: Path):
        """Builder raises FileNotFoundError when evaluation_cases.yaml is missing."""
        bad_config_dir = tmp_path / "non_existent_configs"
        builder = BenchmarkReportBuilder(
            run_dir=CANONICAL_RUN_DIR,
            output_dir=tmp_path,
            config_dir=bad_config_dir,
        )
        with pytest.raises(FileNotFoundError):
            builder.load_data()
