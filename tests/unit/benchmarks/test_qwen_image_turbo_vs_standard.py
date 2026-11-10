"""
tests/unit/benchmarks/test_qwen_image_turbo_vs_standard.py
Unit tests for Qwen-Image 2.1 Turbo (8-step) vs Standard (20-step) benchmark suite.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import pytest

from benchmarks.runners.qwen_image_variant_runner import (
    ImageVariantRunResult,
    QwenImageVariantReport,
    benchmark_qwen_image_variants,
)
from deployments.kaggle.session1_visual_pipeline import (
    benchmark_qwen_image_variants as exported_image_variants,
)


def test_qwen_image_variants_steps_and_unet():
    """Verifies standard is 20 steps / Q4_K_M UNet and turbo is 8 steps / Turbo UNet."""
    report = benchmark_qwen_image_variants()

    assert isinstance(report, QwenImageVariantReport)
    assert report.standard.steps == 20
    assert report.standard.cfg == 4.0
    assert "turbo" not in report.standard.unet_name.lower()

    assert report.turbo.steps == 8
    assert report.turbo.cfg == 1.0
    assert "turbo" in report.turbo.unet_name.lower()


def test_qwen_image_variants_speedup_and_step_rate_ratio():
    """Verifies that Turbo achieves >= 1.5x acceleration while preserving step-rate invariance."""
    report = benchmark_qwen_image_variants()

    assert report.speedup_factor >= 1.5
    # Step rate ratio (seconds_per_step_std / seconds_per_step_turbo) must remain near 1.0 (0.7..1.4)
    assert 0.7 <= report.step_rate_ratio <= 1.4
    assert report.invariants_passed is True


def test_qwen_image_variants_vram_bounds():
    """Verifies that both variants strictly respect the 14,540 MB VRAM ceiling on Tesla T4."""
    report = benchmark_qwen_image_variants()

    assert report.standard.peak_vram_mb <= 14540.0
    assert report.turbo.peak_vram_mb <= 14540.0
    assert report.invariants_details["vram_safe_standard"] is True
    assert report.invariants_details["vram_safe_turbo"] is True


def test_qwen_image_variants_perceptual_std():
    """Verifies perceptual non-collapse check (standard deviation > 20.0)."""
    report = benchmark_qwen_image_variants()

    assert report.standard.image_std_dev > 20.0
    assert report.turbo.image_std_dev > 20.0
    assert report.invariants_details["perceptual_std_standard"] is True
    assert report.invariants_details["perceptual_std_turbo"] is True


def test_qwen_image_variants_json_report_export():
    """Verifies JSON serialization and file export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        report = benchmark_qwen_image_variants(output_dir=out_dir)

        report_file = out_dir / "qwen_image_variant_benchmark.json"
        assert report_file.exists()

        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["model_id"] == "qwen-image-2.1"
        assert data["speedup_factor"] == report.speedup_factor
        assert "standard" in data
        assert "turbo" in data
        assert data["invariants_passed"] is True


def test_session1_exported_image_benchmark():
    """Verifies that session1_visual_pipeline exports benchmark_qwen_image_variants."""
    assert callable(exported_image_variants)
    rep = exported_image_variants()
    assert isinstance(rep, QwenImageVariantReport)
