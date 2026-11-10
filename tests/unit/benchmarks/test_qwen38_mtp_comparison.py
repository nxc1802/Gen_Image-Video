"""
tests/unit/benchmarks/test_qwen38_mtp_comparison.py
Unit tests for Qwen3.8 MTP (Multi-Token Prediction) Speculative Decoding Comparison Suite.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import pytest

from benchmarks.runners.mtp_baseline_runner import (
    MTPComparisonReport,
    MTPRunResult,
    run_qwen38_mtp_comparison,
)
from deployments.kaggle.session1_visual_pipeline import (
    run_qwen38_mtp_comparison as exported_mtp_comparison,
)


def test_qwen38_mtp_comparison_identical_parameters():
    """Verifies baseline and MTP runs strictly enforce identical prompt, seed, and token limits."""
    prompt = "Explain speculative decoding with multi-token prediction."
    seed = 12345
    max_tokens = 256
    temperature = 0.5

    report = run_qwen38_mtp_comparison(
        prompt=prompt,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    assert isinstance(report, MTPComparisonReport)
    assert report.baseline.seed == seed
    assert report.mtp.seed == seed
    assert report.baseline.prompt == prompt
    assert report.mtp.prompt == prompt
    assert report.baseline.max_tokens == max_tokens
    assert report.mtp.max_tokens == max_tokens
    assert report.baseline.temperature == temperature
    assert report.mtp.temperature == temperature


def test_qwen38_mtp_comparison_speedup_calculation():
    """Verifies that speedup factor is mathematically authentic and reflects throughput gains."""
    def mock_eval(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
        tps = 24.5 if is_mtp else 15.0
        dur = (max_tokens / tps) * 1000.0
        return {
            "content": f"Decoded text (mtp={is_mtp})",
            "tokens_predicted": max_tokens,
            "duration_ms": round(dur, 2),
            "ttft_ms": 35.0,
            "decode_tokens_per_second": tps,
            "prompt_tokens_per_second": 250.0,
            "vram_mb": 13200.0 if is_mtp else 11600.0,
            "phases": {
                "model_loading_ms": 0.0,
                "prompt_processing_ms": 35.0,
                "decode_ms": round(dur - 35.0, 2),
                "sampling_ms": 2.0,
                "vae_ms": 0.0,
                "artifact_encoding_ms": 1.0,
            },
        }

    report = run_qwen38_mtp_comparison(
        prompt="Test speedup",
        seed=42,
        max_tokens=300,
        mock_evaluator=mock_eval,
    )

    expected_speedup = round(24.5 / 15.0, 2)  # 1.63x
    assert report.speedup_factor == expected_speedup
    assert report.speedup_factor >= 1.25
    assert report.invariants_passed is True


def test_qwen38_mtp_comparison_vram_overhead_bounded():
    """Verifies that MTP draft model VRAM overhead is calculated and bounded <= 2048 MB."""
    def mock_eval(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
        return {
            "content": "output",
            "tokens_predicted": 50,
            "duration_ms": 2000.0,
            "ttft_ms": 30.0,
            "decode_tokens_per_second": 25.0 if is_mtp else 16.0,
            "prompt_tokens_per_second": 200.0,
            "vram_mb": 13300.0 if is_mtp else 11700.0,  # 1600 MB overhead
            "phases": {
                "model_loading_ms": 0.0,
                "prompt_processing_ms": 30.0,
                "decode_ms": 1970.0,
                "sampling_ms": 0.0,
                "vae_ms": 0.0,
                "artifact_encoding_ms": 0.0,
            },
        }

    report = run_qwen38_mtp_comparison(mock_evaluator=mock_eval)
    assert report.vram_overhead_mb == 1600.0
    assert report.vram_overhead_mb <= 2048.0
    assert report.invariants_details["vram_overhead_bounded"] is True


def test_qwen38_mtp_comparison_authentic_ttft_and_phases():
    """Verifies authentic TTFT and that LLM phases have vae_ms == 0.0."""
    report = run_qwen38_mtp_comparison()

    assert report.baseline.ttft_ms > 0
    assert report.mtp.ttft_ms > 0
    assert report.baseline.phases["vae_ms"] == 0.0
    assert report.mtp.phases["vae_ms"] == 0.0
    assert "prompt_processing_ms" in report.baseline.phases
    assert "decode_ms" in report.baseline.phases


def test_qwen38_mtp_comparison_json_serialization():
    """Verifies JSON serialization and file export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "mtp_acceleration_report.json"
        report = run_qwen38_mtp_comparison(output_path=out_file)

        assert out_file.exists()
        saved_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert saved_data["model_id"] == "qwen-text"
        assert saved_data["speedup_factor"] == report.speedup_factor
        assert "baseline" in saved_data
        assert "mtp" in saved_data


def test_session1_exported_mtp_comparison():
    """Verifies that session1_visual_pipeline exports run_qwen38_mtp_comparison."""
    assert callable(exported_mtp_comparison)
    rep = exported_mtp_comparison()
    assert isinstance(rep, MTPComparisonReport)
