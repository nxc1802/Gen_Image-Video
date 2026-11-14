"""
tests/unit/deployments/test_kernel_smoke.py
Automated Smoke Tests for Kaggle Deployment Kernel Scripts & Runners (P1.7):
1. Python AST syntax validity for all standalone generated scripts
2. Credential safety: zero leaked secrets, token masking, and secure env resolution
3. Correct parameter resolution and CLI entrypoints
"""
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict
from unittest.mock import patch, MagicMock
import pytest

from deployments.kaggle.runner import (
    KaggleAPIClient,
    KaggleAuthError,
    KaggleDualT4Runner,
    resolve_kaggle_api_token,
    resolve_kaggle_username,
)
from deployments.kaggle.certify_kernel import (
    Tier1ModelCertifier,
    TIER_1_PROFILES,
)
from deployments.kaggle.session1_visual_pipeline import generate_kernel_code as gen_s1
from deployments.kaggle.session2_audio_pipeline import generate_kernel_code as gen_s2
from deployments.kaggle.run_studio_quality_pipeline import generate_kernel_code as gen_studio
from deployments.kaggle.run_high_quality_test import generate_high_quality_kernel_code as gen_hq
from deployments.kaggle.live_gpu_test import generate_kaggle_gpu_kernel_code as gen_live

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ==============================================================================
# Suite 1: Python AST Syntax Validity
# ==============================================================================

class TestKernelAstValidity:
    """Asserts that all standalone generated scripts compile to valid Python AST."""

    GENERATORS = [
        ("session1_visual_pipeline", gen_s1),
        ("session2_audio_pipeline", gen_s2),
        ("run_studio_quality_pipeline", gen_studio),
        ("run_high_quality_test", gen_hq),
        ("live_gpu_test", gen_live),
    ]

    @pytest.mark.parametrize("name,gen_fn", GENERATORS)
    def test_generated_kernel_ast_validity(self, name: str, gen_fn):
        code = gen_fn()
        assert isinstance(code, str) and len(code) > 500, f"Generator {name} produced empty or short output"
        try:
            tree = ast.parse(code)
            assert len(tree.body) > 10, f"Generator {name} has fewer than 10 AST body statements"
        except SyntaxError as e:
            pytest.fail(f"Generator {name} produced invalid Python AST at line {e.lineno}:{e.offset} - {e.msg}")

    def test_runner_prepare_kernel_script_ast_validity(self):
        runner = KaggleDualT4Runner(mode="preflight")
        script = runner.prepare_kernel_script()
        assert isinstance(script, str) and len(script) > 1000
        try:
            tree = ast.parse(script)
            assert len(tree.body) > 20
        except SyntaxError as e:
            pytest.fail(f"runner.prepare_kernel_script produced invalid Python AST: {e}")


# ==============================================================================
# Suite 2: Credential Safety & Zero Leaked Secrets
# ==============================================================================

class TestKernelCredentialSafety:
    """Asserts zero hardcoded credentials, safe env handling, and proper token masking."""

    SECRET_PATTERNS = [
        re.compile(r"KGAT_[a-zA-Z0-9]{20,}"),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH)?\s*PRIVATE KEY-----"),
        re.compile(r"(?<![A-Z0-9])(AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"),
    ]

    @pytest.mark.parametrize("name,gen_fn", [
        ("session1_visual", gen_s1),
        ("session2_audio", gen_s2),
        ("studio_quality", gen_studio),
        ("high_quality", gen_hq),
        ("live_gpu", gen_live),
    ])
    def test_generated_kernel_contains_zero_leaked_secrets(self, name: str, gen_fn):
        code = gen_fn()
        for pat in self.SECRET_PATTERNS:
            match = pat.search(code)
            assert match is None, f"Found leaked secret {match.group(0)[:8]}... in {name}"

    def test_kaggle_client_repr_masks_api_key(self):
        client = KaggleAPIClient(username="testuser", api_key="KGAT_super_secret_token_123456789")
        rep = repr(client)
        assert "super_secret" not in rep
        assert "KGAT...6789" in rep

    def test_env_token_resolution_precedence(self, monkeypatch):
        # 1. Parameter takes precedence
        monkeypatch.setenv("KAGGLE_API_TOKEN", "env_token_1")
        monkeypatch.setenv("KAGGLE_KEY", "env_key_2")
        tok = resolve_kaggle_api_token(explicit_api_key="explicit_token")
        assert tok == "explicit_token"

        # 2. Standard KAGGLE_API_TOKEN takes precedence over legacy KAGGLE_KEY
        tok_env = resolve_kaggle_api_token()
        assert tok_env == "env_token_1"

        # 3. Fallback to KAGGLE_KEY
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        tok_legacy = resolve_kaggle_api_token()
        assert tok_legacy == "env_key_2"

        # 4. Fail fast if none provided
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/nonexistent")
        with pytest.raises(KaggleAuthError):
            resolve_kaggle_api_token()


# ==============================================================================
# Suite 3: Parameter Resolution & CLI Entrypoints
# ==============================================================================

class TestKernelCliAndParameterResolution:
    """Verifies CLI entrypoints and parameter passing."""

    TARGET_SCRIPTS = [
        "deployments/kaggle/runner.py",
        "deployments/kaggle/certify_kernel.py",
        "deployments/kaggle/session1_visual_pipeline.py",
        "deployments/kaggle/session2_audio_pipeline.py",
        "deployments/kaggle/live_gpu_test.py",
    ]

    @pytest.mark.parametrize("rel_path", TARGET_SCRIPTS)
    def test_cli_help_flag_exits_cleanly(self, rel_path: str):
        full_path = REPO_ROOT / rel_path
        res = subprocess.run([sys.executable, str(full_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "show this help message and exit" in res.stdout

    def test_certify_kernel_preflight_execution(self, tmp_path, legacy_catalog):
        out_file = tmp_path / "cert_test.json"
        certifier = Tier1ModelCertifier(manifests_dir=legacy_catalog, mode="preflight")
        results = certifier.execute_all(output_path=out_file)
        assert results["all_gates_passed"] is True
        assert out_file.exists()

    def test_runner_local_mode_dispatch(self, tmp_path, legacy_workspace):
        out_file = tmp_path / "runner_test.json"
        runner = KaggleDualT4Runner(workspace_dir=legacy_workspace, mode="preflight")
        results = runner.run(output_path=out_file)
        assert results["all_gates_passed"] is True
        assert out_file.exists()
