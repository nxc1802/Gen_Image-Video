"""
tests/adversarial/test_adversarial_p0_it2_kaggle.py
Empirical Adversarial Test Suite for Phase 0 Iteration 2
(Kaggle Module Import Safety, Runner Local Execution, Kwargs Resilience, and Monitor Backward-Compatibility).

Adversarially probes:
1. Import Safety & Zero Network Calls:
   - Import all 11 modules in deployments/kaggle/ in an environment devoid of credentials.
   - Trap any network connection attempts (socket.socket, socket.create_connection, urllib, http.client).
   - Assert 0 import errors and 0 network attempts both in-process and in pristine subprocesses.
2. Runner Local Execution Subprocess:
   - Clean subprocess invocation of `python -m deployments.kaggle.runner --local`.
   - Clean subprocess invocation with explicit `--out` flag.
   - Assert exit code 0 and successful local certification output.
3. Hostile Kwargs & Parameter Fuzzing on KaggleDualT4Runner:
   - SQL injection, buffer injection, template injection, hostile casing in mode and kwargs.
   - Lazy client property auth boundary & credential leak checks.
4. Backward Compatibility of monitor_both_sessions:
   - Safe import of get_session_configs and run_monitor.
   - Dynamic code generation AST parsing.
   - Lazy attribute resolution via __getattr__ for legacy callers.
   - Mock-driven end-to-end monitor execution.
"""

import ast
import http.client
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import urllib.request

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class NetworkAttemptForbiddenError(RuntimeError):
    """Raised when an unintended network call is made during import or offline execution."""
    pass


def _forbidden_socket_connect(*args, **kwargs):
    raise NetworkAttemptForbiddenError(
        f"Unintended network attempt detected! socket.connect called with args={args}, kwargs={kwargs}"
    )


def _forbidden_create_connection(*args, **kwargs):
    raise NetworkAttemptForbiddenError(
        f"Unintended network attempt detected! socket.create_connection called with args={args}, kwargs={kwargs}"
    )


# ---------------------------------------------------------------------------
# 1. EMPIRICAL CHALLENGE 1: All 11 Modules Import Without Credentials or Network
# ---------------------------------------------------------------------------

class TestKaggleAllModulesZeroCredentialsZeroNetwork:
    """Empirically verifies that all 11 modules in deployments/kaggle/ import cleanly with 0 network calls."""

    KAGGLE_MODULES = [
        "deployments.kaggle.__init__",
        "deployments.kaggle.certify_kernel",
        "deployments.kaggle.live_gpu_test",
        "deployments.kaggle.monitor_both_sessions",
        "deployments.kaggle.prepare_datasets",
        "deployments.kaggle.run_high_quality_test",
        "deployments.kaggle.run_studio_quality_pipeline",
        "deployments.kaggle.runner",
        "deployments.kaggle.session1_visual_pipeline",
        "deployments.kaggle.session2_audio_pipeline",
        "deployments.kaggle.smoke_runner",
    ]

    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch, tmp_path):
        """Scrub all credentials and block network connections."""
        for env_key in [
            "KAGGLE_USERNAME",
            "KAGGLE_API_TOKEN",
            "KAGGLE_KEY",
            "KAGGLE_CONFIG_DIR",
            "KGAT_TOKEN",
            "FACTORY_API_KEY",
        ]:
            monkeypatch.delenv(env_key, raising=False)

        empty_config_dir = tmp_path / "empty_kaggle_conf"
        empty_config_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(empty_config_dir))

        # Block network sockets in-process
        monkeypatch.setattr(socket.socket, "connect", _forbidden_socket_connect)
        monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)

    def test_verify_exactly_11_modules_exist(self):
        """Asserts that exactly 11 Python modules exist in deployments/kaggle/."""
        kaggle_dir = REPO_ROOT / "deployments" / "kaggle"
        found_modules = sorted([f.stem for f in kaggle_dir.glob("*.py")])
        expected = [
            "__init__",
            "certify_kernel",
            "live_gpu_test",
            "monitor_both_sessions",
            "prepare_datasets",
            "run_high_quality_test",
            "run_studio_quality_pipeline",
            "runner",
            "session1_visual_pipeline",
            "session2_audio_pipeline",
            "smoke_runner",
        ]
        assert found_modules == expected, f"Module mismatch in deployments/kaggle: {found_modules}"

    @pytest.mark.parametrize("module_name", KAGGLE_MODULES)
    def test_module_imports_in_process_without_credentials_or_network(self, module_name):
        """Imports each module in-process under socket interception."""
        mod = importlib.import_module(module_name)
        assert mod is not None
        assert hasattr(mod, "__file__")

    @pytest.mark.parametrize("module_name", KAGGLE_MODULES)
    def test_module_imports_isolated_subprocess_without_credentials_or_network(self, module_name, tmp_path):
        """Imports each module in an isolated subprocess completely free of cached imports and credentials."""
        clean_env = dict(os.environ)
        for k in [
            "KAGGLE_USERNAME",
            "KAGGLE_API_TOKEN",
            "KAGGLE_KEY",
            "KAGGLE_CONFIG_DIR",
            "KGAT_TOKEN",
            "FACTORY_API_KEY",
        ]:
            clean_env.pop(k, None)

        empty_dir = tmp_path / "subprocess_no_kaggle"
        empty_dir.mkdir(parents=True, exist_ok=True)
        clean_env["KAGGLE_CONFIG_DIR"] = str(empty_dir)
        clean_env["PYTHONPATH"] = str(REPO_ROOT)

        script = f"""
import socket, sys, importlib

def _block(*args, **kwargs):
    raise RuntimeError("Unintended network connection attempted during import of {module_name}!")

socket.socket.connect = _block
socket.create_connection = _block

mod = importlib.import_module("{module_name}")
assert mod is not None
print("IMPORT_OK")
"""
        res = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert res.returncode == 0, f"Module {module_name} failed to import cleanly:\nSTDERR: {res.stderr}\nSTDOUT: {res.stdout}"
        assert "IMPORT_OK" in res.stdout


# ---------------------------------------------------------------------------
# 2. EMPIRICAL CHALLENGE 2: Clean Subprocess `runner --local` Execution
# ---------------------------------------------------------------------------

class TestKaggleRunnerSubprocessExecution:
    """Empirically executes the runner CLI via clean subprocess without credentials."""

    def _get_clean_env(self, tmp_path: Path) -> Dict[str, str]:
        env = dict(os.environ)
        for k in [
            "KAGGLE_USERNAME",
            "KAGGLE_API_TOKEN",
            "KAGGLE_KEY",
            "KAGGLE_CONFIG_DIR",
            "KGAT_TOKEN",
            "FACTORY_API_KEY",
        ]:
            env.pop(k, None)
        empty_dir = tmp_path / "nonexistent_kaggle_conf"
        empty_dir.mkdir(parents=True, exist_ok=True)
        env["KAGGLE_CONFIG_DIR"] = str(empty_dir)
        env["PYTHONPATH"] = str(REPO_ROOT)
        return env

    def test_runner_local_subprocess_exit_zero(self, tmp_path):
        """Executes `python -m deployments.kaggle.runner --local` in a clean subprocess."""
        clean_env = self._get_clean_env(tmp_path)
        out_file = tmp_path / "cli_cert_results.json"

        cmd = [
            sys.executable,
            "-m",
            "deployments.kaggle.runner",
            "--local",
            "--out",
            str(out_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CLI runner failed with returncode {result.returncode}:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Running Tier 1 model certification locally..." in result.stdout
        assert "Certification finished. Success: True" in result.stdout
        assert out_file.exists(), "Expected certification results JSON file was not created!"

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data.get("all_gates_passed") is True
        assert "models" in data
        assert len(data["models"]) >= 3

    def test_runner_local_default_out_subprocess(self, tmp_path):
        """Executes `python -m deployments.kaggle.runner --local` with default output path."""
        clean_env = self._get_clean_env(tmp_path)
        cmd = [
            sys.executable,
            "-m",
            "deployments.kaggle.runner",
            "--local",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CLI runner failed with returncode {result.returncode}:\n{result.stderr}"
        assert "Certification finished. Success: True" in result.stdout


# ---------------------------------------------------------------------------
# 3. EMPIRICAL CHALLENGE 3: Hostile / Arbitrary Kwargs & Mode Injection
# ---------------------------------------------------------------------------

class TestKaggleDualT4RunnerKwargsAndAdversarial:
    """Stress-tests KaggleDualT4Runner with hostile kwargs, mode fuzzing, and auth boundaries."""

    def test_hostile_kwargs_injection_exact_dispatch_spec(self, tmp_path):
        """Directly tests the dispatch requirement: KaggleDualT4Runner(mode='preflight', evil_flag='DROP TABLE', timeout=99999)."""
        from deployments.kaggle.runner import KaggleDualT4Runner

        runner = KaggleDualT4Runner(
            mode="preflight",
            evil_flag="DROP TABLE",
            timeout=99999,
        )

        assert runner.mode == "preflight"
        assert runner.kwargs.get("evil_flag") == "DROP TABLE"
        assert runner.extra_options.get("evil_flag") == "DROP TABLE"
        assert runner.kwargs.get("timeout") == 99999

        # Ensure local certification ignores hostile kwargs without failing
        out_path = tmp_path / "evil_cert.json"
        res = runner.run_local_certification(output_path=out_path)
        assert res.get("all_gates_passed") is True

        # Ensure unified run() dispatcher passes cleanly
        res_unified = runner.run(output_path=out_path)
        assert res_unified.get("all_gates_passed") is True

    @pytest.mark.parametrize(
        "hostile_mode,expected_normalized",
        [
            ("PREFLIGHT", "preflight"),
            ("  preflight  ", "preflight"),
            ("local", "local"),
            ("  LOCAL\t", "local"),
            ("auto", "auto"),
            ("remote", "remote"),
            ("DROP TABLE users;", "drop table users;"),
            ("'; SELECT * FROM kernels; --", "'; select * from kernels; --"),
            (None, "remote"),
            (12345, "remote"),
            (b"preflight", "remote"),
            ([], "remote"),
            ({}, "remote"),
        ],
    )
    def test_hostile_mode_fuzzing(self, hostile_mode, expected_normalized):
        """Asserts KaggleDualT4Runner mode normalization handles arbitrary non-string or hostile casing."""
        from deployments.kaggle.runner import KaggleDualT4Runner

        runner = KaggleDualT4Runner(mode=hostile_mode)
        assert runner.mode == expected_normalized

    def test_hostile_arbitrary_kwargs_types(self):
        """Injects deep dictionaries, recursive objects, format strings, and giant strings into kwargs."""
        from deployments.kaggle.runner import KaggleDualT4Runner

        hostile_kwargs = {
            "sql_injection": "1' OR '1'='1",
            "template_injection": "{{ 7 * 7 }}",
            "format_string": "%(secret)s",
            "null_byte": "admin\x00root",
            "giant_string": "A" * 50000,
            "deep_dict": {"a": {"b": {"c": [1, 2, 3]}}},
            "lambda_fn": lambda x: x,
            "int_overflow": 2**128,
            "special_chars": "🚀!@#$%^&*()_+{}[]:;\"'<>?,./~`",
        }

        runner = KaggleDualT4Runner(mode="preflight", **hostile_kwargs)

        assert runner.kwargs["sql_injection"] == "1' OR '1'='1"
        assert runner.kwargs["giant_string"] == "A" * 50000
        assert runner.kwargs["int_overflow"] == 2**128
        assert runner.extra_options["format_string"] == "%(secret)s"

    def test_client_property_auth_boundary(self, monkeypatch, tmp_path):
        """Accessing runner.client without credentials must raise KaggleAuthError cleanly without leaking secrets."""
        for k in ["KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no_kaggle"))

        from deployments.kaggle.runner import KaggleAuthError, KaggleDualT4Runner

        runner = KaggleDualT4Runner(mode="preflight")
        with pytest.raises(KaggleAuthError) as exc_info:
            _ = runner.client
        assert "Kaggle username not configured" in str(exc_info.value)

    def test_client_property_with_explicit_or_injected_client(self):
        """Providing client explicitly preserves client instance and does not attempt env auth."""
        from deployments.kaggle.runner import KaggleAPIClient, KaggleDualT4Runner

        mock_client = KaggleAPIClient(username="test_user", api_key="test_key")
        runner = KaggleDualT4Runner(client=mock_client, mode="preflight")

        assert runner.client is mock_client
        assert runner.client.username == "test_user"


# ---------------------------------------------------------------------------
# 4. EMPIRICAL CHALLENGE 4: Backward Compatibility of monitor_both_sessions
# ---------------------------------------------------------------------------

class TestMonitorBothSessionsCompatibilityAndExecution:
    """Verifies get_session_configs, run_monitor, and lazy __getattr__ compatibility."""

    def test_get_session_configs_contract_and_structure(self, tmp_path):
        """Asserts get_session_configs generates full session structure with valid kernel code."""
        from deployments.kaggle.monitor_both_sessions import get_session_configs

        configs = get_session_configs(
            username="certified_developer",
            repo_root=tmp_path,
            extra_artifact_dir=str(tmp_path / "extra_artifacts"),
        )

        assert "visual" in configs
        assert "audio" in configs

        for session_name in ("visual", "audio"):
            s = configs[session_name]
            assert "slug" in s and s["slug"].startswith("factory-v4-lite-")
            assert "title" in s
            assert "code_fn" in s and callable(s["code_fn"])
            assert "dataset_sources" in s and isinstance(s["dataset_sources"], list)
            assert "out_dir" in s and isinstance(s["out_dir"], Path)
            assert "artifact_dir" in s
            assert s["done"] is False
            assert s["last_log_line"] == 0

            # Generate kernel code and verify it is non-trivial valid Python
            code_str = s["code_fn"]()
            assert isinstance(code_str, str)
            assert len(code_str) > 100
            # Validate AST syntax
            parsed_ast = ast.parse(code_str)
            assert parsed_ast is not None

        # Verify dataset sources reference the parameterized username
        assert "certified_developer/qwen38-27b-gguf" in configs["visual"]["dataset_sources"]
        assert "certified_developer/vieneu-tts-v3-turbo" in configs["audio"]["dataset_sources"]

    def test_monitor_both_sessions_lazy_getattr_uncredentialed(self, monkeypatch, tmp_path):
        """When uncredentialed, accessing legacy module attributes raises KaggleAuthError cleanly."""
        for k in ["KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no_kaggle"))

        import deployments.kaggle.monitor_both_sessions as mbs

        # Catch through base Exception and assert error message/type name to remain immune to re-import split-brain
        with pytest.raises(Exception) as exc_info:
            _ = mbs.client
        assert "Kaggle username not configured" in str(exc_info.value)
        assert exc_info.type.__name__ == "KaggleAuthError"

        with pytest.raises(Exception) as exc_info:
            _ = mbs.sessions
        assert "Kaggle username not configured" in str(exc_info.value)
        assert exc_info.type.__name__ == "KaggleAuthError"

        with pytest.raises(AttributeError) as exc_info:
            _ = mbs.unknown_symbol_xyz
        assert "has no attribute 'unknown_symbol_xyz'" in str(exc_info.value)

    def test_monitor_both_sessions_lazy_getattr_credentialed(self, monkeypatch, tmp_path):
        """When credentials are present, accessing legacy module attributes returns valid objects."""
        monkeypatch.setenv("KAGGLE_USERNAME", "legacy_user")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "legacy_token")

        import deployments.kaggle.monitor_both_sessions as mbs

        client = mbs.client
        assert type(client).__name__ == "KaggleAPIClient"
        assert client.username == "legacy_user"

        sessions = mbs.sessions
        assert "visual" in sessions
        assert "audio" in sessions
        assert "legacy_user/qwen38-27b-gguf" in sessions["visual"]["dataset_sources"]

    def test_run_monitor_end_to_end_mocked(self, tmp_path):
        """Executes run_monitor with a mocked KaggleAPIClient to verify complete monitoring lifecycle."""
        from deployments.kaggle.monitor_both_sessions import run_monitor
        from deployments.kaggle.runner import KaggleAPIClient

        mock_client = MagicMock(spec=KaggleAPIClient)
        mock_client.username = "mock_tester"
        mock_client.push_kernel.return_value = {
            "url": "https://www.kaggle.com/mock_tester/kernel",
            "ref": "mock_tester/kernel",
        }
        mock_client.get_kernel_status.return_value = {"status": "complete", "failureMessage": None}
        mock_client.get_kernel_output.return_value = {
            "log": "Step 1: Init\nStep 2: Done\n",
            "files": [],
        }

        results = run_monitor(
            client=mock_client,
            extra_artifact_dir=str(tmp_path / "mock_artifacts"),
            poll_interval_sec=0.01,
        )

        assert "visual" in results
        assert "audio" in results
        assert results["visual"]["log"] == "Step 1: Init\nStep 2: Done\n"
        assert results["audio"]["log"] == "Step 1: Init\nStep 2: Done\n"
        assert mock_client.push_kernel.call_count == 2
