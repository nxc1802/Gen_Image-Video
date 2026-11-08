"""
tests/adversarial/test_adversarial_p5_matrix_challenger.py

Milestone Phase 5 Adversarial Challenger Probes:
Comprehensive stress testing of the 163-case benchmark matrix infrastructure,
runner robustness, input fault injection, lease lifecycle isolation, and zero mock PASS leaks.

Adheres strictly to:
- Factory_v4_Lite_Full_Implementation_and_Benchmark_Plan.md §9, §10.5, §10.6
- Mandatory Integrity Mandate: Zero mock PASS counting as live inference.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Dict, List
import uuid

import pytest
import yaml

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    UnsupportedCapabilityError,
)
from factory_contracts.jobs.job_contract import JobState
from factory.apps.control_api.src.main import app
import benchmarks.runners.live_runner as lr
from benchmarks.runners.live_runner import (
    BenchmarkRunner,
    create_canonical_record,
    detect_cuda_environment,
)

# Align benchmark token with standard suite auth token to prevent session cross-pollution
STANDARD_TEST_TOKEN = "sk-factory-test-token-valid"
os.environ["FACTORY_API_KEY"] = STANDARD_TEST_TOKEN
lr.BENCHMARK_API_KEY = STANDARD_TEST_TOKEN

from services.music.acestep.adapter import ACEStepAdapter
from services.music.yue2.adapter import YuE2Adapter
from services.music.yue2.score import (
    diff_scores,
    extract_chords,
    extract_headers,
    generate_sample_abc,
    validate_abc_score,
)
from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.parameter_mappings import VideoParameterMapper
from starlette.testclient import TestClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_CASES_PATH = WORKSPACE_ROOT / "benchmarks" / "configs" / "evaluation_cases.yaml"
HW_PROFILES_PATH = WORKSPACE_ROOT / "benchmarks" / "configs" / "hardware_profiles.yaml"
MUSIC_CONFIG_PATH = WORKSPACE_ROOT / "benchmarks" / "configs" / "music_comparison.yaml"
CANONICAL_RESULTS_DIR = WORKSPACE_ROOT / "benchmarks" / "results" / "p5_matrix_live_04"


# =====================================================================
# 1. Config Integrity & Runner Fault Injection
# =====================================================================

class TestBenchmarkConfigAndRunnerFaultInjection:
    """Stress tests runner robustness when subjected to missing/malformed configs."""

    def test_runner_rejects_missing_cases_file(self, tmp_path: Path):
        """Runner must raise FileNotFoundError if evaluation_cases.yaml does not exist."""
        non_existent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(
                cases_config_path=non_existent,
                hardware_config_path=HW_PROFILES_PATH,
                music_config_path=MUSIC_CONFIG_PATH,
            )

    def test_runner_rejects_missing_hardware_profiles_file(self, tmp_path: Path):
        """Runner must raise FileNotFoundError if hardware_profiles.yaml is missing."""
        non_existent = tmp_path / "missing_hw.yaml"
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(
                cases_config_path=EVAL_CASES_PATH,
                hardware_config_path=non_existent,
                music_config_path=MUSIC_CONFIG_PATH,
            )

    def test_runner_rejects_missing_music_config_file(self, tmp_path: Path):
        """Runner must raise FileNotFoundError if music_comparison.yaml is missing."""
        non_existent = tmp_path / "missing_music.yaml"
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(
                cases_config_path=EVAL_CASES_PATH,
                hardware_config_path=HW_PROFILES_PATH,
                music_config_path=non_existent,
            )

    def test_runner_rejects_corrupted_yaml(self, tmp_path: Path):
        """Runner must fail cleanly on corrupted YAML syntax."""
        broken_yaml = tmp_path / "broken.yaml"
        broken_yaml.write_text("cases: [unclosed_list: {foo: bar", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            BenchmarkRunner(
                cases_config_path=broken_yaml,
                hardware_config_path=HW_PROFILES_PATH,
                music_config_path=MUSIC_CONFIG_PATH,
            )

    def test_runner_handles_empty_cases_list(self):
        """Runner must cleanly initialize and execute 0 cases without crash when cases is empty."""
        temp_dir = WORKSPACE_ROOT / "benchmarks" / "results" / "temp_adversarial_empty"
        empty_cases_file = temp_dir / "empty_cases.yaml"
        temp_dir.mkdir(parents=True, exist_ok=True)
        empty_cases_file.write_text("cases: []\n", encoding="utf-8")

        runner = BenchmarkRunner(
            cases_config_path=empty_cases_file,
            hardware_config_path=HW_PROFILES_PATH,
            music_config_path=MUSIC_CONFIG_PATH,
            output_base_dir=temp_dir,
            run_id="run_empty",
        )
        try:
            summary = asyncio.run(runner.run_all())
            assert summary["cases_evaluated"] == 0
            assert (runner.output_dir / "results.jsonl").exists()
            assert (runner.output_dir / "metrics.csv").exists()
            assert (runner.output_dir / "failures.json").exists()
        finally:
            runner.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_runner_handles_unmatched_case_filter(self):
        """Runner with a filter matching 0 cases executes gracefully and produces valid empty artifacts."""
        temp_dir = WORKSPACE_ROOT / "benchmarks" / "results" / "temp_adversarial_filter"
        temp_dir.mkdir(parents=True, exist_ok=True)

        runner = BenchmarkRunner(
            cases_config_path=EVAL_CASES_PATH,
            hardware_config_path=HW_PROFILES_PATH,
            music_config_path=MUSIC_CONFIG_PATH,
            output_base_dir=temp_dir,
            run_id="run_filter",
        )
        try:
            summary = asyncio.run(runner.run_all(case_filter="NON_EXISTENT_FILTER_GROUP_XYZ"))
            assert summary["cases_evaluated"] == 0
            with open(runner.output_dir / "results.jsonl") as f:
                lines = f.readlines()
            assert len(lines) == 0
        finally:
            runner.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_runner_handles_unknown_hardware_profile(self):
        """Specifying an unknown hardware profile falls back to empty profile without fabricating GPUs."""
        temp_dir = WORKSPACE_ROOT / "benchmarks" / "results" / "temp_adversarial_hw"
        temp_dir.mkdir(parents=True, exist_ok=True)

        runner = BenchmarkRunner(
            cases_config_path=EVAL_CASES_PATH,
            hardware_config_path=HW_PROFILES_PATH,
            music_config_path=MUSIC_CONFIG_PATH,
            hardware_profile_name="quantum-supercomputer-v99",
            output_base_dir=temp_dir,
            run_id="run_hw",
        )
        try:
            assert runner.hw_profile == {}
            if not detect_cuda_environment()["cuda_available"]:
                assert runner.is_cuda_available() is False
        finally:
            runner.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 2. Honest Hardware Blocker & Zero Mock PASS Mandate
# =====================================================================

class TestZeroMockAndHardwareBlockerIntegrity:
    """Verifies that no mock inferences pass as live certification and hardware blockers are honest."""

    @classmethod
    def setup_class(cls):
        cls.runner = BenchmarkRunner(
            cases_config_path=EVAL_CASES_PATH,
            hardware_config_path=HW_PROFILES_PATH,
            music_config_path=MUSIC_CONFIG_PATH,
        )

    @classmethod
    def teardown_class(cls):
        cls.runner.close()

    def test_evaluate_live_gpu_model_case_on_non_cuda_host(self):
        """On non-CUDA platform, all heavy model inference cases MUST return HARDWARE_BLOCKED."""
        if self.runner.is_cuda_available():
            pytest.skip("Test requires non-CUDA environment (Darwin/Apple Silicon)")

        heavy_cases = [
            {"id": "W01", "suite": "live_model", "model": "wan2.2", "variant": "14b", "task": "text_to_video", "hardware_requirement": "Kaggle Dual-T4", "description": "T2V"},
            {"id": "H01", "suite": "live_model", "model": "minimax-h3", "variant": "fp16", "task": "text_to_video_audio", "hardware_requirement": "Kaggle Dual-T4", "description": "T2VA"},
            {"id": "I01", "suite": "live_model", "model": "qwen-image-2.1", "variant": "turbo", "task": "text_to_image", "hardware_requirement": "Kaggle Dual-T4", "description": "T2I"},
            {"id": "YUE-01", "suite": "live_model", "model": "yue2", "variant": "q8_0", "task": "lyrics_to_music", "hardware_requirement": "NVIDIA BF16 >=24GB", "description": "Lyrics to music"},
            {"id": "ACE-01", "suite": "live_model", "model": "acestep-1.5", "variant": "turbo", "task": "text_to_music", "hardware_requirement": "Kaggle Dual-T4", "description": "Text to music"},
            {"id": "GPU-01", "suite": "live_model", "model": "qwen-image-2.1", "variant": "turbo", "task": "runtime_load", "hardware_requirement": "Kaggle Dual-T4", "description": "Cold load"},
        ]

        for case in heavy_cases:
            if case["id"].startswith("GPU-"):
                res = asyncio.run(self.runner.evaluate_gpu_runtime(case))
            else:
                res = asyncio.run(self.runner.evaluate_live_gpu_model_case(case))

            assert res["status"] == "HARDWARE_BLOCKED", f"Case {case['id']} must be HARDWARE_BLOCKED on non-CUDA host, got {res['status']}"
            assert res["integrity"]["mock_used"] is False, f"mock_used must be False in {case['id']}"
            assert res["integrity"]["fallback_used"] is False, f"fallback_used must be False in {case['id']}"
            assert res["artifact"]["validated"] is False
            assert res["error"] is not None
            assert res["error"]["code"] == "HARDWARE_BLOCKED"
            assert "remediation" in res["error"]

    def test_lora_training_transparently_not_implemented(self):
        """Case ACE-23 (LoRA training) must honestly report NOT_IMPLEMENTED."""
        case = {
            "id": "ACE-23",
            "group": "acestep",
            "suite": "live_model",
            "model": "acestep-1.5",
            "variant": "base",
            "task": "lora_train",
            "hardware_requirement": "Kaggle Dual-T4",
            "description": "LoRA fine-tuning training loop",
        }
        res = asyncio.run(self.runner.evaluate_music_symbolic_and_contract(case))
        assert res["status"] == "NOT_IMPLEMENTED"
        assert res["integrity"]["mock_used"] is False
        assert res["error"]["code"] == "NOT_IMPLEMENTED"

    def test_canonical_record_builder_integrity(self):
        """create_canonical_record enforces non-negotiable schema keys and mock_used default False."""
        rec = create_canonical_record(
            test_id="TEST-01",
            suite="test_suite",
            status="PASS",
            model="test-model",
            variant="test-var",
            task="test_task",
            git_commit="abcdef123456",
            gpu_names=[],
            parameters={"seed": 42},
            measurements={"queue_seconds": 0.01},
            artifact={"validated": True},
            integrity={"mock_used": False, "fallback_used": False},
        )
        assert rec["test_id"] == "TEST-01"
        assert rec["integrity"]["mock_used"] is False
        assert rec["integrity"]["fallback_used"] is False
        assert rec["environment"]["backend_version"] == "factory_v4_lite_1.0"
        assert "queue_seconds" in rec["measurements"]
        assert "rtf" in rec["measurements"]


# =====================================================================
# 3. Concurrency, Queue Leases & Lifecycle Isolation
# =====================================================================

class TestLeaseLifecycleAndConcurrencyIsolation:
    """Stress tests concurrency isolation, GPU device leasing, and state cleanup via live TestClient."""

    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app, base_url="http://testserver")
        cls.client.__enter__()
        cls.auth_headers = {"Authorization": f"Bearer {STANDARD_TEST_TOKEN}"}

    @classmethod
    def teardown_class(cls):
        try:
            cls.client.__exit__(None, None, None)
        except Exception:
            pass

    def test_single_job_per_gpu_lease_conflict_and_resolution(self):
        """Two workers cannot lease the same GPU device concurrently."""
        # 1. Submit job 1 (Image -> maps to GPU 0)
        sub1 = self.client.post(
            "/v1/jobs",
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Concurrency Job 1"}},
            headers=self.auth_headers,
        )
        assert sub1.status_code == 201
        job1_id = sub1.json()["job_id"]

        # 2. Submit job 2 (Image -> maps to GPU 0)
        sub2 = self.client.post(
            "/v1/jobs",
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Concurrency Job 2"}},
            headers=self.auth_headers,
        )
        assert sub2.status_code == 201
        job2_id = sub2.json()["job_id"]

        # 3. Worker A leases GPU 0 -> acquires job 1
        lease1 = self.client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-concurrency-a",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            },
            headers=self.auth_headers,
        )
        assert lease1.status_code == 200
        assert lease1.json().get("job") is not None
        leased1_id = lease1.json()["job"]["job_id"]
        assert leased1_id == job1_id

        # 4. Worker B attempts to lease GPU 0 while Worker A is busy -> must get 200 with job None
        lease2_blocked = self.client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-concurrency-b",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            },
            headers=self.auth_headers,
        )
        assert lease2_blocked.status_code == 200
        assert lease2_blocked.json().get("job") is None, "Worker B must NOT acquire a lease while GPU 0 is busy!"

        # 5. Worker A completes job 1 -> frees GPU 0
        comp1 = self.client.post(
            f"/v1/jobs/{job1_id}/complete",
            json={"worker_id": "worker-concurrency-a", "result": {"output": "job1_done"}},
            headers=self.auth_headers,
        )
        assert comp1.status_code == 200

        # 6. Worker B retries leasing GPU 0 -> successfully acquires job 2
        lease2_acquired = self.client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-concurrency-b",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            },
            headers=self.auth_headers,
        )
        assert lease2_acquired.status_code == 200
        assert lease2_acquired.json().get("job") is not None
        assert lease2_acquired.json()["job"]["job_id"] == job2_id

        # 7. Clean up job 2
        comp2 = self.client.post(
            f"/v1/jobs/{job2_id}/complete",
            json={"worker_id": "worker-concurrency-b", "result": {"output": "job2_done"}},
            headers=self.auth_headers,
        )
        assert comp2.status_code == 200

    def test_cancellation_of_running_job_releases_lease(self):
        """Cancelling a leased job immediately releases the GPU device lease for future workers."""
        # Submit job
        sub = self.client.post(
            "/v1/jobs",
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Job to cancel"}},
            headers=self.auth_headers,
        )
        assert sub.status_code == 201
        jid = sub.json()["job_id"]

        # Worker leases GPU 0
        lease = self.client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-cancel-test",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            },
            headers=self.auth_headers,
        )
        assert lease.status_code == 200
        assert lease.json().get("job") is not None

        # Cancel the leased job
        del_res = self.client.delete(f"/v1/jobs/{jid}", headers=self.auth_headers)
        assert del_res.status_code == 200
        assert del_res.json().get("state") == "CANCELLED"

        # Now GPU 0 must be idle: submitting and leasing a new job must succeed immediately
        sub2 = self.client.post(
            "/v1/jobs",
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Job after cancel"}},
            headers=self.auth_headers,
        )
        new_jid = sub2.json()["job_id"]

        lease2 = self.client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-after-cancel",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            },
            headers=self.auth_headers,
        )
        assert lease2.status_code == 200
        assert lease2.json().get("job") is not None
        assert lease2.json()["job"]["job_id"] == new_jid

        # Clean up
        self.client.post(
            f"/v1/jobs/{new_jid}/complete",
            json={"worker_id": "worker-after-cancel", "result": {"done": True}},
            headers=self.auth_headers,
        )

    def test_parameter_mapper_immutability(self):
        """Parameter mappers must not mutate caller input dictionaries."""
        dummy_workflow = {
            "3": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": 1.0, "seed": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
        }
        original_inputs = {
            "variant": "turbo",
            "prompt": "Test prompt",
            "steps": 8,
            "cfg": 1.0,
            "nested_param": {"flag": True},
        }
        cloned_inputs = copy.deepcopy(original_inputs)

        mapped = ImageParameterMapper.apply(
            workflow=copy.deepcopy(dummy_workflow),
            task="text_to_image",
            inputs=original_inputs,
        )
        assert mapped is not None
        assert original_inputs == cloned_inputs, "ImageParameterMapper must not mutate caller inputs dictionary!"


# =====================================================================
# 4. Symbolic Music and Parameter Validation Robustness
# =====================================================================

class TestSymbolicMusicAndAdapterRobustness:
    """Stress tests ABC score validation, diffing, and music adapters against adversarial payloads."""

    def test_malformed_abc_missing_headers_rejected(self):
        """validate_abc_score rejects scores without required headers."""
        bad_scores = [
            "",  # empty
            "   \n\n\t",  # whitespace
            "|: C D E F :|",  # missing Key header K:
            "X: 1\n|: C D E F :|",  # missing Key header K:
            "T: Song Title\n|: C D E F :|",  # missing Key header K:
            "X: 1\nK: InvalidKey123\n|: C D :|",  # invalid Key format
            "X: 1\nK: C\nM: 99/99\n|: C D :|",  # invalid meter format
            "X: 1\nK: C\nQ: -10\n|: C D :|",  # invalid tempo
        ]
        for s in bad_scores:
            with pytest.raises(InvalidRequestError):
                validate_abc_score(s)

    def test_malformed_abc_adversarial_unbalanced_delimiters(self):
        """validate_abc_score safely rejects unbalanced quotes, brackets, and braces."""
        adversarial_payloads = [
            'X: 1\nK: C\n|: "unclosed quote c d :|',
            "X: 1\nK: C\n|: [unclosed bracket c d :|",
            "X: 1\nK: C\n|: {unclosed brace c d :|",
            "X: 1\nK: C\n|: (unclosed paren c d :|",
            "X: 1\nK: C\n|: unmatched closing ] :|",
            "X: 1\nK: C\n|: unmatched closing } :|",
        ]
        for payload in adversarial_payloads:
            with pytest.raises(InvalidRequestError):
                validate_abc_score(payload)

    def test_diff_scores_identical_and_divergent(self):
        """diff_scores produces expected diffs without crashing."""
        sample = generate_sample_abc()
        # Identical scores: chord_diff has empty added/removed
        diff_same = diff_scores(sample, sample)
        assert len(diff_same.get("chord_diff", {}).get("added", [])) == 0
        assert len(diff_same.get("chord_diff", {}).get("removed", [])) == 0

        # Divergent scores: line diff detects changes
        divergent = sample.replace('"C"', '"Dm"')
        diff_div = diff_scores(sample, divergent)
        assert len(diff_div.get("line_diff", [])) > 0

    def test_acestep_adapter_out_of_bound_bpm_rejection(self):
        """ACEStepAdapter rejects invalid BPM values."""
        for invalid_bpm in [-10, 0, 10, 350, 9999]:
            with pytest.raises(InvalidRequestError):
                ACEStepAdapter.map_inputs(
                    task="text_to_music",
                    inputs={"prompt": "Jazz", "bpm": invalid_bpm, "variant": "turbo"},
                )

    def test_acestep_adapter_turbo_rejects_base_only_tasks(self):
        """Turbo variant cannot execute extract, lego, or complete tasks."""
        base_only_tasks = ["extract", "lego", "complete"]
        for task in base_only_tasks:
            with pytest.raises(UnsupportedCapabilityError):
                ACEStepAdapter.map_inputs(
                    task=task,
                    inputs={"variant": "turbo", "source_audio_asset_id": "asset-xyz"},
                )


# =====================================================================
# 5. Full 163-Case Matrix Completeness & Live Artifact Verification
# =====================================================================

class TestMatrixCompletenessAndCanonicalLiveArtifacts:
    """Rigorous audit of evaluation_cases.yaml and benchmarks/results/p5_matrix_live_04/."""

    def test_evaluation_cases_yaml_exact_163_unique_cases(self):
        """Assert exactly 163 unique cases across all 15 operational groups."""
        with open(EVAL_CASES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cases: List[Dict[str, Any]] = data.get("cases", [])
        assert len(cases) == 163, f"Expected exactly 163 cases, found {len(cases)}"

        ids = [c["id"] for c in cases]
        assert len(set(ids)) == 163, f"Duplicate case IDs found: {[cid for cid in ids if ids.count(cid) > 1]}"

        expected_group_counts = {
            "router_infrastructure": 28,  # R01-R28
            "gpu_runtime": 10,           # GPU-01-GPU-10
            "text": 5,                   # T01-T05
            "vlm": 5,                    # V01-V05
            "decision": 3,               # D01-D03
            "embedding": 4,              # E01-E04
            "speech": 11,                # S01-S11
            "image": 15,                 # I01-I15
            "wan_video": 8,              # W01-W08
            "minimax_video": 10,         # H01-H10
            "video_integration": 2,      # X01-X02
            "acestep": 24,               # ACE-01-ACE-24 (with ACE-09b)
            "yue2": 20,                  # YUE-01-YUE-20
            "enhancement": 6,            # EN01-EN06
            "cross_feature": 12,         # P01-P12
        }

        actual_group_counts: Dict[str, int] = {}
        for c in cases:
            grp = c.get("group", "unknown")
            actual_group_counts[grp] = actual_group_counts.get(grp, 0) + 1

        assert actual_group_counts == expected_group_counts, (
            f"Group counts mismatch: actual {actual_group_counts} != expected {expected_group_counts}"
        )

    def test_canonical_run_results_jsonl_integrity(self):
        """Audits benchmarks/results/p5_matrix_live_04/results.jsonl against canonical schema and zero mock rule."""
        results_path = CANONICAL_RESULTS_DIR / "results.jsonl"
        assert results_path.exists(), f"results.jsonl not found at {results_path}"

        records: List[Dict[str, Any]] = []
        with open(results_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        assert len(records) == 163, f"Expected 163 records in results.jsonl, got {len(records)}"

        required_canonical_keys = {
            "test_id", "suite", "status", "model", "variant",
            "task", "environment", "parameters", "measurements",
            "artifact", "integrity", "error",
        }
        env_subkeys = {"git_commit", "model_revision", "backend_version", "gpu_names", "gpu_ids"}
        meas_subkeys = {"queue_seconds", "load_seconds", "inference_seconds", "end_to_end_seconds", "peak_vram_mb_per_gpu", "actual_duration_seconds", "rtf"}
        integrity_subkeys = {"mock_used", "fallback_used", "weights_verified"}

        status_counts = {"PASS": 0, "HARDWARE_BLOCKED": 0, "NOT_IMPLEMENTED": 0, "FAIL": 0, "ERROR": 0}

        for r in records:
            tid = r.get("test_id")
            # 1. Required top-level keys
            missing_top = required_canonical_keys - set(r.keys())
            assert not missing_top, f"Record {tid} is missing keys: {missing_top}"

            # 2. Required nested keys
            missing_env = env_subkeys - set(r["environment"].keys())
            assert not missing_env, f"Record {tid} missing environment keys: {missing_env}"

            missing_meas = meas_subkeys - set(r["measurements"].keys())
            assert not missing_meas, f"Record {tid} missing measurements keys: {missing_meas}"

            missing_integ = integrity_subkeys - set(r["integrity"].keys())
            assert not missing_integ, f"Record {tid} missing integrity keys: {missing_integ}"

            # 3. Mandatory Integrity Mandate: mock_used must be strictly False
            assert r["integrity"]["mock_used"] is False, f"MOCK DETECTED in {tid}!"
            assert r["integrity"]["fallback_used"] is False, f"FALLBACK DETECTED in {tid}!"

            st = r["status"]
            status_counts[st] = status_counts.get(st, 0) + 1

        # 4. Strict status distribution check per dispatch contract
        assert status_counts["PASS"] == 51, f"Expected 51 PASS, got {status_counts['PASS']}"
        assert status_counts["HARDWARE_BLOCKED"] == 111, f"Expected 111 HARDWARE_BLOCKED, got {status_counts['HARDWARE_BLOCKED']}"
        assert status_counts["NOT_IMPLEMENTED"] == 1, f"Expected 1 NOT_IMPLEMENTED, got {status_counts['NOT_IMPLEMENTED']}"
        assert status_counts["FAIL"] == 0, f"Expected 0 FAIL, got {status_counts['FAIL']}"
        assert status_counts["ERROR"] == 0, f"Expected 0 ERROR, got {status_counts['ERROR']}"

    def test_canonical_run_failures_json_and_metrics_csv(self):
        """Audits metadata.json, failures.json, and metrics.csv for exact consistency."""
        meta_path = CANONICAL_RESULTS_DIR / "metadata.json"
        failures_path = CANONICAL_RESULTS_DIR / "failures.json"
        metrics_path = CANONICAL_RESULTS_DIR / "metrics.csv"

        assert meta_path.exists()
        assert failures_path.exists()
        assert metrics_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["cases_evaluated"] == 163
        assert meta["total_planned_cases"] == 163
        assert meta["status_summary"]["PASS"] == 51
        assert meta["status_summary"]["HARDWARE_BLOCKED"] == 111
        assert meta["status_summary"]["NOT_IMPLEMENTED"] == 1

        with open(failures_path, "r", encoding="utf-8") as f:
            failures = json.load(f)
        assert len(failures) == 112
        for fl in failures:
            assert fl["status"] in ("HARDWARE_BLOCKED", "NOT_IMPLEMENTED")
            assert fl["error"] is not None
            assert "code" in fl["error"]

        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 163
        for row in rows:
            assert row["mock_used"] == "False"
