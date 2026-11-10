"""
benchmarks.runner
Unified Multi-Modal Benchmark Execution Engine for Factory v4 Lite (Plan §4.4, §6.1-§6.6).

Features:
1. Supports 'smoke', 'full', and 'perf' benchmark execution modes.
2. Discrete Dual-T4 memory bounds enforcement (GPU 0 vs GPU 1, 14.2 GB max allocatable per card).
3. Sampled NVML GPU VRAM time series over the inference window.
4. Strict status semantics (Plan §6.5): PASS, FAIL, ERROR, HARDWARE_BLOCKED, BACKEND_UNAVAILABLE, NOT_IMPLEMENTED.
5. MANDATORY INTEGRITY MANDATE: Zero simulation-as-certification. Genuine execution or honest non-pass status.
6. Unified reporting into results.json, metrics.csv, and summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from starlette.testclient import TestClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Setup environment for in-memory Factory Lite operation
BENCHMARK_API_KEY = os.environ.get("FACTORY_API_KEY", "sk-factory-benchmark-key-2026")
os.environ["FACTORY_API_KEY"] = BENCHMARK_API_KEY
os.environ["ENVIRONMENT"] = "test"
os.environ["FACTORY_ENV"] = "test"
os.environ["STORAGE_BACKEND"] = "ephemeral_disk"
os.environ.setdefault("SCRATCH_DIR", "/tmp/factory_lite/artifacts")
SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
os.environ["RATE_LIMIT_ENABLED"] = "false"

from factory.apps.control_api.src.main import app
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP, get_subsystem_gpu
from factory.apps.control_api.src.queue.repository import get_job_repository
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.watchdog import (
    MODEL_EXECUTION_TIMEOUTS,
    SERVICE_EXECUTION_TIMEOUTS,
    TASK_EXECUTION_TIMEOUTS,
)
from factory_contracts.errors import (
    ErrorCode,
    FactoryError,
    InsufficientVRAMError,
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
)
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobState,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import JobCompletionRequest, WorkerLeaseRequest
from model_registry.loader import ManifestLoader

from benchmarks.report import BenchmarkReportGenerator, CaseExecutionRecord
from benchmarks.validators import ValidationResult, validate_artifact

logger = logging.getLogger("benchmark_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Dual-T4 Limits
TOTAL_VRAM_PER_GPU_MB: float = 15360.0  # 15.0 GB physical ceiling
MAX_ALLOCATABLE_VRAM_PER_GPU_MB: float = 14540.0  # 14.2 GB allocatable ceiling per card


def get_current_git_commit() -> str:
    """Retrieves current git commit hash with authentic provenance."""
    for env_key in ("GIT_COMMIT", "COMMIT_SHA", "CI_COMMIT_SHA"):
        val = os.getenv(env_key)
        if val and len(val.strip()) >= 7:
            return val.strip()

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass

    return "UNKNOWN_UNCOMMITTED"


@dataclass
class NVMLSample:
    timestamp_ms: float
    gpu_index: int
    used_vram_mb: float
    total_vram_mb: float
    utilization_percent: float


class NVMLSampler:
    """Samples GPU memory and utilization across discrete accelerators during inference."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.samples: List[NVMLSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_initialized = False
        self._gpu_count = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_initialized = self._gpu_count > 0
        except Exception:
            self._nvml_initialized = False

    def is_available(self) -> bool:
        return self._nvml_initialized

    def _sample_loop(self):
        import pynvml
        while self._running:
            t_ms = time.time() * 1000.0
            for i in range(self._gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.samples.append(
                        NVMLSample(
                            timestamp_ms=t_ms,
                            gpu_index=i,
                            used_vram_mb=round(mem.used / (1024 * 1024), 1),
                            total_vram_mb=round(mem.total / (1024 * 1024), 1),
                            utilization_percent=float(util.gpu),
                        )
                    )
                except Exception:
                    pass
            time.sleep(self.interval_seconds)

    def start(self):
        if not self._nvml_initialized:
            return
        self.samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[NVMLSample]:
        if not self._running:
            return []
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return list(self.samples)

    def get_peak_vram_mb(self, gpu_index: int) -> Optional[float]:
        relevant = [s.used_vram_mb for s in self.samples if s.gpu_index == gpu_index]
        return max(relevant) if relevant else None


class DualT4MemoryGuard:
    """Enforces discrete per-GPU memory limits (14.2 GB max allocatable per GPU)."""

    def check_bounds(
        self,
        model_id: str,
        min_vram_gb: float,
        target_gpu: int,
        allocation_policy: str = "single_gpu",
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that required VRAM does not exceed the discrete 14.2 GB limit
        for the assigned accelerator partition.
        """
        required_mb = min_vram_gb * 1024.0

        if allocation_policy == "dual_gpu_split":
            # Allowed to use up to 14.2 GB on each card
            if required_mb > (MAX_ALLOCATABLE_VRAM_PER_GPU_MB * 2):
                return False, f"Model requires {min_vram_gb:.1f} GB VRAM, exceeding total Dual-T4 split capacity (28.4 GB)."
            return True, None

        if required_mb > MAX_ALLOCATABLE_VRAM_PER_GPU_MB:
            return (
                False,
                f"Model '{model_id}' requires {min_vram_gb:.1f} GB VRAM, which exceeds the discrete "
                f"14.2 GB ({MAX_ALLOCATABLE_VRAM_PER_GPU_MB:.1f} MB) limit on GPU {target_gpu}. "
                f"Dual-T4 hardware does not provide a 32 GB shared virtual memory pool.",
            )

        return True, None


class UnifiedBenchmarkRunner:
    """Unified Benchmark Execution Engine for Factory v4 Lite."""

    def __init__(
        self,
        cases_file: Optional[Path] = None,
        manifests_dir: Optional[Path] = None,
        output_base_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.cases_file = cases_file or WORKSPACE_ROOT / "benchmarks" / "cases.yaml"
        self.manifests_dir = manifests_dir or WORKSPACE_ROOT / "manifests"
        self.run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = (output_base_dir or WORKSPACE_ROOT / "benchmarks" / "results") / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.git_commit = get_current_git_commit()
        self.sampler = NVMLSampler()
        self.memory_guard = DualT4MemoryGuard()

        # Load cases
        with open(self.cases_file, "r", encoding="utf-8") as f:
            cases_doc = yaml.safe_load(f)
            self.cases: List[Dict[str, Any]] = cases_doc.get("cases", [])

        # Load manifests
        self.manifest_loader = ManifestLoader(self.manifests_dir)
        self.models_manifest = self.manifest_loader.load_models()

        # Client targeting real server or in-memory FastAPI app
        self.base_url = base_url or os.environ.get("FACTORY_BASE_URL")
        if self.base_url:
            import httpx
            self.client = httpx.Client(base_url=self.base_url, timeout=60.0)
            self._is_test_client = False
        else:
            self.client = TestClient(app, base_url="http://testserver")
            self.client.__enter__()
            self._is_test_client = True

        # Hardware probing
        self.hardware_info = self._detect_hardware()

    @property
    def auth_headers(self) -> Dict[str, str]:
        """Resolves current FACTORY_API_KEY dynamically."""
        api_key = os.environ.get("FACTORY_API_KEY") or BENCHMARK_API_KEY
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def close(self):
        """Closes client and cleans up lifespan background tasks."""
        try:
            if getattr(self, "_is_test_client", False):
                self.client.__exit__(None, None, None)
            else:
                self.client.close()
        except Exception:
            pass



    def _detect_hardware(self) -> Dict[str, Any]:
        """Probes hardware environment."""
        cuda_avail = False
        device_count = 0
        gpu_names: List[str] = []

        try:
            import torch
            if torch.cuda.is_available():
                cuda_avail = True
                device_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        except Exception:
            pass

        return {
            "cuda_available": cuda_avail,
            "gpu_count": device_count,
            "gpu_names": gpu_names,
            "nvml_available": self.sampler.is_available(),
            "max_allocatable_vram_mb": MAX_ALLOCATABLE_VRAM_PER_GPU_MB,
        }

    def resolve_timeouts(self, case: Dict[str, Any], subsystem: str) -> Dict[str, float]:
        """
        Resolves 3-tier state-aware timeouts: queue_timeout, load_timeout, inference_timeout.
        Honors task, model, and service hierarchy from watchdog.py without rigid 180s imposition.
        """
        is_test = getattr(self, "_is_test_client", False)
        model_id = case.get("model", "")
        task = case.get("task", "")

        # 1. Queue timeout: allowed time in CREATED / QUEUED before worker lease
        default_queue = 0.1 if is_test else 60.0
        queue_timeout = float(os.environ.get("BENCHMARK_QUEUE_TIMEOUT", default_queue))

        # 2. Load timeout: allowed time in ADMITTED or loading stage
        default_load = 0.5 if is_test else 120.0
        load_timeout = float(os.environ.get("BENCHMARK_LOAD_TIMEOUT", default_load))

        # 3. Inference timeout: priority resolution
        explicit_case_timeout = case.get("timeout_seconds") or case.get("inference_timeout")
        if explicit_case_timeout:
            inf_timeout = float(explicit_case_timeout)
        elif os.environ.get("BENCHMARK_INFERENCE_TIMEOUT"):
            inf_timeout = float(os.environ["BENCHMARK_INFERENCE_TIMEOUT"])
        elif task in TASK_EXECUTION_TIMEOUTS:
            inf_timeout = float(TASK_EXECUTION_TIMEOUTS[task])
        elif model_id in MODEL_EXECUTION_TIMEOUTS:
            inf_timeout = float(MODEL_EXECUTION_TIMEOUTS[model_id])
        elif subsystem in SERVICE_EXECUTION_TIMEOUTS:
            inf_timeout = float(SERVICE_EXECUTION_TIMEOUTS[subsystem])
        else:
            inf_timeout = 300.0

        return {
            "queue_timeout": queue_timeout,
            "load_timeout": load_timeout,
            "inference_timeout": inf_timeout,
        }

    def _extract_checkpoint_provenance(
        self, model_id: str, job_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extracts checkpoint repository, revision, and commit hash from worker telemetry or manifest."""
        chk_repo = None
        chk_rev = None
        chk_hash = None

        # 1. Check worker or job telemetry first
        if isinstance(job_data, dict):
            worker = job_data.get("worker", {})
            if isinstance(worker, dict):
                chk = worker.get("checkpoint", {})
                if isinstance(chk, dict):
                    chk_repo = chk.get("repo")
                    chk_rev = chk.get("revision")
                    chk_hash = chk.get("hash")
            if not chk_repo:
                telem = job_data.get("telemetry", {})
                if isinstance(telem, dict):
                    chk = telem.get("checkpoint", {})
                    if isinstance(chk, dict):
                        chk_repo = chk.get("repo")
                        chk_rev = chk.get("revision")
                        chk_hash = chk.get("hash")

        # 2. Fallback to model manifest declaration
        if not chk_repo:
            manifest = self.models_manifest.get(model_id)
            if manifest:
                if hasattr(manifest, "weights"):
                    chk_repo = getattr(manifest.weights, "repository", None) or getattr(manifest.weights, "primary_checkpoint", None)
                    chk_rev = getattr(manifest.weights, "checkpoint_revision", None) or getattr(manifest, "revision", None)
                    chk_hash = getattr(manifest.weights, "sha256", None)
                if not chk_rev and hasattr(manifest, "revision") and manifest.revision:
                    chk_rev = str(manifest.revision)
                if hasattr(manifest, "model") and hasattr(manifest.model, "checkpoints"):
                    chks = manifest.model.checkpoints
                    if chks and len(chks) > 0:
                        c0 = chks[0]
                        chk_repo = getattr(c0, "repo", None) or (c0.get("repo") if isinstance(c0, dict) else None) or chk_repo
                        chk_rev = getattr(c0, "revision", None) or (c0.get("revision") if isinstance(c0, dict) else None) or chk_rev
                        chk_hash = getattr(c0, "hash", None) or (c0.get("hash") if isinstance(c0, dict) else None) or chk_hash

        return chk_repo, chk_rev, chk_hash

    def evaluate_case(self, case: Dict[str, Any], mode: str) -> CaseExecutionRecord:
        """
        Executes and evaluates a single benchmark case according to strict status semantics:
        PASS, FAIL, ERROR, HARDWARE_BLOCKED, BACKEND_UNAVAILABLE, NOT_IMPLEMENTED.
        """
        cid = case["id"]
        model_id = case["model"]
        task = case["task"]
        variant = case.get("variant", "standard")
        target_gpu = case.get("target_gpu", 0)
        min_vram_gb = case.get("min_vram_gb", 0.0)
        is_negative = case.get("is_negative", False)
        inputs = case.get("inputs", {})
        expected_status = case.get("expected_status", "PASS")
        expected_output = case.get("expected_output", {})

        # 1. Gate 1: Hardware Memory & Partition Feasibility
        manifest = self.models_manifest.get(model_id)
        alloc_policy = "single_gpu"
        if manifest and hasattr(manifest, "hardware"):
            alloc_policy = getattr(manifest.hardware, "allocation_policy", "single_gpu")

        ok_mem, mem_reason = self.memory_guard.check_bounds(
            model_id=model_id,
            min_vram_gb=min_vram_gb,
            target_gpu=target_gpu,
            allocation_policy=alloc_policy,
        )
        if not ok_mem:
            logger.info(f"[{cid}] HARDWARE_BLOCKED: {mem_reason}")
            return CaseExecutionRecord(
                case_id=cid,
                run_id=self.run_id,
                git_commit=self.git_commit,
                model_id=model_id,
                task=task,
                variant=variant,
                mode=mode,
                status="HARDWARE_BLOCKED",
                target_gpu=target_gpu,
                min_vram_gb=min_vram_gb,
                error_code="INSUFFICIENT_VRAM",
                error_message=mem_reason,
                is_negative=is_negative,
                contract_pass=False if is_negative else None,
                inference_pass=False,
            )

        # 2. Gate 2: Manifest Capability & Backend Availability Check
        if not is_negative:
            if manifest:
                installed = getattr(manifest, "installed", True)
                backend_ready = getattr(manifest, "backend_ready", True)
                if not installed or not backend_ready:
                    logger.info(f"[{cid}] BACKEND_UNAVAILABLE: model '{model_id}' installed={installed}, backend_ready={backend_ready}")
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="BACKEND_UNAVAILABLE",
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        error_code="BACKEND_UNAVAILABLE",
                        error_message=f"Model engine or weights not ready/installed for '{model_id}' per manifest declaration.",
                        is_negative=is_negative,
                        contract_pass=None,
                        inference_pass=False,
                    )

                # Check capabilities truth
                caps_truth = getattr(manifest, "capabilities_truth", {})
                if task in caps_truth and not caps_truth[task].implemented:
                    logger.info(f"[{cid}] NOT_IMPLEMENTED: task '{task}' not implemented for '{model_id}'")
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="NOT_IMPLEMENTED",
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        error_code="CAPABILITY_NOT_SUPPORTED",
                        error_message=f"Task '{task}' is declared unbacked/unimplemented in model manifest.",
                        is_negative=is_negative,
                        contract_pass=None,
                        inference_pass=False,
                    )

        # 3. Gate 3: Negative / Fail-Closed Rejection Test Execution
        if is_negative:
            t0 = time.perf_counter()
            subsystem = model_id.split("-")[0]
            if "image" in model_id:
                subsystem = "image"
            elif "video" in model_id or "wan" in model_id or "minimax" in model_id:
                subsystem = "video"
            elif "music" in model_id or "ace" in model_id or "yue" in model_id:
                subsystem = "music"
            elif "speech" in model_id:
                subsystem = "speech"
            elif "decision" in model_id:
                subsystem = "decision"
            elif "embed" in model_id:
                subsystem = "embedding"
            elif "text" in model_id:
                subsystem = "text"
            elif "vlm" in model_id:
                subsystem = "vision"

            inputs_payload = dict(inputs) if isinstance(inputs, dict) else {}
            if variant and variant != "standard":
                inputs_payload["variant"] = variant

            payload = {
                "service": subsystem,
                "model": model_id,
                "task": task,
                "inputs": inputs_payload,
            }

            resp = self.client.post("/v1/jobs", json=payload, headers=self.auth_headers)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Must reject with non-success HTTP status (400, 422, or 413)
            if resp.status_code in (400, 422, 413):
                expected_error_code = case.get("expected_error_code")
                actual_error_code = None
                try:
                    resp_json = resp.json()
                    if isinstance(resp_json, dict):
                        err_obj = resp_json.get("error", {})
                        if isinstance(err_obj, dict):
                            actual_error_code = err_obj.get("code")
                            if not actual_error_code:
                                details = err_obj.get("details", {})
                                if isinstance(details, dict):
                                    actual_error_code = details.get("error_code")
                except Exception:
                    pass

                # If expected error code specified, enforce exact or semantic equivalence
                if expected_error_code:
                    code_matches = False
                    if actual_error_code == expected_error_code:
                        code_matches = True
                    elif expected_error_code == "INVALID_INPUT" and actual_error_code in ("INVALID_INPUT", "INVALID_REQUEST", "VALIDATION_ERROR"):
                        code_matches = True
                    elif expected_error_code == "CAPABILITY_NOT_SUPPORTED" and actual_error_code in ("CAPABILITY_NOT_SUPPORTED", "UNSUPPORTED_CAPABILITY"):
                        code_matches = True

                    if not code_matches:
                        logger.warning(
                            f"[{cid}] FAIL: negative case rejected with code '{actual_error_code}', expected '{expected_error_code}'"
                        )
                        return CaseExecutionRecord(
                            case_id=cid,
                            run_id=self.run_id,
                            git_commit=self.git_commit,
                            model_id=model_id,
                            task=task,
                            variant=variant,
                            mode=mode,
                            status="FAIL",
                            contract_pass=False,
                            inference_pass=False,
                            target_gpu=target_gpu,
                            min_vram_gb=min_vram_gb,
                            e2e_latency_ms=round(elapsed_ms, 2),
                            error_code="ERROR_CODE_MISMATCH",
                            error_message=f"Expected error code '{expected_error_code}', got '{actual_error_code}' (HTTP {resp.status_code})",
                            is_negative=True,
                        )

                logger.info(f"[{cid}] PASS: cleanly rejected with HTTP {resp.status_code} (code: {actual_error_code})")
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="PASS",
                    contract_pass=True,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=round(elapsed_ms, 2),
                    is_negative=True,
                    evidence_logs=f"HTTP {resp.status_code} [{actual_error_code}]: {resp.text[:200]}",
                )
            else:
                logger.warning(f"[{cid}] FAIL: negative case unexpectedly accepted with HTTP {resp.status_code}")
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=False,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=round(elapsed_ms, 2),
                    error_code="NEGATIVE_CHECK_FAILED",
                    error_message=f"Expected HTTP 400 rejection for negative case, got HTTP {resp.status_code}",
                    is_negative=True,
                )

        # 4. Gate 4: Positive Execution Path
        # Determine subsystem
        subsystem = model_id.split("-")[0]
        if "image" in model_id:
            subsystem = "image"
        elif "video" in model_id or "wan" in model_id or "minimax" in model_id:
            subsystem = "video"
        elif "music" in model_id or "ace" in model_id or "yue" in model_id:
            subsystem = "music"
        elif "speech" in model_id:
            subsystem = "speech"
        elif "decision" in model_id:
            subsystem = "decision"
        elif "embed" in model_id:
            subsystem = "embedding"
        elif "text" in model_id:
            subsystem = "text"
        elif "vlm" in model_id:
            subsystem = "vision"

        inputs_payload = dict(inputs) if isinstance(inputs, dict) else {}
        if variant and variant != "standard":
            inputs_payload["variant"] = variant

        payload = {
            "service": subsystem,
            "model": model_id,
            "task": task,
            "inputs": inputs_payload,
        }


        # Start NVML sampling over inference window
        self.sampler.start()
        t_start = time.perf_counter()

        try:
            # Submit job to FastAPI Control API
            submit_resp = self.client.post("/v1/jobs", json=payload, headers=self.auth_headers)

            if submit_resp.status_code not in (200, 201):
                raw_err = submit_resp.json() if submit_resp.headers.get("content-type", "").startswith("application/json") else submit_resp.text
                err_code = raw_err.get("error", {}).get("code") if isinstance(raw_err, dict) else "SUBMISSION_REJECTED"
                err_msg = raw_err.get("error", {}).get("message") if isinstance(raw_err, dict) else str(raw_err)

                # Check if backend unavailable
                if submit_resp.status_code == 503 or "BACKEND_UNAVAILABLE" in str(err_code) or "UNAVAILABLE" in str(err_code):
                    status = "BACKEND_UNAVAILABLE"
                else:
                    status = "FAIL"

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                samples = self.sampler.stop()

                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status=status,
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=round(elapsed_ms, 2),
                    error_code=err_code,
                    error_message=err_msg,
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

            # Job submitted successfully
            job_data = submit_resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                samples = self.sampler.stop()
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=round(elapsed_ms, 2),
                    error_code="INVALID_RESPONSE",
                    error_message="Submission succeeded but no job_id returned in response.",
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

            # Polling to terminal state using 3-tier timeout hierarchy
            timeouts = self.resolve_timeouts(case, subsystem)
            queue_timeout = timeouts["queue_timeout"]
            load_timeout = timeouts["load_timeout"]
            inference_timeout = timeouts["inference_timeout"]

            terminal_states = {"SUCCEEDED", "FAILED", "CANCELLED"}
            current_state = str(job_data.get("state") or job_data.get("status") or "").upper()

            is_test_client = getattr(self, "_is_test_client", False)
            poll_interval = 0.01 if is_test_client else 0.5
            poll_start = time.perf_counter()
            state_enter_time = poll_start

            while current_state not in terminal_states:
                now = time.perf_counter()
                time_in_state = now - state_enter_time
                total_poll_time = now - poll_start

                # 1. Queue Timeout: Time allowed in CREATED or QUEUED waiting for worker lease
                if current_state in ("CREATED", "QUEUED") and time_in_state >= queue_timeout:
                    samples = self.sampler.stop()
                    elapsed_ms = (now - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="BACKEND_UNAVAILABLE",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code="QUEUE_TIMEOUT",
                        error_message=f"Job '{job_id}' not leased by worker after {queue_timeout:.1f}s in state '{current_state}'. Backend inference daemon for '{model_id}' is not resident on GPU {target_gpu}.",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )

                # 2. Load Timeout: Time allowed in ADMITTED loading weights
                if current_state == "ADMITTED" and time_in_state >= load_timeout:
                    samples = self.sampler.stop()
                    elapsed_ms = (now - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="FAIL",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code="LOAD_TIMEOUT",
                        error_message=f"Job '{job_id}' exceeded load timeout ({load_timeout:.1f}s) in state '{current_state}'.",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )

                # 3. Retrying Timeout: Time allowed in RETRYING state
                if current_state == "RETRYING" and time_in_state >= queue_timeout:
                    samples = self.sampler.stop()
                    elapsed_ms = (now - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="FAIL",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code="RETRY_EXHAUSTED",
                        error_message=f"Job '{job_id}' remained in RETRYING state for {queue_timeout:.1f}s.",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )

                # 4. Inference Timeout: Time allowed in RUNNING or STREAMING, or overall poll
                if current_state in ("RUNNING", "STREAMING") and (time_in_state >= inference_timeout or total_poll_time >= inference_timeout):
                    samples = self.sampler.stop()
                    elapsed_ms = (now - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="FAIL",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code="INFERENCE_TIMEOUT",
                        error_message=f"Job '{job_id}' timed out after {inference_timeout:.1f}s (last state: '{current_state}').",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )

                time.sleep(poll_interval)
                try:
                    poll_resp = self.client.get(f"/v1/jobs/{job_id}", headers=self.auth_headers)
                except Exception as poll_exc:
                    logger.warning(f"[{cid}] Exception during job poll: {poll_exc}")
                    continue

                if poll_resp.status_code == 200:
                    job_data = poll_resp.json()
                    new_state = str(job_data.get("state") or job_data.get("status") or "").upper()
                    if new_state != current_state:
                        current_state = new_state
                        state_enter_time = time.perf_counter()
                elif poll_resp.status_code == 404:
                    samples = self.sampler.stop()
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="FAIL",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code="JOB_NOT_FOUND",
                        error_message=f"Job '{job_id}' was not found during polling (HTTP 404).",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )
                elif poll_resp.status_code >= 500:
                    samples = self.sampler.stop()
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    return CaseExecutionRecord(
                        case_id=cid,
                        run_id=self.run_id,
                        git_commit=self.git_commit,
                        model_id=model_id,
                        task=task,
                        variant=variant,
                        mode=mode,
                        status="ERROR",
                        contract_pass=None,
                        inference_pass=False,
                        target_gpu=target_gpu,
                        min_vram_gb=min_vram_gb,
                        e2e_latency_ms=round(elapsed_ms, 2),
                        error_code=f"HTTP_{poll_resp.status_code}",
                        error_message=f"Control API server error HTTP {poll_resp.status_code} while polling job '{job_id}': {poll_resp.text[:200]}",
                        nvml_trace_count=len(samples),
                        is_negative=False,
                    )

            # Polling concluded - terminal state reached
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            samples = self.sampler.stop()
            e2e_latency_ms = round(elapsed_ms, 2)

            v0_peak = self.sampler.get_peak_vram_mb(0)
            v1_peak = self.sampler.get_peak_vram_mb(1)

            chk_repo, chk_rev, chk_hash = self._extract_checkpoint_provenance(model_id, job_data)

            # Handle non-successful terminal states
            if current_state == "FAILED":
                err_dict = job_data.get("error", {})
                err_code = err_dict.get("code") if isinstance(err_dict, dict) else "JOB_FAILED"
                err_msg = err_dict.get("message") if isinstance(err_dict, dict) else str(err_dict)
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    error_code=str(err_code),
                    error_message=str(err_msg),
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )
            elif current_state == "CANCELLED":
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    error_code="JOB_CANCELLED",
                    error_message=f"Job entered terminal state 'CANCELLED'.",
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )
            elif current_state != "SUCCEEDED":
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="BACKEND_UNAVAILABLE",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    error_code="BACKEND_UNAVAILABLE",
                    error_message=f"Job remained in non-terminal state '{current_state}'. Backend worker not active.",
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

            # State is SUCCEEDED: Retrieve artifact deterministically
            output_content: Any = None
            artifact_path: Optional[str] = None
            primary_artifact: Optional[Dict[str, Any]] = None

            artifacts = job_data.get("artifacts", [])
            expected_mime = expected_output.get("mime")
            expected_format = expected_output.get("format")

            # 1. Match from job_data artifacts list by expected MIME or ID
            if isinstance(artifacts, list) and artifacts:
                if expected_mime:
                    for art in artifacts:
                        if isinstance(art, dict) and art.get("mime_type") == expected_mime:
                            primary_artifact = art
                            break
                if primary_artifact is None and isinstance(artifacts[0], dict):
                    primary_artifact = artifacts[0]

            if primary_artifact:
                sp = primary_artifact.get("storage_path")
                if sp and os.path.exists(sp):
                    artifact_path = sp
                    output_content = Path(sp)
                elif job_id:
                    art_id = primary_artifact.get("artifact_id")
                    scratch_job_dir = SCRATCH_DIR / job_id
                    if scratch_job_dir.exists() and art_id:
                        matched_files = list(scratch_job_dir.glob(f"{art_id}*"))
                        if matched_files:
                            artifact_path = str(matched_files[0])
                            output_content = Path(artifact_path)

            # 2. Check local scratch artifacts directory by format (not arbitrary files[0])
            if output_content is None and job_id:
                scratch_job_dir = SCRATCH_DIR / job_id
                if scratch_job_dir.exists():
                    glob_pat = f"*.{expected_format}" if expected_format else "*"
                    candidates = [
                        f for f in scratch_job_dir.glob(glob_pat)
                        if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".log")
                    ]
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
                        artifact_path = str(candidates[0])
                        output_content = Path(artifact_path)

            # 3. Check job_data result/output payloads
            if output_content is None:
                if "output" in job_data and job_data["output"] is not None:
                    output_content = job_data["output"]
                elif "result" in job_data and job_data["result"] is not None:
                    output_content = job_data["result"]

            # 4. If still None, retrieve binary or structured artifact via /v1/jobs/{job_id}/result
            if output_content is None and job_id:
                result_resp = self.client.get(f"/v1/jobs/{job_id}/result", headers=self.auth_headers)
                if result_resp.status_code == 200:
                    ct = result_resp.headers.get("content-type", "")
                    if "application/json" in ct:
                        output_content = result_resp.json()
                    else:
                        output_content = result_resp.content

            if output_content is None:
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    error_code="ARTIFACT_MISSING",
                    error_message=f"Job '{job_id}' succeeded but no artifact or result payload was delivered.",
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

            # Bind expected SHA-256 and byte sizes from primary_artifact
            rules = dict(expected_output.get("rules") or {})
            if primary_artifact:
                meta_sha = primary_artifact.get("sha256")
                meta_size = primary_artifact.get("size_bytes")
                if meta_sha and "expected_sha256" not in rules:
                    rules["expected_sha256"] = meta_sha
                if meta_size and "expected_size_bytes" not in rules:
                    rules["expected_size_bytes"] = meta_size

            # Validate output artifact
            val_res: ValidationResult = validate_artifact(
                content=output_content,
                expected_mime=expected_output.get("mime"),
                expected_format=expected_output.get("format"),
                min_bytes=expected_output.get("min_bytes", 1),
                rules=rules,
            )

            # Measured inference time - NEVER synthetic multipliers
            measured_inf_ms = (
                job_data.get("telemetry", {}).get("inference_duration_ms")
                if isinstance(job_data.get("telemetry"), dict)
                else None
            )
            if measured_inf_ms is None and isinstance(job_data.get("timings"), dict):
                measured_inf_ms = job_data["timings"].get("inference_ms")

            # Telemetry Integrity: When measured_inf_ms is None, keep final_inference_time_ms = None. NEVER substitute e2e_latency_ms.
            final_inference_time_ms = round(measured_inf_ms, 2) if measured_inf_ms is not None else None

            measured_load_ms = None
            if isinstance(job_data.get("telemetry"), dict):
                measured_load_ms = job_data["telemetry"].get("load_duration_ms")
            if measured_load_ms is None and isinstance(job_data.get("timings"), dict):
                measured_load_ms = job_data["timings"].get("load_ms")
            final_load_time_ms = round(measured_load_ms, 2) if measured_load_ms is not None else None

            # Reject INFERENCE_PASS if worker lease, execution evidence, or non-empty output is missing
            has_worker = bool(job_data.get("worker"))
            has_output = bool(val_res.valid and val_res.size_bytes > 0)
            has_duration = bool(measured_inf_ms is not None or (e2e_latency_ms is not None and e2e_latency_ms > 0))

            if not (has_worker and has_output and has_duration):
                missing_evidence = []
                if not has_worker:
                    missing_evidence.append("worker lease")
                if not has_output:
                    missing_evidence.append("non-empty validated output")
                if not has_duration:
                    missing_evidence.append("execution duration evidence")
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    inference_time_ms=final_inference_time_ms,
                    load_time_ms=final_load_time_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    error_code="EXECUTION_EVIDENCE_MISSING",
                    error_message=f"Job marked SUCCEEDED but missing evidence: {', '.join(missing_evidence)}.",
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

            if val_res.valid:
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="PASS",
                    contract_pass=None,
                    inference_pass=True,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    inference_time_ms=final_inference_time_ms,
                    load_time_ms=final_load_time_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    output_path=str(artifact_path) if artifact_path else None,
                    output_bytes=val_res.size_bytes,
                    artifact_mime=val_res.mime_type,
                    artifact_sha256=val_res.sha256,
                    artifact_dimensions=val_res.dimensions,
                    artifact_duration_seconds=val_res.duration_seconds,
                    artifact_validated=True,
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )
            else:
                val_err_code = getattr(val_res, "error_code", None) or "VALIDATION_FAILED"
                return CaseExecutionRecord(
                    case_id=cid,
                    run_id=self.run_id,
                    git_commit=self.git_commit,
                    model_id=model_id,
                    task=task,
                    variant=variant,
                    mode=mode,
                    status="FAIL",
                    contract_pass=None,
                    inference_pass=False,
                    target_gpu=target_gpu,
                    min_vram_gb=min_vram_gb,
                    e2e_latency_ms=e2e_latency_ms,
                    inference_time_ms=final_inference_time_ms,
                    load_time_ms=final_load_time_ms,
                    peak_vram_mb_gpu0=v0_peak,
                    peak_vram_mb_gpu1=v1_peak,
                    checkpoint_repo=chk_repo,
                    checkpoint_revision=chk_rev,
                    checkpoint_hash=chk_hash,
                    output_bytes=val_res.size_bytes,
                    artifact_mime=val_res.mime_type,
                    artifact_sha256=val_res.sha256,
                    artifact_validated=False,
                    error_code=val_err_code,
                    error_message=val_res.error,
                    nvml_trace_count=len(samples),
                    is_negative=False,
                )

        except Exception as exc:
            self.sampler.stop()
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception(f"[{cid}] Unexpected error during execution: {exc}")
            return CaseExecutionRecord(
                case_id=cid,
                run_id=self.run_id,
                git_commit=self.git_commit,
                model_id=model_id,
                task=task,
                variant=variant,
                mode=mode,
                status="ERROR",
                target_gpu=target_gpu,
                min_vram_gb=min_vram_gb,
                e2e_latency_ms=round(elapsed_ms, 2),
                error_code="UNEXPECTED_ERROR",
                error_message=str(exc),
            )

    def run_benchmark(
        self,
        mode: str = "smoke",
        model_filter: Optional[str] = None,
        case_id_filter: Optional[str] = None,
        perf_iterations: int = 3,
    ) -> List[CaseExecutionRecord]:
        """
        Executes benchmark suite for declared cases matching criteria.
        """
        logger.info(f"Starting Factory v4 Lite Unified Benchmark [mode={mode}, run_id={self.run_id}]...")

        selected_cases = []
        for c in self.cases:
            if mode not in c.get("modes", ["smoke", "full", "perf"]):
                continue
            if model_filter and c["model"] != model_filter:
                continue
            if case_id_filter and c["id"] != case_id_filter:
                continue
            selected_cases.append(c)

        logger.info(f"Loaded {len(selected_cases)} cases matching criteria.")
        results: List[CaseExecutionRecord] = []

        for idx, case in enumerate(selected_cases, 1):
            cid = case["id"]
            logger.info(f"[{idx}/{len(selected_cases)}] Evaluating {cid} ({case['model']} - {case['task']})...")

            if mode == "perf" and not case.get("is_negative", False):
                iter_records: List[CaseExecutionRecord] = []
                for it in range(perf_iterations):
                    r = self.evaluate_case(case, mode=mode)
                    iter_records.append(r)

                # Filter strictly to passing iterations for performance metrics
                passing_runs = [
                    r for r in iter_records
                    if r.status in ("PASS", "INFERENCE_PASS") or getattr(r, "inference_pass", False) is True
                ]
                passing_e2e_lats = sorted([r.e2e_latency_ms for r in passing_runs if r.e2e_latency_ms is not None])
                passing_inf_lats = sorted([r.inference_time_ms for r in passing_runs if r.inference_time_ms is not None])

                def calc_percentile(data: List[float], p: float) -> Optional[float]:
                    if not data:
                        return None
                    k = (len(data) - 1) * (p / 100.0)
                    f = math.floor(k)
                    c = math.ceil(k)
                    if f == c:
                        return round(data[int(k)], 2)
                    return round(data[f] * (c - k) + data[c] * (k - f), 2)

                median_e2e = calc_percentile(passing_e2e_lats, 50.0)
                p95_e2e = calc_percentile(passing_e2e_lats, 95.0)
                median_inf = calc_percentile(passing_inf_lats, 50.0)
                p95_inf = calc_percentile(passing_inf_lats, 95.0)

                # Aggregate peak VRAM as maximum across all iterations
                v0_peaks = [r.peak_vram_mb_gpu0 for r in iter_records if r.peak_vram_mb_gpu0 is not None]
                v1_peaks = [r.peak_vram_mb_gpu1 for r in iter_records if r.peak_vram_mb_gpu1 is not None]
                max_v0 = max(v0_peaks) if v0_peaks else None
                max_v1 = max(v1_peaks) if v1_peaks else None

                success_rate = round((len(passing_runs) / perf_iterations) * 100.0, 1) if perf_iterations > 0 else 0.0

                base_rec = passing_runs[0] if passing_runs else iter_records[0]
                perf_rec = copy.deepcopy(base_rec)
                perf_rec.e2e_latency_ms = median_e2e
                perf_rec.inference_time_ms = median_inf
                perf_rec.peak_vram_mb_gpu0 = max_v0
                perf_rec.peak_vram_mb_gpu1 = max_v1
                if not passing_runs:
                    perf_rec.status = base_rec.status
                    perf_rec.inference_pass = False
                else:
                    perf_rec.status = "PASS"
                    perf_rec.inference_pass = True

                perf_rec.perf_metadata = {
                    "iterations_total": perf_iterations,
                    "iterations_passed": len(passing_runs),
                    "iteration_success_rate": success_rate,
                    "e2e_latency_median_ms": median_e2e,
                    "e2e_latency_p95_ms": p95_e2e,
                    "inference_time_median_ms": median_inf,
                    "inference_time_p95_ms": p95_inf,
                    "all_iterations": [r.to_canonical_dict() for r in iter_records],
                }
                results.append(perf_rec)
            else:
                rec = self.evaluate_case(case, mode=mode)
                results.append(rec)

        # Generate deliverables
        report_gen = BenchmarkReportGenerator(
            run_id=self.run_id,
            git_commit=self.git_commit,
            mode=mode,
            hardware_info=self.hardware_info,
            records=results,
            output_dir=self.output_dir,
        )
        report_gen.generate_all()

        logger.info(f"Benchmark run {self.run_id} finished. Results written to {self.output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Factory v4 Lite Unified Benchmark Runner")
    parser.add_argument("--mode", choices=["smoke", "full", "perf"], default="smoke", help="Execution mode")
    parser.add_argument("--model", type=str, default=None, help="Filter by model ID")
    parser.add_argument("--case-id", type=str, default=None, help="Filter by specific case ID")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--cases-file", type=str, default=None, help="Path to cases.yaml")
    parser.add_argument("--base-url", type=str, default=None, help="FastAPI server URL")
    args = parser.parse_args()

    cases_path = Path(args.cases_file) if args.cases_file else None
    out_dir = Path(args.output_dir) if args.output_dir else None

    runner = UnifiedBenchmarkRunner(cases_file=cases_path, output_base_dir=out_dir, base_url=args.base_url)
    try:
        results = runner.run_benchmark(
            mode=args.mode,
            model_filter=args.model,
            case_id_filter=args.case_id,
        )
        pass_count = sum(1 for r in results if r.status == "PASS")
        logger.info(f"Run completed: {pass_count}/{len(results)} PASS.")
    finally:
        runner.close()


if __name__ == "__main__":
    main()

