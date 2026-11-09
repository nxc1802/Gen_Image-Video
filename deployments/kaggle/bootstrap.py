"""
deployments.kaggle.bootstrap
Unified Kaggle Dual-T4 Bootstrap & Environment Preparation Engine for Factory v4 Lite.

Consolidates:
1. Hardware topology detection (GPU detection, Dual-T4 verification, 14.2 GB VRAM allocatable limits, RAM, disk).
2. Kaggle dataset discovery, validation, and mounting / symlinking.
3. Subsystem runtime dependency validation (torch, transformers, diffusers, vllm, audio, video).
4. Ephemeral scratch storage initialization (/tmp/factory_lite/artifacts/) and secret hygiene.
5. Reproducible execution entrypoint for Kaggle notebooks and automated benchmarks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kaggle_bootstrap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Dual-T4 Invariants
TOTAL_VRAM_PER_GPU_MB = 15360.0  # 15.0 GB physical ceiling
MAX_ALLOCATABLE_VRAM_PER_GPU_MB = 14540.0  # 14.2 GB allocatable limit (leaves >=820 MB for CUDA context/activations)

SUBSYSTEM_GPU_MAP: Dict[str, int] = {
    "text": 0,
    "decision": 0,
    "vlm": 0,
    "vision": 0,
    "speech": 0,
    "image": 0,
    "embedding": 1,
    "rerank": 1,
    "video": 1,
    "music": 1,
    "enhancement": 1,
}

# Known Kaggle dataset slugs and expected weight files
DATASET_SPECS: List[Dict[str, Any]] = [
    {
        "id": "qwen-text",
        "slug": "qwen25-7b-instruct-awq",
        "huggingface": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "service": "text",
        "target_gpu": 0,
        "files": ["model.safetensors.index.json"],
    },
    {
        "id": "qwen-image-2.1",
        "slug": "qwen-image-2-1-gguf",
        "huggingface": "abenzerps/Qwen-Image-2.1-GGUF",
        "service": "image",
        "target_gpu": 0,
        "files": ["qwen-image-2.1-Q4_K_M.gguf"],
    },
    {
        "id": "qwen-image-2.1-turbo",
        "slug": "qwen-image-2-1-turbo-gguf",
        "huggingface": "unsloth/Qwen-Image-2.1-Turbo-GGUF",
        "service": "image",
        "target_gpu": 0,
        "files": ["qwen-image-2.1-turbo-Q4_K_M.gguf"],
    },
    {
        "id": "wan-2.2",
        "slug": "wan22-ti2v-5b-diffusers",
        "huggingface": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "service": "video",
        "target_gpu": 1,
        "files": ["model_index.json"],
    },
    {
        "id": "minimax-h3",
        "slug": "hailuo-h3-diffusers",
        "huggingface": "MiniMax/Hailuo-H3-Diffusers",
        "service": "video",
        "target_gpu": 1,
        "files": ["model_index.json"],
    },
    {
        "id": "ace-step-1.5",
        "slug": "acestep-1-5",
        "huggingface": "ACE-Studio/ACE-Step-1.5",
        "service": "music",
        "target_gpu": 1,
        "files": ["model.safetensors"],
    },
    {
        "id": "yue2",
        "slug": "yue-s1-7b",
        "huggingface": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "service": "music",
        "target_gpu": 1,
        "files": ["pytorch_model.bin"],
    },
    {
        "id": "qwen-speech",
        "slug": "qwen3-asr-1-7b",
        "huggingface": "Qwen/Qwen3-ASR-1.7B",
        "service": "speech",
        "target_gpu": 0,
        "files": ["model.safetensors"],
    },
    {
        "id": "qwen-decision",
        "slug": "qwen25-3b-instruct",
        "huggingface": "Qwen/Qwen2.5-3B-Instruct",
        "service": "decision",
        "target_gpu": 0,
        "files": ["model.safetensors"],
    },
    {
        "id": "qwen-vlm",
        "slug": "qwen2-vl-7b-instruct-awq",
        "huggingface": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "service": "vision",
        "target_gpu": 0,
        "files": ["model.safetensors.index.json"],
    },
    {
        "id": "qwen-embedding",
        "slug": "qwen3-embedding-0-6b",
        "huggingface": "Qwen/Qwen3-Embedding-0.6B",
        "service": "embedding",
        "target_gpu": 1,
        "files": ["model.safetensors"],
    },
    {
        "id": "enhancement-suite-v1",
        "slug": "real-esrgan",
        "huggingface": "xinntao/Real-ESRGAN",
        "service": "enhancement",
        "target_gpu": 1,
        "files": ["RealESRGAN_x4plus.pth"],
    },
]

REQUIRED_PACKAGES: Dict[str, str] = {
    "torch": ">=2.4.0",
    "fastapi": ">=0.100.0",
    "pydantic": ">=2.0.0",
    "yaml": ">=6.0",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "transformers": ">=4.40.0",
    "diffusers": ">=0.30.0",
    "torchaudio": ">=2.4.0",
    "torchvision": ">=0.19.0",
    "soundfile": ">=0.12.0",
    "imageio": ">=2.30.0",
    "pynvml": ">=11.5.0",
}


@dataclass
class HardwareReport:
    platform_system: str
    platform_release: str
    python_version: str
    cpu_count: int
    system_ram_gb: float
    scratch_disk_free_gb: float
    cuda_available: bool
    gpu_count: int
    gpu_names: List[str] = field(default_factory=list)
    vram_per_gpu_mb: List[float] = field(default_factory=list)
    dual_t4_detected: bool = False
    max_allocatable_vram_mb: float = MAX_ALLOCATABLE_VRAM_PER_GPU_MB


@dataclass
class BootstrapResult:
    success: bool
    timestamp: str
    hardware: HardwareReport
    dependencies: Dict[str, Any]
    datasets: Dict[str, Any]
    storage: Dict[str, Any]
    auth_configured: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KaggleBootstrap:
    """Consolidated bootstrap and environment verification runner for Kaggle Dual-T4."""

    def __init__(
        self,
        kaggle_input_dir: Optional[Path] = None,
        scratch_artifacts_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ):
        self.kaggle_input_dir = kaggle_input_dir or Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
        self.scratch_artifacts_dir = scratch_artifacts_dir or Path(
            os.environ.get("SCRATCH_DIR", "/tmp/factory_lite/artifacts")
        )
        self.models_dir = models_dir or Path(os.environ.get("MODELS_DIR", WORKSPACE_ROOT / "models"))

    def detect_hardware(self) -> HardwareReport:
        """Detects host platform, memory, disk, and CUDA GPUs."""
        sys_info = platform.uname()

        # System RAM
        system_ram_gb = 0.0
        try:
            if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                system_ram_gb = round(
                    (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3), 2
                )
        except Exception:
            pass

        # Scratch disk space
        scratch_free_gb = 0.0
        try:
            probe_path = self.scratch_artifacts_dir.parent if self.scratch_artifacts_dir.parent.exists() else Path("/tmp")
            stat = shutil.disk_usage(probe_path)
            scratch_free_gb = round(stat.free / (1024**3), 2)
        except Exception:
            pass

        # CUDA & GPUs
        cuda_available = False
        gpu_count = 0
        gpu_names: List[str] = []
        vram_per_gpu_mb: List[float] = []

        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_names.append(torch.cuda.get_device_name(i))
                    props = torch.cuda.get_device_properties(i)
                    vram_per_gpu_mb.append(round(props.total_memory / (1024 * 1024), 1))
        except Exception:
            pass

        # Check NVML fallback if torch.cuda not initialized
        if gpu_count == 0:
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count > 0:
                    cuda_available = True
                    for i in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode("utf-8")
                        gpu_names.append(name)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_per_gpu_mb.append(round(mem.total / (1024 * 1024), 1))
            except Exception:
                pass

        # Determine if Dual-T4
        dual_t4_detected = (
            gpu_count >= 2
            and any("t4" in name.lower() or "tesla t4" in name.lower() for name in gpu_names)
        )

        return HardwareReport(
            platform_system=sys_info.system,
            platform_release=sys_info.release,
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            system_ram_gb=system_ram_gb,
            scratch_disk_free_gb=scratch_free_gb,
            cuda_available=cuda_available,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            vram_per_gpu_mb=vram_per_gpu_mb,
            dual_t4_detected=dual_t4_detected,
            max_allocatable_vram_mb=MAX_ALLOCATABLE_VRAM_PER_GPU_MB,
        )

    def validate_dependencies(self) -> Dict[str, Any]:
        """Validates critical and optional runtime dependencies."""
        installed: Dict[str, str] = {}
        missing_required: List[str] = []
        missing_optional: List[str] = []

        for pkg, spec in REQUIRED_PACKAGES.items():
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                installed[pkg] = ver
            except ImportError:
                missing_required.append(pkg)

        for pkg, spec in OPTIONAL_PACKAGES.items():
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                installed[pkg] = ver
            except ImportError:
                missing_optional.append(pkg)

        return {
            "all_required_present": len(missing_required) == 0,
            "installed": installed,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        }

    def discover_and_mount_datasets(
        self,
        create_symlinks: bool = True,
    ) -> Dict[str, Any]:
        """
        Discovers Kaggle datasets in /kaggle/input (or custom dir) and optionally
        symlinks model weights to models_dir.
        """
        discovered: Dict[str, Any] = {}
        missing_datasets: List[str] = []

        if not self.kaggle_input_dir.exists():
            logger.info(f"Kaggle input directory {self.kaggle_input_dir} not mounted.")
            return {
                "input_dir": str(self.kaggle_input_dir),
                "mounted": False,
                "discovered": {},
                "missing": [spec["slug"] for spec in DATASET_SPECS],
            }

        if create_symlinks:
            self.models_dir.mkdir(parents=True, exist_ok=True)

        for spec in DATASET_SPECS:
            model_id = spec["id"]
            slug = spec["slug"]
            dataset_path = self.kaggle_input_dir / slug

            if dataset_path.exists() and dataset_path.is_dir():
                found_files = []
                for fname in spec["files"]:
                    fpath = dataset_path / fname
                    if fpath.exists():
                        found_files.append(fname)
                        if create_symlinks:
                            dest_link = self.models_dir / fname
                            if not dest_link.exists():
                                try:
                                    dest_link.symlink_to(fpath)
                                except Exception:
                                    pass

                discovered[model_id] = {
                    "slug": slug,
                    "path": str(dataset_path),
                    "found_files": found_files,
                    "target_gpu": spec["target_gpu"],
                    "complete": len(found_files) == len(spec["files"]),
                }
            else:
                missing_datasets.append(slug)

        return {
            "input_dir": str(self.kaggle_input_dir),
            "mounted": True,
            "discovered": discovered,
            "missing": missing_datasets,
        }

    def initialize_environment(
        self,
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Initializes ephemeral scratch storage and configures required environment variables.
        When dry_run=True, verifies configuration without creating directories or mutating os.environ.
        """
        if dry_run:
            current_key = api_key or os.environ.get("FACTORY_API_KEY")
            return {
                "scratch_artifacts_dir": str(self.scratch_artifacts_dir),
                "storage_backend": "ephemeral_disk",
                "factory_db_backend": "memory",
                "api_key_configured": bool(current_key),
                "environment": "dry_run",
                "dry_run": True,
            }

        # Setup scratch artifacts directory
        self.scratch_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Resolve API Key
        final_key = api_key or os.environ.get("FACTORY_API_KEY")
        if not final_key:
            try:
                from kaggle_secrets import UserSecretsClient
                final_key = UserSecretsClient().get_secret("FACTORY_API_KEY")
            except Exception:
                pass
        if not final_key:
            import secrets
            final_key = f"sk-kaggle-{secrets.token_urlsafe(24)}"

        os.environ["FACTORY_API_KEY"] = final_key
        os.environ["STORAGE_BACKEND"] = "ephemeral_disk"
        os.environ["SCRATCH_DIR"] = str(self.scratch_artifacts_dir)
        os.environ["FACTORY_DB_BACKEND"] = "memory"
        os.environ["ENVIRONMENT"] = "production" if "/kaggle" in str(self.kaggle_input_dir) else "test"

        return {
            "scratch_artifacts_dir": str(self.scratch_artifacts_dir),
            "storage_backend": os.environ["STORAGE_BACKEND"],
            "factory_db_backend": os.environ["FACTORY_DB_BACKEND"],
            "api_key_configured": bool(final_key),
            "environment": os.environ["ENVIRONMENT"],
            "dry_run": False,
        }

    def install_dependencies(self, packages: Optional[List[str]] = None) -> bool:
        """Installs missing dependencies via pip."""
        target_pkgs = packages or list(REQUIRED_PACKAGES.keys())
        logger.info(f"Installing dependencies: {target_pkgs}")
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + target_pkgs
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(f"Dependency installation failed: {exc.stderr}")
            return False

    def run_bootstrap(
        self,
        check_only: bool = False,
        install_missing: bool = False,
        mount_datasets: bool = True,
        output_file: Optional[Path] = None,
    ) -> BootstrapResult:
        """Executes complete bootstrap sequence."""
        t0 = time.time()
        logger.info("Executing Factory v4 Lite Kaggle Dual-T4 Bootstrap...")

        hw = self.detect_hardware()
        deps = self.validate_dependencies()

        if install_missing and not deps["all_required_present"]:
            self.install_dependencies(deps["missing_required"])
            deps = self.validate_dependencies()

        if check_only:
            datasets_info = self.discover_and_mount_datasets(create_symlinks=False)
            storage_info = self.initialize_environment(dry_run=True)
        else:
            datasets_info = (
                self.discover_and_mount_datasets(create_symlinks=True)
                if mount_datasets
                else {"mounted": False, "discovered": {}, "missing": []}
            )
            storage_info = self.initialize_environment(dry_run=False)

        is_kaggle_env = Path("/kaggle").exists()
        errors: List[str] = []
        if is_kaggle_env:
            if not deps["all_required_present"]:
                errors.append(f"Missing required dependencies: {deps['missing_required']}")
            if hw.gpu_count < 2:
                errors.append(f"Expected Dual-T4 GPU configuration on Kaggle, but detected {hw.gpu_count} GPU(s).")
        else:
            if not deps["all_required_present"]:
                logger.info(f"Local dev notice: dependencies not yet installed in local environment: {deps['missing_required']}")
            if hw.gpu_count < 2:
                logger.info(f"Local dev notice: non-GPU host ({hw.gpu_count} GPUs detected).")


        result = BootstrapResult(
            success=len(errors) == 0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            hardware=hw,
            dependencies=deps,
            datasets=datasets_info,
            storage=storage_info,
            auth_configured=storage_info["api_key_configured"],
            errors=errors,
        )

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Bootstrap report saved to {output_file}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Factory v4 Lite Unified Kaggle Bootstrap")
    parser.add_argument("--check-only", action="store_true", help="Only verify environment without modifying files")
    parser.add_argument("--install-missing", action="store_true", help="Install missing packages via pip")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input", help="Kaggle datasets root directory")
    parser.add_argument("--output", type=str, default="bootstrap_summary.json", help="Path to save bootstrap summary")
    args = parser.parse_args()

    bootstrapper = KaggleBootstrap(kaggle_input_dir=Path(args.input_dir))
    res = bootstrapper.run_bootstrap(
        check_only=args.check_only,
        install_missing=args.install_missing,
        mount_datasets=not args.check_only,
        output_file=Path(args.output),
    )

    if not res.success:
        logger.warning(f"Bootstrap completed with warnings/errors: {res.errors}")
        sys.exit(1 if not args.check_only else 0)
    else:
        logger.info("Bootstrap completed successfully. System ready for benchmark execution.")
        sys.exit(0)


if __name__ == "__main__":
    main()
