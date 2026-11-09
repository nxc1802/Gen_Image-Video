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
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.request

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

# Service backend mapping for health probes and port reconciliation
SERVICE_BACKEND_MAP: Dict[str, Dict[str, Any]] = {
    "qwen-text": {"service_id": "text-service", "port": 8080, "health_endpoint": "/health", "target_gpu": 0},
    "qwen-vlm": {"service_id": "vision-service", "port": 8081, "health_endpoint": "/health", "target_gpu": 0},
    "qwen-decision": {"service_id": "decision-service", "port": 8082, "health_endpoint": "/health", "target_gpu": 0},
    "qwen-embedding": {"service_id": "embed-service", "port": 8083, "health_endpoint": "/health", "target_gpu": 1},
    "qwen-speech": {"service_id": "speech-service", "port": 8288, "health_endpoint": "/health", "target_gpu": 0},
    "qwen-image-2.1": {"service_id": "image-service", "port": 8188, "health_endpoint": "/system_stats", "target_gpu": 0},
    "qwen-image-2.1-turbo": {"service_id": "image-service", "port": 8188, "health_endpoint": "/system_stats", "target_gpu": 0},
    "wan-2.2": {"service_id": "video-service", "port": 8189, "health_endpoint": "/system_stats", "target_gpu": 1},
    "minimax-h3": {"service_id": "video-service", "port": 8189, "health_endpoint": "/system_stats", "target_gpu": 1},
    "ace-step-1.5": {"service_id": "music-service", "port": 8008, "health_endpoint": "/health", "target_gpu": 1},
    "yue2": {"service_id": "music-service", "port": 8009, "health_endpoint": "/health", "target_gpu": 1},
    "enhancement-suite-v1": {"service_id": "enhancement-service", "port": 8002, "health_endpoint": "/health", "target_gpu": 1},
}

# Known Kaggle dataset slugs and expected weight files
DATASET_SPECS: List[Dict[str, Any]] = [
    {
        "id": "qwen-text",
        "slug": "qwen25-7b-instruct-awq",
        "huggingface": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "service": "text",
        "target_gpu": 0,
        "format": "safetensors",
        "min_vram_gb": 12.0,
        "files": ["model.safetensors.index.json"],
    },
    {
        "id": "qwen-image-2.1",
        "slug": "qwen-image-2-1-gguf",
        "huggingface": "abenzerps/Qwen-Image-2.1-GGUF",
        "service": "image",
        "target_gpu": 0,
        "format": "gguf",
        "min_vram_gb": 14.0,
        "files": ["qwen-image-2.1-Q4_K_M.gguf"],
        "components": {
            "clip": "qwen3vl_8b_fp8_scaled.safetensors",
            "vae": "qwen_image_2.1_vae_bf16.safetensors",
            "control_pose": "qwen_image_control_pose.safetensors",
            "control_depth": "qwen_image_control_depth.safetensors",
        },
    },
    {
        "id": "qwen-image-2.1-turbo",
        "slug": "qwen-image-2-1-turbo-gguf",
        "huggingface": "unsloth/Qwen-Image-2.1-Turbo-GGUF",
        "service": "image",
        "target_gpu": 0,
        "format": "gguf",
        "min_vram_gb": 14.0,
        "files": ["qwen-image-2.1-turbo-Q4_K_M.gguf"],
    },
    {
        "id": "wan-2.2",
        "slug": "wan22-ti2v-5b-diffusers",
        "huggingface": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "service": "video",
        "target_gpu": 1,
        "format": "diffusers",
        "min_vram_gb": 14.0,
        "files": ["model_index.json"],
    },
    {
        "id": "minimax-h3",
        "slug": "hailuo-h3-diffusers",
        "huggingface": "MiniMax/Hailuo-H3-Diffusers",
        "service": "video",
        "target_gpu": 1,
        "format": "diffusers",
        "min_vram_gb": 14.0,
        "files": ["model_index.json"],
        "components": {
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
        },
    },
    {
        "id": "ace-step-1.5",
        "slug": "acestep-1-5",
        "huggingface": "ACE-Studio/ACE-Step-1.5",
        "service": "music",
        "target_gpu": 1,
        "format": "safetensors",
        "min_vram_gb": 12.0,
        "files": ["model.safetensors"],
    },
    {
        "id": "yue2",
        "slug": "yue-s1-7b",
        "huggingface": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "service": "music",
        "target_gpu": 1,
        "format": "safetensors",
        "min_vram_gb": 12.0,
        "files": ["model.safetensors"],
    },
    {
        "id": "qwen-speech",
        "slug": "qwen3-asr-1-7b",
        "huggingface": "Qwen/Qwen3-ASR-1.7B",
        "service": "speech",
        "target_gpu": 0,
        "format": "safetensors",
        "min_vram_gb": 8.0,
        "files": ["model.safetensors"],
        "components": {
            "tts_base": "tts_base/model.safetensors",
        },
    },
    {
        "id": "qwen-decision",
        "slug": "qwen25-3b-instruct",
        "huggingface": "Qwen/Qwen2.5-3B-Instruct",
        "service": "decision",
        "target_gpu": 0,
        "format": "safetensors",
        "min_vram_gb": 6.0,
        "files": ["model.safetensors"],
    },
    {
        "id": "qwen-vlm",
        "slug": "qwen2-vl-7b-instruct-awq",
        "huggingface": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "service": "vision",
        "target_gpu": 0,
        "format": "safetensors",
        "min_vram_gb": 14.0,
        "files": ["model.safetensors.index.json"],
    },
    {
        "id": "qwen-embedding",
        "slug": "qwen3-embedding-0-6b",
        "huggingface": "Qwen/Qwen3-Embedding-0.6B",
        "service": "embedding",
        "target_gpu": 1,
        "format": "safetensors",
        "min_vram_gb": 4.0,
        "files": ["model.safetensors"],
        "components": {
            "reranker": "reranker/model.safetensors",
        },
    },
    {
        "id": "enhancement-suite-v1",
        "slug": "real-esrgan",
        "huggingface": "xinntao/Real-ESRGAN",
        "service": "enhancement",
        "target_gpu": 1,
        "format": "safetensors",
        "min_vram_gb": 6.0,
        "files": ["RealESRGAN_x4plus.pth"],
        "components": {
            "rife": "rife.pth",
            "bsrgan": "bsrgan.pth",
            "audio_denoise": "deepfilternet.pth",
        },
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
    backend_health: Dict[str, Any] = field(default_factory=dict)
    model_readiness: Dict[str, Any] = field(default_factory=dict)

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

    @staticmethod
    def _inspect_file(path: Path) -> Tuple[bool, Optional[str], int]:
        """Validates that a path exists, is a regular file, and has non-zero size."""
        if not path.exists():
            return False, "missing", 0
        if not path.is_file():
            return False, "not_a_regular_file", 0
        try:
            size = path.stat().st_size
            if size == 0:
                return False, "zero_bytes", 0
            return True, None, size
        except Exception as exc:
            return False, f"stat_error: {exc}", 0

    @classmethod
    def _verify_safetensors_index(
        cls,
        index_path: Path,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Parses model.safetensors.index.json to extract and verify constituent shards.
        Returns (valid, found_shards, missing_or_corrupt_shards).
        """
        valid, reason, _ = cls._inspect_file(index_path)
        if not valid:
            return False, [], [f"{index_path.name} ({reason})"]

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                return False, [], [f"{index_path.name} (weight_map is missing or empty)"]
            shards = sorted(set(weight_map.values()))
        except Exception as exc:
            return False, [], [f"{index_path.name} (parse error: {exc})"]

        found_shards: List[str] = []
        missing_shards: List[str] = []
        base_dir = index_path.parent
        for shard in shards:
            s_path = base_dir / shard
            s_valid, s_reason, _ = cls._inspect_file(s_path)
            if s_valid:
                found_shards.append(shard)
            else:
                missing_shards.append(f"{shard} ({s_reason})")

        return len(missing_shards) == 0, found_shards, missing_shards

    def discover_and_mount_datasets(
        self,
        create_symlinks: bool = True,
    ) -> Dict[str, Any]:
        """
        Discovers Kaggle datasets in /kaggle/input (or custom dir) and optionally
        symlinks model weights to models_dir with model_id namespacing.
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
                found_files: List[str] = []
                missing_files: List[str] = []
                corrupted_files: List[str] = []
                found_components: Dict[str, str] = {}
                missing_components: Dict[str, str] = {}
                mounted_symlinks: List[str] = []
                is_complete = True

                # Check safetensors index if present
                index_path = dataset_path / "model.safetensors.index.json"
                if index_path.exists():
                    idx_valid, idx_found, idx_missing = self._verify_safetensors_index(index_path)
                    if idx_valid:
                        found_files.append("model.safetensors.index.json")
                        found_files.extend(idx_found)
                    else:
                        is_complete = False
                        if self._inspect_file(index_path)[0]:
                            found_files.append("model.safetensors.index.json")
                        corrupted_files.extend(idx_missing)
                else:
                    # Check primary weight files declared
                    has_primary = False
                    for fname in spec.get("files", []):
                        if fname == "model.safetensors.index.json":
                            continue
                        fpath = dataset_path / fname
                        f_valid, f_reason, _ = self._inspect_file(fpath)
                        if f_valid:
                            found_files.append(fname)
                            has_primary = True
                        elif f_reason == "zero_bytes":
                            corrupted_files.append(f"{fname} (zero_bytes)")
                            is_complete = False
                        else:
                            missing_files.append(fname)
                    if spec.get("format") == "safetensors" and not has_primary:
                        is_complete = False
                    elif spec.get("format") == "gguf" and not has_primary:
                        is_complete = False

                # Check Diffusers models (wan-2.2, minimax-h3)
                if spec.get("format") == "diffusers":
                    m_index = dataset_path / "model_index.json"
                    m_valid, m_reason, _ = self._inspect_file(m_index)
                    if m_valid:
                        if "model_index.json" not in found_files:
                            found_files.append("model_index.json")
                    elif m_reason == "zero_bytes":
                        corrupted_files.append("model_index.json (zero_bytes)")
                        is_complete = False
                    else:
                        missing_files.append("model_index.json")
                        is_complete = False

                    # Diffusers marker alone is NOT proof of download: require actual non-zero weight files
                    weight_files = [
                        p for p in dataset_path.rglob("*")
                        if p.is_file()
                        and p.suffix.lower() in (".safetensors", ".bin", ".gguf", ".pt", ".pth")
                        and p.stat().st_size > 0
                    ]
                    if not weight_files:
                        corrupted_files.append("diffusers_pipeline_missing_weight_files (marker-only incomplete download)")
                        is_complete = False
                    else:
                        for wf in weight_files:
                            rel_w = str(wf.relative_to(dataset_path))
                            if rel_w not in found_files:
                                found_files.append(rel_w)

                # Validate declared components
                components_dict = spec.get("components", {})
                for comp_name, comp_target in components_dict.items():
                    c_path = dataset_path / comp_target
                    if not c_path.exists():
                        c_path = dataset_path / Path(comp_target).name

                    c_valid, c_reason, _ = self._inspect_file(c_path)
                    if c_valid:
                        rel_c = str(c_path.relative_to(dataset_path))
                        found_components[comp_name] = rel_c
                        if rel_c not in found_files:
                            found_files.append(rel_c)
                    elif c_reason == "zero_bytes":
                        missing_components[comp_name] = f"{comp_target} (zero_bytes)"
                        corrupted_files.append(f"{comp_name}: {comp_target} (zero_bytes)")
                        is_complete = False
                    else:
                        missing_components[comp_name] = f"{comp_target} (missing)"
                        is_complete = False

                # Symlinking with model_id namespacing: self.models_dir / model_id / fname
                if create_symlinks:
                    dest_model_dir = self.models_dir / model_id
                    dest_model_dir.mkdir(parents=True, exist_ok=True)
                    for fname in found_files:
                        src_path = dataset_path / fname
                        if src_path.exists() and src_path.is_file():
                            dest_link = dest_model_dir / fname
                            dest_link.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                if dest_link.is_symlink() or dest_link.exists():
                                    if dest_link.is_symlink() and dest_link.resolve() == src_path.resolve():
                                        pass
                                    else:
                                        dest_link.unlink(missing_ok=True)
                                        dest_link.symlink_to(src_path)
                                else:
                                    dest_link.symlink_to(src_path)
                                mounted_symlinks.append(str(dest_link))
                            except Exception as exc:
                                logger.warning(f"Failed to symlink {src_path} -> {dest_link}: {exc}")

                discovered[model_id] = {
                    "slug": slug,
                    "path": str(dataset_path),
                    "found_files": found_files,
                    "missing_files": missing_files,
                    "found_components": found_components,
                    "missing_components": missing_components,
                    "corrupted_files": corrupted_files,
                    "mounted_symlinks": mounted_symlinks,
                    "target_gpu": spec["target_gpu"],
                    "complete": is_complete and len(corrupted_files) == 0 and len(missing_components) == 0,
                }
            else:
                missing_datasets.append(slug)

        return {
            "input_dir": str(self.kaggle_input_dir),
            "mounted": True,
            "discovered": discovered,
            "missing": missing_datasets,
        }

    def check_hardware_feasibility(
        self,
        hw: HardwareReport,
        model_id: str,
        variant: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates whether the hardware can execute the specified model.
        Enforces 14.2 GB VRAM allocatable limit per discrete GPU.
        For YuE2, unquantized 7B BF16 reports HARDWARE_BLOCKED on T4 unless Track B weights are loaded.
        """
        if model_id == "yue2":
            is_track_b = (
                quantization in ("awq", "gguf", "int8", "int4", "4bit", "8bit")
                or variant in ("track_b", "quantized", "gguf")
            )
            # Dual T4 provides 14.2 GB allocatable per GPU and lacks native BF16 (Turing CC 7.5).
            # Track A requires >= 24GB VRAM with native BF16.
            if not is_track_b:
                return {
                    "model_id": model_id,
                    "status": "HARDWARE_BLOCKED",
                    "feasible": False,
                    "reason": (
                        "YuE2 unquantized 7B BF16 requires >= 24GB VRAM and native BF16 tensor cores (Track A). "
                        "Tesla T4 provides 14.2 GB allocatable VRAM per GPU without native BF16. "
                        "YuE2 reports HARDWARE_BLOCKED on T4 unless Track B (quantized/GGUF) weights are loaded."
                    ),
                }

        # Check single-card ceiling (14.2 GB allocatable)
        min_vram_gb = 0.0
        for spec in DATASET_SPECS:
            if spec["id"] == model_id:
                min_vram_gb = spec.get("min_vram_gb", 0.0)
                break

        max_allocatable_gb = MAX_ALLOCATABLE_VRAM_PER_GPU_MB / 1024.0
        if min_vram_gb > max_allocatable_gb:
            return {
                "model_id": model_id,
                "status": "HARDWARE_BLOCKED",
                "feasible": False,
                "reason": (
                    f"Model {model_id} requires {min_vram_gb:.1f} GB VRAM, exceeding single-card "
                    f"14.2 GB allocatable ceiling on Tesla T4."
                ),
            }

        return {
            "model_id": model_id,
            "status": "FEASIBLE",
            "feasible": True,
            "reason": f"Hardware within 14.2 GB allocatable limit for {model_id}.",
        }

    def probe_backend(
        self,
        host: str,
        port: int,
        endpoint: str = "/health",
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """Probes HTTP endpoint of a native backend daemon."""
        url = f"http://{host}:{port}{endpoint}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "KaggleBootstrapHealthProbe/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = resp.status
                is_live = 200 <= status_code < 300
                return {
                    "url": url,
                    "online": is_live,
                    "status_code": status_code,
                    "error": None,
                }
        except urllib.error.HTTPError as e:
            return {
                "url": url,
                "online": False,
                "status_code": e.code,
                "error": f"HTTP {e.code}: {e.reason}",
            }
        except Exception as exc:
            return {
                "url": url,
                "online": False,
                "status_code": None,
                "error": str(exc),
            }

    def probe_all_backends(
        self,
        host: str = "127.0.0.1",
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """Probes all native backends defined in SERVICE_BACKEND_MAP."""
        results: Dict[str, Any] = {}
        for mid, srv in SERVICE_BACKEND_MAP.items():
            results[mid] = self.probe_backend(host, srv["port"], srv["health_endpoint"], timeout=timeout)
        return results

    def check_model_readiness(
        self,
        model_id: str,
        dataset_entry: Optional[Dict[str, Any]] = None,
        hw_report: Optional[HardwareReport] = None,
        backend_probe: Optional[Dict[str, Any]] = None,
        host: str = "127.0.0.1",
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Marks model readiness as verified ONLY when:
        1. Hardware feasibility is satisfied (not HARDWARE_BLOCKED).
        2. Checkpoint weights are complete and non-corrupted.
        3. Native backend daemon is actively running and responsive to health probes.
        """
        hw = hw_report or self.detect_hardware()

        # 1. Hardware Feasibility
        feasibility = self.check_hardware_feasibility(hw, model_id)
        if not feasibility["feasible"]:
            return {
                "model_id": model_id,
                "ready": False,
                "status": feasibility["status"],
                "reason": feasibility["reason"],
                "checkpoint_valid": bool(dataset_entry and dataset_entry.get("complete")),
                "backend_online": False,
            }

        # 2. Checkpoint Integrity
        checkpoint_valid = bool(dataset_entry and dataset_entry.get("complete"))
        if not checkpoint_valid:
            return {
                "model_id": model_id,
                "ready": False,
                "status": "CHECKPOINT_MISSING",
                "reason": f"Model {model_id} checkpoint missing or incomplete in dataset mount.",
                "checkpoint_valid": False,
                "backend_online": False,
            }

        # 3. Active Backend Health Probe
        if backend_probe is None:
            srv = SERVICE_BACKEND_MAP.get(model_id, {"port": 8080, "health_endpoint": "/health"})
            backend_probe = self.probe_backend(host, srv["port"], srv["health_endpoint"], timeout=timeout)

        if not backend_probe.get("online"):
            return {
                "model_id": model_id,
                "ready": False,
                "status": "BACKEND_UNAVAILABLE",
                "reason": f"Native backend at {backend_probe.get('url')} is offline ({backend_probe.get('error')}).",
                "checkpoint_valid": True,
                "backend_online": False,
                "probe": backend_probe,
            }

        return {
            "model_id": model_id,
            "ready": True,
            "status": "VERIFIED",
            "reason": f"Model {model_id} verified: checkpoint complete, hardware feasible, native backend active.",
            "checkpoint_valid": True,
            "backend_online": True,
            "probe": backend_probe,
        }

    def evaluate_all_model_readiness(
        self,
        datasets_info: Dict[str, Any],
        hw: HardwareReport,
        backend_health: Dict[str, Any],
        host: str = "127.0.0.1",
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluates readiness across all models."""
        readiness: Dict[str, Any] = {}
        discovered = datasets_info.get("discovered", {})
        for spec in DATASET_SPECS:
            mid = spec["id"]
            d_entry = discovered.get(mid)
            b_probe = backend_health.get(mid)
            readiness[mid] = self.check_model_readiness(
                model_id=mid,
                dataset_entry=d_entry,
                hw_report=hw,
                backend_probe=b_probe,
                host=host,
                timeout=timeout,
            )
        return readiness

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
        probe_host: str = "127.0.0.1",
        probe_timeout: float = 1.0,
    ) -> BootstrapResult:
        """Executes complete bootstrap sequence."""
        t0 = time.time()
        logger.info("Executing Factory v4 Lite Kaggle Dual-T4 Bootstrap...")

        hw = self.detect_hardware()
        deps = self.validate_dependencies()

        if install_missing and not check_only and not deps["all_required_present"]:
            self.install_dependencies(deps["missing_required"])
            deps = self.validate_dependencies()
        elif install_missing and check_only and not deps["all_required_present"]:
            logger.info(
                f"Dry-run: dependency installation suppressed in --check-only mode for: {deps['missing_required']}"
            )

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

        backend_health = self.probe_all_backends(host=probe_host, timeout=probe_timeout)
        model_readiness = self.evaluate_all_model_readiness(
            datasets_info=datasets_info,
            hw=hw,
            backend_health=backend_health,
            host=probe_host,
            timeout=probe_timeout,
        )

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
            backend_health=backend_health,
            model_readiness=model_readiness,
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save bootstrap summary (default: bootstrap_summary.json; suppressed in --check-only unless explicitly specified)",
    )
    args = parser.parse_args()

    target_output: Optional[Path] = None
    if args.output is not None:
        target_output = Path(args.output)
    elif not args.check_only:
        target_output = Path("bootstrap_summary.json")

    bootstrapper = KaggleBootstrap(kaggle_input_dir=Path(args.input_dir))
    res = bootstrapper.run_bootstrap(
        check_only=args.check_only,
        install_missing=args.install_missing,
        mount_datasets=not args.check_only,
        output_file=target_output,
    )

    if not res.success:
        logger.warning(f"Bootstrap completed with warnings/errors: {res.errors}")
        sys.exit(1 if not args.check_only else 0)
    else:
        logger.info("Bootstrap completed successfully. System ready for benchmark execution.")
        sys.exit(0)


if __name__ == "__main__":
    main()
