# Kaggle Dual Tesla T4 Deployment & Execution Guide

## 1. Environment & Hardware Specifications

Factory v4 is architected for and verified on Kaggle Dual NVIDIA Tesla T4 environments:
- **Accelerators**: 2x NVIDIA Tesla T4 (16 GB GDDR6 per GPU, 32 GB total non-unified memory).
- **Allocatable Memory Ceiling**: **14.2 GB (14,540 MB)** allocatable VRAM per GPU (preserving >= 820 MB CUDA context headroom).
- **Driver / CUDA**: NVIDIA Driver 535+ with CUDA 12.1+.
- **Host System RAM**: ~30 GB System RAM on Kaggle standard Linux VMs.

---

## 2. Kaggle Notebook Configuration

When launching or creating a Kaggle Notebook for Factory v4 deployment:

| Setting | Required Value | Notes |
|---|---|---|
| **Accelerator** | `GPU T4 x2` | Dual GPU accelerator selection is mandatory. |
| **Internet** | `Internet On` | Mandatory for downloading wheels, ComfyUI nodes, and Hugging Face weights. |
| **Persistence** | `No persistence` or `Files only` | Scratch storage resides on `/tmp` and local ephemeral disk. |
| **Environment** | `Always use latest environment` | Python 3.10+ / 3.12 compatible. |

---

## 3. Kaggle Secrets Configuration

Factory v4 never hardcodes API tokens. Secrets must be attached to the Kaggle notebook via the Kaggle UI:
**Menu Path**: `Add-ons` -> `Secrets` -> Enter Key and Value.

| Secret Key | Description | Required? |
|---|---|---|
| `KAGGLE_USERNAME` | Your Kaggle username | Recommended |
| `KAGGLE_API_TOKEN` | Modern Kaggle Bearer Token (`KGAT_...`) from kaggle.com account settings | Required for Kaggle API dataset synchronization |
| `KAGGLE_KEY` | Legacy hex API key | Fallback if `KAGGLE_API_TOKEN` is unavailable |
| `HF_TOKEN` | Hugging Face Access Token | Required for gated models or private checkpoints |
| `FACTORY_API_KEY` | Factory Bearer key for Control Plane IPC | Recommended (defaults to auto-generated secret) |

### Programmatic Secrets Resolution
The runtime automatically resolves secrets in order of precedence:
1. Explicit CLI arguments (`--api-key`, `--token`).
2. Kaggle Secrets client (`kaggle_secrets.UserSecretsClient().get_secret(...)`).
3. Operating system environment variables (`os.environ["KAGGLE_API_TOKEN"]`, `os.environ["HF_TOKEN"]`).
4. Local file fallback (`~/.kaggle/kaggle.json` or `$KAGGLE_CONFIG_DIR/kaggle.json`).

---

## 4. Execution Workflow

### Step 1: Pre-Flight Verification & Bootstrap
Verify GPU topology, driver status, VRAM availability, and dataset mounts without executing workloads:

```bash
# Dry run verification
python -m deployments.kaggle.bootstrap --check-only

# Verify hardware and print topology report
python -m deployments.kaggle.live_gpu_test
```

### Step 2: Dataset Preparation
Download and verify checkpoint datasets from Kaggle / Hugging Face:

```bash
python -m deployments.kaggle.prepare_datasets
```

### Step 3: Run Visual Pipeline (Session 1)
Executes Qwen3.8-27B (Dual T4 tensor split) -> Qwen-Image 2.1 Turbo (8 Euler steps on GPU 0) -> MiniMax H3 (Two-pass decoupled ComfyUI on GPU 0):

```bash
python deployments/kaggle/session1_visual_pipeline.py
```

### Step 4: Run Audio Pipeline (Session 2)
Executes Qwen3-ASR (GPU 0), Qwen3-TTS (GPU 0), ACE-Step 1.5 (GPU 1), and YuE2 (GPU 1):

```bash
python deployments/kaggle/session2_audio_pipeline.py
```

### Step 5: Multi-Session Monitoring & Benchmark Aggregation
Monitor parallel pipeline execution and aggregate NVML telemetry:

```bash
python deployments/kaggle/monitor_both_sessions.py
```
