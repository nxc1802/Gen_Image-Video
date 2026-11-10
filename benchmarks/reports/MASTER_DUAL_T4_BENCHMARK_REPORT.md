# Factory v4 Lite — Dual-T4 Infrastructure Feasibility & Multi-Modal Attestation Report

> [!WARNING]
> **Audit Assessment (2026-11-10):**  
> This report documents **Physical Hardware Telemetry Attestation** and **Infrastructure Feasibility Smoke Tests with Alternative Pretrained Models** on Kaggle Dual Tesla T4.  
> It does **NOT** constitute final Production Certification for the target foundation models (`wan-2.2`, `minimax-h3`, `yue2`, `qwen-image-2.1`, `qwen-speech`) nor does it prove end-to-end Factory Control API $\to$ Queue $\to$ Worker $\to$ Native Engine execution.  
> Final Production Certification remains **BLOCKED / IN PROGRESS** pending native checkpoint verification and Factory E2E testing.

> **Report Type:** Feasibility & Telemetry Audit Report  
> **Master Execution ID:** `kaggle_dual_t4_live_20261110_final`  
> **Attestation Timestamp:** `2026-11-10T13:09:36Z` (20:09:36 GMT+7)  
> **Target Hardware:** Kaggle Dual NVIDIA Tesla T4 (2 × 15,360 MB GDDR6)  
> **Driver & CUDA:** NVIDIA Driver `580.178.04` | CUDA Version `13.0`  
> **Git Repository:** `nxc1802/Gen_Image-Video` | **Git Commit:** `cddf57f4584bcd84336d2df44a6d0b5c93e695cd`  

---

## 1. Executive Summary & Audit Alignment

In accordance with the Master Plan principles (**GPU-First, Inference-First, Evidence-Driven, Minimal Architecture**) and following an independent engineering audit on 2026-11-10, this report establishes a strict, truthful accounting of the platform's current maturity.

### 5-Level Certification Hierarchy

To eliminate false PASS and prevent conflating hardware telemetry or alternative model smoke tests with genuine foundation model certification, all future evaluation is governed by this hierarchy:

| Level | Classification | Definition |
|:---:|---|---|
| **L1** | `HARDWARE_ATTESTED` | Live NVML queries confirm physical GPU allocation, memory, and driver version. |
| **L2** | `SYNTHETIC_TENSOR_PASS` | Basic CUDA tensor operations (`torch.randn`, small Conv/Linear) verify register functionality. |
| **L3** | `FEASIBILITY_SMOKE_PASS` | Inference succeeds on GPU using an **alternative lightweight pretrained model** (e.g. SD-Turbo, Qwen2.5-0.5B, MiniLM, gTTS). Proves stack feasibility, **NOT** target model certification. |
| **L4** | `NATIVE_MODEL_PASS` | Exact checkpoint declared in the manifest is loaded into the designated native engine and completes inference on target GPU. |
| **L5** | `FACTORY_E2E_PASS` | Complete lifecycle execution: `Client Request` $\to$ `Control API` $\to$ `Queue/Scheduler` $\to$ `Worker Lease` $\to$ `Native Engine` $\to$ `Artifact Validation`. |

---

## 2. Complete 12-Model Manifest Matrix & Truthful Certification Status

Factory v4 Lite declares **12 distinct models** in `manifests/models/*.yaml`. Below is the honest, unvarnished accounting of their actual operational state:

| # | Model ID | Declared Task | Manifest Target Checkpoint | Target GPU | Level Attained | Detailed Current Reality |
|---|---|---|---|:---:|:---:|---|
| 1 | **`qwen-text`** | `chat_completion` | `Qwen3.8-27B-Q4_K_M.gguf` (17.67 GB) | **GPU 0** | **L3 (Feasibility)** | Executed `Qwen2.5-0.5B-Instruct` (208 tokens). Target 27B weights mounted on Kaggle but pending native llama-server E2E pass. |
| 2 | **`qwen-decision`** | `choice_routing` | GGUF / vLLM Decision Adapter | **GPU 0** | **L2 (Synthetic)** | Executed PyTorch synthetic linear score pass (173.2 ms). Native vLLM decision backend unconfigured on host. |
| 3 | **`qwen-vlm`** | `image_qa` | `Qwen3.8-27B` + `mmproj` (18.1 GB) | **GPU 0** | **L2 (Synthetic)** | Executed Conv2d patch projection on random tensor. Native multimodal engine pending Shard 1 execution. |
| 4 | **`qwen-image-2.1`** | `text_to_image` | Qwen-Image DiT FP16 (12.4 GB) | **GPU 0** | **L3 (Feasibility)** | Executed `stabilityai/sd-turbo` (512×512, 428 KB). Does **NOT** certify Qwen-Image 2.1. |
| 5 | **`qwen-image-2.1-turbo`**| `text_to_image_turbo`| Qwen-Image Turbo DiT (12.4 GB) | **GPU 0** | **L3 (Feasibility)** | Executed `stabilityai/sd-turbo` (1-2 steps). Native Qwen-Image Turbo pending Shard 1. |
| 6 | **`wan-2.2`** | `text_to_video` | `Wan2.2-TI2V-5B-Q4_K_M.gguf` (4.5 GB) | **GPU 1** | **L0 (Declared)** | Manifest declared. Shard 3 kernel pending execution. |
| 7 | **`minimax-h3`** | `t2va`, `i2va` | `MiniMax-H3-FL2VA-Q4_K_M.gguf` (10.6 GB) | **GPU 1** | **L2 (Synthetic)** | Executed Conv3d + H.264 MP4 encode from cropped still image. Does **NOT** certify video diffusion. |
| 8 | **`qwen-speech`** | `text_to_speech` | CosyVoice / Whisper (6.0 GB) | **GPU 0** | **L3 (Feasibility)** | Executed `gTTS` external service + ffmpeg WAV. Does **NOT** certify on-device Qwen-Speech. |
| 9 | **`ace-step-1.5`** | `text_to_music` | DiT Audio Latent (3.2 GB) | **GPU 1** | **L2 (Synthetic)** | Executed Conv1d stereo projection. Native DiT audio diffusion pending Shard 2. |
| 10 | **`yue2`** | `vocal_full_song` | Quantized GGUF `q8_0` (12.1 GB) | **GPU 1** | **L0 (Declared)** | Manifest declared. Shard 2 kernel pending execution. |
| 11 | **`enhancement-suite-v1`** | `super_resolution_4x` | Real-ESRGAN / RIFE (1.8 GB) | **GPU 1** | **L2 (Synthetic)** | Executed PyTorch PixelShuffle on random tensor. Native Real-ESRGAN/RIFE pending Shard 3. |
| 12 | **`qwen-embedding`** | `dense_embedding` | `Qwen-Embedding` (1024-dim) | **GPU 1** | **L3 (Feasibility)** | Executed `all-MiniLM-L6-v2` (384-dim). Does **NOT** certify Qwen-Embedding 1024-dim. |

---

## 3. Feasibility Smoke Test Evidence (Run: 2026-11-10)

The feasibility smoke tests executed in `deployments/kaggle/run_high_quality_test.py` confirm that the Kaggle compute environment can install Python libraries, access GPU accelerators via PyTorch/CUDA, and produce valid media files:

### 3.1. Text Feasibility: `Qwen/Qwen2.5-0.5B-Instruct` on `cuda:0`
- **Purpose:** GPU causal language modeling feasibility check.
- **Model Used:** `Qwen/Qwen2.5-0.5B-Instruct` (0.5B parameters, alternative lightweight model).
- **Inference Time:** `39,844.72 ms` (39.8s) | **Tokens:** `208`
- **Output:** `generated_text_response.txt` (`1,053 bytes`)
- **Caveat:** Validates text generation pipeline capability; does **NOT** validate target 27B model or MTP speculative decoding.

### 3.2. Image Feasibility: `stabilityai/sd-turbo` on `cuda:0`
- **Purpose:** GPU latent diffusion pipeline feasibility check.
- **Model Used:** `stabilityai/sd-turbo` (SD2.1-based adversarial diffusion, alternative model).
- **Inference Time:** `26,990.90 ms` (26.9s)
- **Output:** `high_quality_dragon_halong_512x512.png` (`427,898 bytes`, 512×512 PNG)
- **Caveat:** Validates Diffusers CUDA pipeline; does **NOT** validate Qwen-Image 2.1 DiT architecture.

### 3.3. Speech Feasibility: External TTS + ffmpeg
- **Purpose:** Audio formatting and PCM WAV encoding feasibility check.
- **Tools Used:** `gTTS` (cloud-based Google TTS) + `ffmpeg` PCM conversion.
- **Output:** `factory_intro_speech.wav` (`486,222 bytes`, 16kHz 16-bit mono)
- **Caveat:** Audio was synthesized via external web API; does **NOT** validate local on-GPU neural speech synthesis.

### 3.4. Video Feasibility: Frame Motion Interpolation + H.264
- **Purpose:** Video container and codec encoding feasibility check.
- **Method Used:** Cropping/zooming generated still image into 24 frames, encoded via `imageio-ffmpeg` (libx264).
- **Output:** `cinematic_dragon_halong_motion.mp4` (`83,309 bytes`, 24 frames @ 24fps)
- **Caveat:** Validates video packaging; does **NOT** validate spatio-temporal video diffusion (MiniMax H3 / Wan 2.2).

### 3.5. Embedding Feasibility: `all-MiniLM-L6-v2` on `cuda:1`
- **Purpose:** Sentence embedding and cosine similarity feasibility check.
- **Model Used:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, alternative model).
- **Cosine Similarity:** Matching pair `0.5968` vs differing pair `0.0084`.
- **Caveat:** Validates embedding pipeline; does **NOT** validate Qwen-Embedding 1024-dim model.

---

## 4. Physical Hardware Telemetry (NVML Attested)

Ingested directly from Kaggle compute instance registers:

- **Host Environment:** Linux x86_64, Kaggle Cloud Kernel
- **Driver Version:** `580.178.04`
- **CUDA Version:** `13.0`
- **Device Count:** `2`
- **GPU 0:** Tesla T4 — `15,360.0 MB` Total | `14,908.8 MB` Free | Util: `0%`
- **GPU 1:** Tesla T4 — `15,360.0 MB` Total | `14,908.8 MB` Free | Util: `0%`
- **Hardware Architecture Invariant:** Card 0 and Card 1 possess completely discrete memory pools. No unified 32 GB address space exists. Workloads requiring $>14.2\text{ GB}$ on a single card must be sharded or fail closed.

---

## 5. Artifact Checksum Ledger (For Feasibility Outputs)

| Artifact Filename | Type | Size | SHA-256 Digest |
|---|---|:---:|---|
| `high_quality_dragon_halong_512x512.png` | Feasibility Image | 427,898 B | `4173b54343c159bf2d1ef264a2e1f27de785be4b9212b71c5d17ee916ddbee38` |
| `factory_intro_speech.wav` | Feasibility Audio | 486,222 B | `e5d30cebf85a4094195b0b8ac716b9143447f1d70a7f31eaeb339f82018d5969` |
| `factory_intro_speech.mp3` | Feasibility Audio | 121,536 B | `84fbf0d050d1695cab8a2268e66e29493a9cdb163e793a87bdb516779791852a` |
| `cinematic_dragon_halong_motion.mp4` | Feasibility Video | 83,309 B | `021d16a6b593ef41ac24fcde263b008cff701423499b694df030e0d6ec3883e5` |
| `generated_text_response.txt` | Feasibility Text | 1,053 B | `61d4f8ad000a5400911d5c02333afe85e94202d6eda4ecad38afa0bc6baa948e` |
| `high_quality_report.json` | Run Telemetry | 2,906 B | `03aef4ad31ea4290ce344cb989c076d86d5d6d2a53ef3826e6486cb75a3eed2c` |
| `live_hardware_attested_report.json` | NVML Report | 4,156 B | `6734bcac365ffcf07ea02adf8b7d9aed8f3049aa760e327e3a32f5fe962f8f6d` |

---

## 6. Sharded Certification Roadmap (P0-02 Compliance)

Because the full 12-model suite requires $>80\text{ GB}$ of weights, monolithic execution will crash the Kaggle environment (<73 GB `/tmp`, <20 GB `/kaggle/working`). 

To achieve **Level 4 (Native Model Pass)** and **Level 5 (Factory E2E Pass)**, certification must proceed via 3 discrete shards:

```
                  ┌── Shard 1: Visual Pipeline (Qwen 3.8 VLM + Qwen-Image 2.1 + MiniMax H3)
                  │   └── Script: deployments/kaggle/session1_visual_pipeline.py
                  │
Factory v4 Lite ──┼── Shard 2: Audio & Music (ACE-Step 1.5 + YuE2 + Qwen-Speech)
                  │   └── Script: deployments/kaggle/session2_audio_pipeline.py
                  │
                  └── Shard 3: Video & Enhancement (Wan 2.2 + Enhancement Suite)
                      └── Script: deployments/kaggle/run_studio_quality_pipeline.py
```

---

## 7. Re-Evaluated Master Plan Issues (P0–P3)

| Issue ID | Severity | Description | Current Truthful Status | Action Required to Close |
|---|:---:|---|:---:|---|
| **P0-01** | **Critical** | Zero real GPU inference evidence | **IN PROGRESS** | Execute Golden Path E2E (API $\to$ Queue $\to$ Worker $\to$ Native Engine) using target model. |
| **P0-02** | **Critical** | Monolithic 80+ GB weight deployment exceeds Kaggle disk limit | **IN PROGRESS** | Sharding architecture and scripts established; individual shard runs must be executed and merged. |
| **P0-03** | **Critical** | Manifest capability claims vs runtime readiness mismatch | **IN PROGRESS** | Enforce 5-level hierarchy. Separate declared from feasibility smoke and native verified. |
| **P0-04** | **Critical** | Risk of conflating simulation/telemetry with real inference | **IN PROGRESS** | All overclaims retracted. Strictly fail-closed reporting enforced. |
| **P1-01–08**| **High** | Runtime control, multi-GPU lease, media streaming, upload limits | **IN PROGRESS** | Software implementations verified green locally (981 tests); awaiting live backend regression. |
| **P2-01–10**| **Medium** | Security hardening, trusted proxy, credentials, relative links | **IN PROGRESS** | Non-portable links fixed; credentials moved to environment/secrets; audit ongoing. |
| **P3-01–06**| **Low** | Cleanup, canonical docs, complexity budget | **IN PROGRESS** | Pinned Git SHAs, Next.js upgraded, documentation aligned with reality. |

---

## 8. Conclusion

- **Infrastructure Feasibility:** **PASS** (Dual Tesla T4 GPUs, drivers, libraries, and basic CUDA inference functional).
- **Native Model Certification:** **NOT ACHIEVED** (Alternative models and synthetic passes do not certify foundation models).
- **Factory E2E Certification:** **NOT ACHIEVED** (Control API $\to$ Worker $\to$ Native Engine flow not yet attested on GPU).

**Verdict: INFRASTRUCTURE FEASIBILITY ATTESTED — PRODUCTION CERTIFICATION IN PROGRESS.**
