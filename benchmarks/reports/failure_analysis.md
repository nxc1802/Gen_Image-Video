# Factory v4 Lite — Failure Analysis & Historical Fix Audit

> **Phase 6 Verification Deliverable** | **Plan Reference:** §9.4, §10.5 & §10.7 Section 9  
> **Scope:** Transparent root-cause failure taxonomy, hardware blockers, and complete historical fix audit  

---

## 1. Transparent Root-Cause Taxonomy

Of the 163 evaluated cases in the Phase 5 benchmark matrix, **zero unhandled errors (`0 ERROR`) and zero logic failures (`0 FAIL`) occurred**.
A total of **112 non-PASS cases** were cleanly and transparently classified into two distinct operational categories:

| Failure Category | Status Code | Count | Root Cause Summary | Remediation & Environment |
|---|---|:---:|---|---|
| **Hardware Accelerators Required** | `HARDWARE_BLOCKED` | **111** | Host platform `Darwin arm64` lacks discrete NVIDIA CUDA accelerators | Deploy to Kaggle Dual Tesla T4 kernel with CUDA 13.0 |
| **Contractual Feature Absence** | `NOT_IMPLEMENTED` | **1** | ACE-23 LoRA fine-tuning training workflow contract | Documented optional training pipeline per Plan §11 |

---

## 2. Hardware Blocker Breakdown by Subsystem

The 111 `HARDWARE_BLOCKED` cases are distributed across neural model execution paths:

| Subsystem Group | Blocked Cases | Affected Test IDs | Required Hardware Profile | Root Cause |
|---|:---:|---|---|---|
| `gpu_runtime` | 10 | GPU-01 to GPU-10 | Kaggle Dual Tesla T4 (`gpu_dual` / `gpu_single`) | CUDA driver and nvidia-smi unavailable on Darwin arm64 |
| `text` | 5 | T01 to T05 | NVIDIA CUDA (`gpu_single`) | Qwen-3.8B neural forward pass requires CUDA GPU |
| `vlm` | 5 | V01 to V05 | NVIDIA CUDA (`gpu_single`) | Qwen-VLM vision transformer forward pass requires CUDA GPU |
| `decision` | 3 | D01 to D03 | NVIDIA CUDA (`gpu_single`) | Decision classifier forward pass requires CUDA GPU |
| `embedding` | 4 | E01 to E04 | NVIDIA CUDA (`gpu_single`) | Vector embedding generation requires CUDA GPU |
| `speech` | 9 | S01-S07, S10-S11 | NVIDIA CUDA (`gpu_single`) | ASR Whisper & TTS CosyVoice neural infer requires CUDA GPU |
| `image` | 12 | I01-I09, I11-I13 | NVIDIA CUDA (`gpu_single`) | Qwen-Image 2.1 Standard & Turbo diffusion requires CUDA GPU |
| `wan_video` | 6 | W01 to W06 | NVIDIA CUDA (`gpu_single`) | Wan 2.2 diffusion video generation requires CUDA GPU |
| `minimax_video` | 8 | H01 to H08 | NVIDIA CUDA (`gpu_single`) | MiniMax H3 video generation requires CUDA GPU |
| `acestep` | 18 | ACE-01-03, 05, 07-16, 19-22 | NVIDIA CUDA (`gpu_single`) | ACE-Step 1.5 music diffusion inference requires CUDA GPU |
| `yue2` | 13 | YUE-01, 04-05, 07-14, 18-19 | NVIDIA CUDA (`gpu_single` / `gpu_dual`) | YuE2 autoregressive audio codec generation requires CUDA GPU |
| `enhancement` | 6 | EN01 to EN06 | NVIDIA CUDA (`gpu_single`) | Real-ESRGAN & RIFE frame interpolation requires CUDA GPU |
| `cross_feature` | 12 | P01 to P12 | Kaggle Dual Tesla T4 (`gpu_dual`) | Multi-model cross-pipeline chaining requires dual CUDA GPUs |

---

## 3. Unimplemented Feature Analysis: ACE-23

- **Test ID:** `ACE-23` (`LoRA Fine-Tuning Execution`)
- **Status:** `NOT_IMPLEMENTED`
- **Root Cause:** In strict compliance with Plan §11, optional training and fine-tuning workflows are transparently classified as `NOT_IMPLEMENTED` rather than masquerading as passed inference capabilities or failing basic music generation.
- **Impact:** Zero impact on inference, cover, audio repaint, stem extraction, or score editing workflows.

---

## 4. Comprehensive Historical Fix Audit Log

### Phase 0 Baseline Integrity Fixes (FIX-01 to FIX-20)

| Fix ID | Priority | Subsystem / File | Original Defect / Integrity Hazard | Implemented Remediation | Regression Verification |
|---|:---:|---|---|---|---|
| **FIX-01** | Critical | `queue/manager.py` | Media jobs marked SUCCEEDED with missing physical output files | Enforced physical file validation; missing files trigger `JobState.FAILED` with `ARTIFACT_MISSING` | `test_p0_queue_watchdog_fixes.py::test_fix_01` |
| **FIX-02** | Critical | `queue/manager.py` | Defaulted missing file hashes to all-zero string `"0"*64` | Removed fake zero hashes; computes genuine streaming SHA-256 | `test_p0_queue_watchdog_fixes.py::test_fix_02` |
| **FIX-03** | Critical | `queue/manager.py` | Retries bypassed GPU affinity and could swap GPU 0 / GPU 1 | Partition affinity resolved via `get_subsystem_gpu` and enqueued to correct broker | `test_p0_queue_watchdog_fixes.py::test_fix_03` |
| **FIX-04** | Critical | `queue/policy.py` | Duplicated `SUBSYSTEM_GPU_MAP` across 4 files caused divergence | Consolidated into single authoritative policy in `queue/policy.py` | `test_p0_queue_watchdog_fixes.py::test_fix_04` |
| **FIX-05** | Critical | `queue/watchdog.py` | Rigid 180s timeout prematurely killed slow video/music jobs | Implemented 4-tier timeout hierarchy (task, model, service, default) + heartbeat grace | `test_p0_queue_watchdog_fixes.py::test_fix_05` |
| **FIX-06** | Critical | `api/routers/jobs.py` | Misleading aliases allowed (e.g. 72B mapping to 7B weights) | Purged aliases; strict 1:1 validation between requested model and physical checkpoint | `test_p0_queue_watchdog_fixes.py::test_fix_06` |
| **FIX-07** | Critical | `api/routers/jobs.py` | Unchecked job submission caused impossible jobs to hang in queue | Gated admission against model manifests, hardware profile, and task support | `test_p0_queue_watchdog_fixes.py::test_fix_07` |
| **FIX-08** | Critical | `services/speech/server.py` | Audio URLs not ingested to disk before calling ASR engine | Ingestion helper downloads remote audio to scratch disk before ASR processing | `test_p0_speech_music_fixes.py::test_fix_08` |
| **FIX-09** | Critical | `services/music/launcher` | Legacy gateway launched instead of native ACE-Step API | Launch scripts updated to start native ACE-Step engine with `/health` checks | `test_p0_speech_music_fixes.py::test_fix_09` |
| **FIX-10** | Critical | `acestep/client.py` | Payload field mismatches with native ACE-Step API specifications | Realigned payload schema to match native ACE-Step endpoints | `test_p0_speech_music_fixes.py::test_fix_10` |
| **FIX-11** | Critical | `workflows/image` | Variant parameter ignored; always loaded standard checkpoint | Turbo variant explicitly selects Turbo checkpoint weights | `test_phase2_subsystems.py::test_image_subsystem` |
| **FIX-12** | Critical | `workflows/video` | MiniMax H3 reference input not bound in `8_ref` workflow node | Mapped reference image asset path into node `8_ref` | `test_phase2_subsystems.py::test_video_subsystem` |
| **FIX-13** | Critical | `services/video/worker.py` | Video extension inverted segment order or corrupted audio | Stitching helper enforces chronological segment order and audio crossfading | `test_video_extension_helper.py` |
| **FIX-14** | Critical | `services/enhancement` | `/health` returned `infer_ready: True` without backends online | `/health` checks actual model backend availability | `test_p0_fixes_14_to_20.py::test_fix_14` |
| **FIX-15** | Critical | `tests/e2e/conftest.py` | Mock tests mixed with live suites, risking false certifications | Separated `contract_mock_client` and `live_client` with unambiguous markers | `test_p0_fixes_14_to_20.py::test_fix_15` |
| **FIX-16** | Critical | `deployments/kaggle` | Synthetic tensor benchmarks misclassified as LIVE_PASS | Reclassified synthetic benchmarks to `SYNTHETIC_PASS` (`live_pass: false`) | `test_p0_fixes_14_to_20.py::test_fix_16` |
| **FIX-17** | High | `factory_contracts/errors` | Generic `SERVICE_UNAVAILABLE` errors masked root causes | Introduced actionable error taxonomy (`classification`, `hardware_requirement`) | `test_p0_fixes_14_to_20.py::test_fix_17` |
| **FIX-18** | High | `factory/apps/studio-web` | Demonstration metrics unlabeled in UI | Added visual markers distinguishing demonstration waveforms from live telemetry | `test_p0_fixes_14_to_20.py::test_fix_18` |
| **FIX-19** | High | Project dependencies | Outdated dependencies and security vulnerabilities | Updated vulnerable dependencies; resolved CI audit blocks | `test_p0_fixes_14_to_20.py::test_fix_19` |
| **FIX-20** | High | Deploy configs & docs | Stale launch paths and dead Redis/S3 assumptions in docs | Cleaned documentation and consolidated single-path deployment scripts | `test_p0_fixes_14_to_20.py::test_fix_20` |

### Phase 3 ACE-Step Remediation (VULN-P3-01 to VULN-P3-05)

| Vulnerability ID | Severity | Component | Issue Description | Applied Remediation | Test Verification |
|---|:---:|---|---|---|---|
| **VULN-P3-01** | Critical | `MusicWorker.stage_asset` | Local `file://` asset unlinking deleted user's original source file | Created secure temporary copy in scratch disk; unlinks only staged copy | `test_adversarial_p3_acestep.py::test_file_uri_deletion_hazard_probe` |
| **VULN-P3-02** | High | `MusicWorker.stage_asset` | SSRF vulnerability allowed fetching cloud metadata (169.254.169.254) | Added SSRF pre-flight check blocking loopback, link-local, and metadata IPs | `test_adversarial_p3_acestep.py::test_ssrf_preflight_validation_probe` |
| **VULN-P3-03** | Medium | `ACEStepAdapter` | IEEE 754 NaN float values bypassed range checks and poisoned params | Added `math.isnan()` guards rejecting NaN with `InvalidRequestError` | `test_adversarial_p3_acestep.py::test_nan_numerical_poisoning_bypass_probe` |
| **VULN-P3-04** | Medium | `ACEStepClient` | Unhandled raw `httpx.ConnectError` leaked internal stack traces | Wrapped in `ModelUnavailableError` with sanitized error message | `test_adversarial_p3_acestep.py::test_understand_and_score_leak_raw_httpx_exceptions_probe` |
| **VULN-P3-05** | Low | `ACEStepAdapter` | Musical key regex accepted embedded newlines (e.g. `'C\nmajor'`) | Strict horizontal whitespace `[ \t]*` regex enforcement | `test_adversarial_p3_acestep.py::test_musical_key_embedded_newline_gap_probe` |

### Phase 4 YuE2 Hardening (VULN-P4-01 to VULN-P4-04)

| Vulnerability ID | Severity | Component | Issue Description | Applied Remediation | Test Verification |
|---|:---:|---|---|---|---|
| **VULN-P4-01** | Medium | `yue2/score.py` | Space in key header (e.g. `'K:C major'`) was falsely rejected | Updated `KEY_REGEX` to permit standard ABC 2.1 whitespace between note and mode | `test_adversarial_p4_yue2.py::test_vuln_p4_01_space_in_key_header_falsely_rejected_probe` |
| **VULN-P4-02** | Low/Med | `yue2/score.py` | Zero tempo (`Q:0`) was falsely accepted as valid tempo | Added validation strictly requiring tempo > 0 BPM | `test_adversarial_p4_yue2.py::test_vuln_p4_02_zero_tempo_falsely_accepted_probe` |
| **VULN-P4-03** | Low | `yue2/score.py` | Altered/added chords (`Cadd9`, `Dm7b5`, `G7#9`) were silently dropped | Expanded `CHORD_REGEX` to capture extension degrees and alterations | `test_adversarial_p4_yue2.py::test_vuln_p4_03_altered_and_added_chords_silently_dropped_probe` |
| **VULN-P4-04** | Low | `yue2/score.py` | Inline comment quote (`% note: 'fast`) triggered false syntax error | Comment stripping applied prior to delimiter balance verification | `test_adversarial_p4_yue2.py::test_vuln_p4_04_inline_comment_quote_triggers_false_unclosed_error_probe` |
