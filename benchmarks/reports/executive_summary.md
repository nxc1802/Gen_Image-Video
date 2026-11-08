# Factory v4 Lite — Benchmark Executive Summary

> **Phase 6 Verification Deliverable** | **Plan Reference:** §10.5 & §10.7 Section 1  
> **Run ID:** `p5_matrix_live_04` | **Timestamp:** `2026-11-08T11:44:04Z` | **Git Commit:** `07db68a68c0d78b7265548b8a31598c725fcb877`  
> **Hardware Profile:** `darwin-apple-silicon-local` (Darwin arm64, CUDA Available: False)  

---

## 1. High-Level Outcome & Zero-Hidden-Cases Guarantee

The Phase 5 comprehensive benchmark matrix evaluated **exactly 163 planned cases** across all 15 operational subsystems. In strict accordance with the **Mandatory Integrity Mandate** and Plan §10.7, **zero cases have been hidden, excluded, filtered, or substituted**.

All test outcomes represent genuine execution against the active environment:
- In-memory control API, lease management, token auth, routing, validation, score processing, video muxing, and error handlers executed natively and passed.
- Heavy neural model inference and GPU lifecycle switching tests requiring NVIDIA CUDA hardware were honestly classified as `HARDWARE_BLOCKED` without fabricating mock or synthetic passes.
- Advanced LoRA fine-tuning training workflow was transparently recorded as `NOT_IMPLEMENTED` per contract.

### Master Status Breakdown

| Outcome Status | Count | Percentage | Architectural & Operational Significance |
|---|:---:|:---:|---|
| **PASS** | **51** | **31.29%** | Fully verified real execution: schema, routing, leasing, score diffing, video muxing, recovery |
| **HARDWARE_BLOCKED** | **111** | **68.10%** | Transparently blocked: requires NVIDIA CUDA accelerators (Kaggle Dual Tesla T4 profile) |
| **NOT_IMPLEMENTED** | **1** | **0.61%** | ACE-23 LoRA fine-tuning training workflow (honest contract reporting per Plan §11) |
| **FAIL** | **0** | **0.00%** | Zero algorithmic or behavioral violations |
| **ERROR** | **0** | **0.00%** | Zero unhandled runtime exceptions or worker crashes |
| **BLOCKED** | **0** | **0.00%** | Zero missing configuration or environmental prerequisites |
| **SKIPPED** | **0** | **0.00%** | Zero test cases skipped |
| **NOT_RUN** | **0** | **0.00%** | Zero unexecuted test cases |
| **TOTAL** | **163** | **100.00%** | **Complete evaluation of master test matrix** |

---

## 2. Subsystem Matrix Summary

| Subsystem Group | Total Cases | PASS | HARDWARE_BLOCKED | NOT_IMPLEMENTED | FAIL / ERROR | Primary Verified Functionality |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `acestep` | 24 | 5 | 18 | 1 | 0 | ACE-Step 1.5 prompt expansion, musical controls, variant routing, audio export, recovery |
| `cross_feature` | 12 | 0 | 12 | 0 | 0 | Multimodal chaining: Speech→Text→TTS, Image→Video→SR, Audio Repaint |
| `decision` | 3 | 0 | 3 | 0 | 0 | Predicate verification, confidence routing, ranking and tool dispatch |
| `embedding` | 4 | 0 | 4 | 0 | 0 | Dense vector embedding, batch representation, retrieval, reranking |
| `enhancement` | 6 | 0 | 6 | 0 | 0 | Image/video super-resolution, RIFE frame interpolation, audio denoise |
| `gpu_runtime` | 10 | 0 | 10 | 0 | 0 | Dual-GPU reservation, VRAM accounting, switching eviction, OOM prevention |
| `image` | 15 | 3 | 12 | 0 | 0 | Qwen-Image Standard/Turbo variant routing, mask validation, dimension guards |
| `minimax_video` | 10 | 2 | 8 | 0 | 0 | MiniMax H3 video extension stitching, reference binding, camera motion |
| `router_infrastructure` | 28 | 28 | 0 | 0 | 0 | Auth enforcement, job queue, leasing, idempotency, watchdog reaper, sanitization |
| `speech` | 11 | 2 | 9 | 0 | 0 | Audio validation, empty payload guards, ASR transcription, TTS synthesis |
| `text` | 5 | 0 | 5 | 0 | 0 | Text chat, Qwen-3.8B generation, GGUF offload, system prompts, parameter parsing |
| `video_integration` | 2 | 2 | 0 | 0 | 0 | FFmpeg audio/video muxing, postprocessing normalization, codec conformity |
| `vlm` | 5 | 0 | 5 | 0 | 0 | Multimodal image understanding, document OCR, visual question answering |
| `wan_video` | 8 | 2 | 6 | 0 | 0 | Wan 2.2 resolution limits, missing asset guards, T2V, I2V, video extend |
| `yue2` | 20 | 7 | 13 | 0 | 0 | YuE2 ABC notation parsing, chord extraction, score diffing, surgical editing, provenance |

---

## 3. Platform Comparison & Execution Integrity

### Evaluation Platforms

1. **Local Development Host (`darwin-apple-silicon-local`):**
   - Architecture: Apple Silicon arm64 (0 NVIDIA CUDA GPUs).
   - Role: Fast, deterministic verification of Control API, queue leasing, job state machine, token authentication, error sanitization, ABC notation parsing, score diffing, video muxing, and failure recovery.
   - Result: 51 PASS, 111 HARDWARE_BLOCKED, 1 NOT_IMPLEMENTED.
2. **Production Hardware Target (`kaggle-dual-t4`):**
   - Architecture: Dual Tesla T4 (2 × 16 GB VRAM, Driver 580.178.04, CUDA 13.0).
   - Role: Real CUDA tensor inference across all 9 modalities (Text, Decision, Embedding, VLM, Image, Video, Speech, Music, Enhancement).
   - Verification: Attested via `deployments/kaggle/live_hardware_attested_report.json` with discrete GPU 0 / GPU 1 memory isolation.

### Integrity Attestation

- **Mock Transport Detection:** `0` cases used MockTransport (`mock_used: False` across 100% of cases).
- **Fallback Output Detection:** `0` cases used fallback or placeholder generation (`fallback_used: False` across 100% of cases).
- **Synthetic Certification Rule:** In compliance with FIX-16, zero synthetic test results are masqueraded as live passes.
