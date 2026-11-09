# Factory v4 Lite — Unified Benchmark Summary Report

**Run ID**: `run_20261109_173256`  
**Git Commit**: `6fbfd6bfc8d60d42357876884ecc15d2d4ec4fea`  
**Execution Mode**: `smoke`  
**Timestamp**: `2026-11-09 10:33:08 UTC`  
**Hardware Environment**: 0 GPU(s) — `None (CPU Mode)`  

---

## 1. Executive Summary

- **Total Evaluated Cases**: 91
- **Model Inference Cases**: 82
- **Model Inference Passing Cases**: 0 (0.0%)
- **Negative API Contract Cases**: 9
- **Negative API Contract Passing Cases**: 9 (100.0%)
- **Hardware Blocked**: 0
- **Backend Unavailable**: 82
- **Failed / Error**: 0

### Status Breakdown

| Status | Count | Percentage | Description |
|---|---|---|---|
| `INFERENCE_PASS` | 0 | 0.0% | Authentic model inference succeeded, output artifact validated with real SHA-256 and MIME header. |
| `CONTRACT_PASS` | 9 | 9.9% | Negative / fail-closed API contract test cleanly rejected with expected HTTP status and error code. |
| `PASS` | 0 | 0.0% | Model inference pass count (strictly positive cases). |
| `FAIL` | 0 | 0.0% | Execution finished but artifact or contract validation failed. |
| `ERROR` | 0 | 0.0% | Unexpected runtime exception or system fault. |
| `HARDWARE_BLOCKED` | 0 | 0.0% | Exceeds discrete Dual-T4 14.2 GB VRAM per-card ceiling or non-GPU host. |
| `BACKEND_UNAVAILABLE` | 82 | 90.1% | Model engine/weights uninstalled or unbacked (e.g. Enhancement Suite). |
| `NOT_IMPLEMENTED` | 0 | 0.0% | Capability declared unbacked in model manifest. |
| `NOT_RUN` | 0 | 0.0% | Case skipped or not executed. |

---

## 2. Model Manifest Capability Matrix (11 Foundation Models)

| Model | Total Cases | Inf Total | Inf PASS | Contract PASS | BLOCKED | BACKEND UNAVAIL | FAIL / ERR | Median Latency (ms) | Peak VRAM GPU 0 (MB) | Peak VRAM GPU 1 (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| `ace-step-1.5` | 35 | 33 | 0 | 2 | 0 | 33 | 0 | - | - | - |
| `enhancement-suite-v1` | 4 | 4 | 0 | 0 | 0 | 4 | 0 | - | - | - |
| `minimax-h3` | 6 | 5 | 0 | 1 | 0 | 5 | 0 | - | - | - |
| `qwen-decision` | 3 | 3 | 0 | 0 | 0 | 3 | 0 | - | - | - |
| `qwen-embedding` | 3 | 2 | 0 | 1 | 0 | 2 | 0 | - | - | - |
| `qwen-image-2.1` | 13 | 12 | 0 | 1 | 0 | 12 | 0 | - | - | - |
| `qwen-speech` | 4 | 3 | 0 | 1 | 0 | 3 | 0 | - | - | - |
| `qwen-text` | 6 | 5 | 0 | 1 | 0 | 5 | 0 | - | - | - |
| `qwen-vlm` | 5 | 5 | 0 | 0 | 0 | 5 | 0 | - | - | - |
| `wan-2.2` | 5 | 4 | 0 | 1 | 0 | 4 | 0 | - | - | - |
| `yue2` | 7 | 6 | 0 | 1 | 0 | 6 | 0 | - | - | - |

---

## 3. Dual-T4 Discrete Memory Limit Compliance

The Kaggle Dual Tesla T4 hardware architecture provides two discrete 16 GB accelerators without a unified virtual address space:
- **GPU 0 Partition**: Text, VLM, Decision, Image, Speech (Max Allocatable: **14.2 GB / 14,540 MB**).
- **GPU 1 Partition**: Embedding, Video, Music, Enhancement (Max Allocatable: **14.2 GB / 14,540 MB**).
- **Invariance Contract**: Zero shared 32 GB address pool assumption. Workloads requiring > 14.2 GB on a single accelerator are strictly classified as `HARDWARE_BLOCKED` rather than producing synthetic passes.

---

## 4. Multi-Modal Artifact Verification Matrix

Every passing test case has been cryptographically attested with non-empty byte verification, container magic headers (PNG IHDR, WAV RIFF, MP4 ftyp), and SHA-256 hashes:

| Case ID | Model | Task | Variant | Status | MIME | Output Bytes | SHA-256 (Prefix) |
|---|---|---|---|---|---|---|---|
| `BM-TXT-01` | `qwen-text` | `chat_completion` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-TXT-02` | `qwen-text` | `responses_api` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-TXT-03` | `qwen-text` | `structured_output` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-TXT-04` | `qwen-text` | `tool_calling` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-TXT-05` | `qwen-text` | `streaming` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-VLM-01` | `qwen-vlm` | `image_chat` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-VLM-02` | `qwen-vlm` | `responses_api` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-VLM-03` | `qwen-vlm` | `video_chat` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-VLM-04` | `qwen-vlm` | `visual_grounding` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-VLM-05` | `qwen-vlm` | `document_qa` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-DEC-01` | `qwen-decision` | `predicate_evaluation` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-DEC-02` | `qwen-decision` | `choice_routing` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-DEC-03` | `qwen-decision` | `score_ranking` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-EMB-01` | `qwen-embedding` | `text_embedding` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-EMB-02` | `qwen-embedding` | `reranking` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-SPH-01` | `qwen-speech` | `speech_to_text` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-SPH-02` | `qwen-speech` | `text_to_speech` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-SPH-03` | `qwen-speech` | `voice_cloning` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-01` | `qwen-image-2.1` | `text_to_image` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-02` | `qwen-image-2.1` | `text_to_image` | `turbo` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-03` | `qwen-image-2.1` | `image_to_image` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-04` | `qwen-image-2.1` | `image_to_image` | `turbo` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-05` | `qwen-image-2.1` | `inpaint` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-06` | `qwen-image-2.1` | `outpaint` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-07` | `qwen-image-2.1` | `instruction_edit` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-08` | `qwen-image-2.1` | `reference_image` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-09` | `qwen-image-2.1` | `multi_reference` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-10` | `qwen-image-2.1` | `controlnet` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-11` | `qwen-image-2.1` | `control_pose` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-IMG-12` | `qwen-image-2.1` | `control_depth` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-WAN-01` | `wan-2.2` | `text_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-WAN-02` | `wan-2.2` | `image_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-WAN-03` | `wan-2.2` | `first_last_frame_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-WAN-04` | `wan-2.2` | `video_extension` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-MMX-01` | `minimax-h3` | `text_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-MMX-02` | `minimax-h3` | `image_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-MMX-03` | `minimax-h3` | `first_last_frame_to_video` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-MMX-04` | `minimax-h3` | `video_with_audio` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-MMX-05` | `minimax-h3` | `reference_conditioning` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| `BM-ACE-01` | `ace-step-1.5` | `text_to_music` | `standard` | `BACKEND_UNAVAILABLE` | `-` | 0 | `-` |
| ... | ... | ... | ... | ... | ... | ... | *(See metrics.csv for all 91 cases)* |

---

## 5. Integrity & Verification Attestation

In accordance with the Factory v4 Lite Integrity Mandate:
1. **Zero Simulation-as-Certification**: No synthetic tokens, mock audio waveforms, or dummy video frames were counted as passing live inferences.
2. **Strict Status Semantics**: Unavailable engines (Enhancement Suite) and hardware-infeasible workloads are explicitly marked `BACKEND_UNAVAILABLE` or `HARDWARE_BLOCKED`.
3. **Cryptographic Provenance**: Every output artifact is validated against binary magic bytes and hashed with real SHA-256 digests.

*Generated by Unified Benchmark Report Generator v2.0*
