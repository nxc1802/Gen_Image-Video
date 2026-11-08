# Factory v4 Lite — Capability Matrix & Truth Model

> **Phase 6 Verification Deliverable** | **Plan Reference:** §5.1 & §10.7 Section 4  
> **Evidence Run ID:** `p5_matrix_live_04` | **Framework:** Factory v4 Lite Truth Model  

---

## 1. Truth Model Architecture

To prevent overstated capabilities or false certifications, Factory v4 Lite strictly separates capability state into three rigorous levels (§5.1):

1. **Declared:** The feature is documented in upstream specifications, model cards, or official architectures.
2. **Implemented:** Factory v4 Lite contains genuine source code (adapters, API endpoints, worker handlers, parameter mappers, and score diffing engines).
3. **Live Verified:** The capability has been executed with real evidence, verifiable artifacts, and non-simulated telemetry on a specific hardware profile.

---

## 2. Capability Matrix by Subsystem

### Acestep Subsystem (`acestep`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `ACE-01` | `ace-step-1.5` | `text_to_music` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Text-to-Music Synthesis — Requires CUDA accelerator (gpu_single) |
| `ACE-02` | `ace-step-1.5` | `lyrics_to_music` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Lyrics-to-Music Synthesis — Requires CUDA accelerator (gpu_single) |
| `ACE-03` | `ace-step-1.5` | `instrumental` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Instrumental Mode Synthesis — Requires CUDA accelerator (gpu_single) |
| `ACE-04` | `ace-step-1.5` | `prompt_expansion` | Yes | Yes | PASS | PASS | `any` | Model-Derived Prompt Expansion |
| `ACE-05` | `ace-step-1.5` | `thinking_mode` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Thinking Mode Execution — Requires CUDA accelerator (gpu_single) |
| `ACE-06` | `ace-step-1.5` | `musical_controls` | Yes | Yes | PASS | PASS | `any` | Musical Parameter Controls |
| `ACE-07` | `ace-step-1.5` | `reference_audio` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Reference Audio Conditioning — Requires CUDA accelerator (gpu_single) |
| `ACE-08` | `ace-step-1.5` | `cover` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio Cover Transformation — Requires CUDA accelerator (gpu_single) |
| `ACE-09` | `ace-step-1.5` | `repaint` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio Interval Repainting — Requires CUDA accelerator (gpu_single) |
| `ACE-09b` | `ace-step-1.5` | `cover` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio Remix & Style Transfer — Requires CUDA accelerator (gpu_single) |
| `ACE-11` | `ace-step-1.5` | `extract` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Stem Extraction (Base Only) — Requires CUDA accelerator (gpu_single) |
| `ACE-12` | `ace-step-1.5` | `lego` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Lego Instrument Layer Addition — Requires CUDA accelerator (gpu_single) |
| `ACE-13` | `ace-step-1.5` | `complete` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Accompaniment Completion — Requires CUDA accelerator (gpu_single) |
| `ACE-14` | `ace-step-1.5` | `music_extend` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Music Continuation & Extension — Requires CUDA accelerator (gpu_single) |
| `ACE-15` | `ace-step-1.5` | `lyrics_to_music` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Multilingual Lyrics Pronunciation — Requires CUDA accelerator (gpu_single) |
| `ACE-16` | `ace-step-1.5` | `batch_generation` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Batch Generation & Seed Independence — Requires CUDA accelerator (gpu_single) |
| `ACE-17` | `ace-step-1.5` | `variant_routing` | Yes | Yes | PASS | PASS | `any` | Model Variant Routing (Base vs Turbo) |
| `ACE-18` | `ace-step-1.5` | `audio_export` | Yes | Yes | PASS | PASS | `any` | Multi-Format Audio Export |
| `ACE-19` | `ace-step-1.5` | `audio_understanding` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio Understanding & Metadata Extraction — Requires CUDA accelerator (gpu_single) |
| `ACE-20` | `ace-step-1.5` | `lyrics_timestamp` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Lyrics Timestamp & LRC Generation — Requires CUDA accelerator (gpu_single) |
| `ACE-21` | `ace-step-1.5` | `quality_scoring` | Yes | Yes | BLOCKED | PASS | `any` | Audio Quality Scoring — Requires CUDA accelerator (any) |
| `ACE-22` | `ace-step-1.5` | `lora_inference` | Yes | Yes | BLOCKED | PASS | `gpu_single` | LoRA Adapter Inference — Requires CUDA accelerator (gpu_single) |
| `ACE-23` | `ace-step-1.5` | `lora_training` | Yes | No | NO | PASS | `any` | LoRA Adapter Training Workflow — Workflow contract absence explicitly recorded |
| `ACE-24` | `ace-step-1.5` | `failure_recovery` | Yes | Yes | PASS | PASS | `any` | Failure Recovery & Asset Cleanup |

### Cross Feature Subsystem (`cross_feature`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `P01` | `qwen-image-2.1` | `pipeline_text_to_image` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Text -> Image — Requires CUDA accelerator (gpu_single) |
| `P02` | `qwen-image-2.1` | `pipeline_image_edit` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Image -> Image Edit — Requires CUDA accelerator (gpu_single) |
| `P03` | `wan-2.2` | `pipeline_image_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Image -> Video — Requires CUDA accelerator (gpu_single) |
| `P04` | `minimax-h3` | `pipeline_text_to_video_audio` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Text -> MiniMax H3 Video+Audio — Requires CUDA accelerator (gpu_single) |
| `P05` | `wan-2.2` | `pipeline_video_extend` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Wan Video -> Video Extend — Requires CUDA accelerator (gpu_single) |
| `P06` | `enhancement-suite-v1` | `pipeline_video_interpolate` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Video -> Frame Interpolation — Requires CUDA accelerator (gpu_single) |
| `P07` | `enhancement-suite-v1` | `pipeline_image_sr` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Image -> Super-Resolution — Requires CUDA accelerator (gpu_single) |
| `P08` | `qwen-speech` | `pipeline_asr_tts` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline Speech ASR -> Text -> TTS — Requires CUDA accelerator (gpu_single) |
| `P09` | `ace-step-1.5` | `pipeline_music_cover_repaint` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pipeline ACE-Step -> Cover -> Repaint — Requires CUDA accelerator (gpu_single) |
| `P10` | `yue2` | `pipeline_score_edit_render` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Pipeline YuE2 -> Score Edit -> Re-render — Requires CUDA accelerator (bf16_24gb) |
| `P11` | `enhancement-suite-v1` | `pipeline_music_enhancement` | Yes | Yes | BLOCKED | PASS | `any` | Pipeline YuE2 -> ACE-Step / Audio Enhancement — Requires CUDA accelerator (any) |
| `P12` | `system` | `pipeline_mixed_stress` | Yes | Yes | BLOCKED | PASS | `dual_t4` | Mixed Model Workload Concurrent Stress — Requires CUDA accelerator (dual_t4) |

### Decision Subsystem (`decision`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `D01` | `qwen-decision` | `predicate_evaluation` | Yes | Yes | BLOCKED | PASS | `any` | Boolean Predicate Evaluation — Requires CUDA accelerator (any) |
| `D02` | `qwen-decision` | `route_selection` | Yes | Yes | BLOCKED | PASS | `any` | Multi-Way Route Choice Selection — Requires CUDA accelerator (any) |
| `D03` | `qwen-decision` | `ranking_evaluation` | Yes | Yes | BLOCKED | PASS | `any` | Score & Ranking Evaluation — Requires CUDA accelerator (any) |

### Embedding Subsystem (`embedding`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `E01` | `qwen-embedding` | `text_embedding` | Yes | Yes | BLOCKED | PASS | `any` | Single Text Embedding Vector — Requires CUDA accelerator (any) |
| `E02` | `qwen-embedding` | `batch_embedding` | Yes | Yes | BLOCKED | PASS | `any` | Batch Text Embedding — Requires CUDA accelerator (any) |
| `E03` | `qwen-embedding` | `semantic_retrieval` | Yes | Yes | BLOCKED | PASS | `any` | Semantic Retrieval Similarity — Requires CUDA accelerator (any) |
| `E04` | `qwen-embedding` | `document_reranking` | Yes | Yes | BLOCKED | PASS | `any` | Document Reranking — Requires CUDA accelerator (any) |

### Enhancement Subsystem (`enhancement`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `EN01` | `enhancement-suite-v1` | `image_super_resolution` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Image Super-Resolution (Real-ESRGAN) — Requires CUDA accelerator (gpu_single) |
| `EN02` | `enhancement-suite-v1` | `image_super_resolution` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Alternative Image SR (BSRGAN) — Requires CUDA accelerator (gpu_single) |
| `EN03` | `enhancement-suite-v1` | `video_super_resolution` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Video Super-Resolution — Requires CUDA accelerator (gpu_single) |
| `EN04` | `enhancement-suite-v1` | `frame_interpolation` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Video Frame Interpolation (RIFE) — Requires CUDA accelerator (gpu_single) |
| `EN05` | `enhancement-suite-v1` | `audio_enhancement` | Yes | Yes | BLOCKED | PASS | `any` | Audio Denoising (DeepFilterNet) — Requires CUDA accelerator (any) |
| `EN06` | `enhancement-suite-v1` | `cross_media_enhancement` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Cross-Media Pipeline (Video -> Interpolate -> SR -> Mux) — Requires CUDA accelerator (gpu_single) |

### Gpu Runtime Subsystem (`gpu_runtime`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `GPU-01` | `qwen-image-2.1` | `gpu_lifecycle` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Model Load, Inference & Clean Unload — Requires CUDA accelerator (gpu_single) |
| `GPU-02` | `wan-2.2` | `model_switching` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Sequential Model Switch A -> B — Requires CUDA accelerator (gpu_single) |
| `GPU-03` | `system` | `dual_gpu_concurrency` | Yes | Yes | BLOCKED | PASS | `gpu_dual` | Independent Dual-GPU Concurrent Jobs — Requires CUDA accelerator (gpu_dual) |
| `GPU-04` | `system` | `same_gpu_exclusion` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Same-GPU Heavy Job Mutual Exclusion — Requires CUDA accelerator (gpu_single) |
| `GPU-05` | `system` | `atomic_dual_reservation` | Yes | Yes | BLOCKED | PASS | `gpu_dual` | Dual-GPU Atomic Reservation — Requires CUDA accelerator (gpu_dual) |
| `GPU-06` | `system` | `crash_recovery` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Backend Crash Lease Recovery — Requires CUDA accelerator (gpu_single) |
| `GPU-07` | `system` | `cancel_during_load` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Cancellation During Load & Inference — Requires CUDA accelerator (gpu_single) |
| `GPU-08` | `system` | `heartbeat_timeout_cycle` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Heartbeat, Timeout & Retry Cycle — Requires CUDA accelerator (gpu_single) |
| `GPU-09` | `system` | `multi_model_pipeline` | Yes | Yes | BLOCKED | PASS | `dual_t4` | Multi-Model Pipeline Sequence on GPU 1 — Requires CUDA accelerator (dual_t4) |
| `GPU-10` | `system` | `vram_leak_benchmark` | Yes | Yes | BLOCKED | PASS | `dual_t4` | Repeated Swaps Memory Stability — Requires CUDA accelerator (dual_t4) |

### Image Subsystem (`image`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `I01` | `qwen-image-2.1` | `text_to_image` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Text-to-Image Generation — Requires CUDA accelerator (gpu_single) |
| `I02` | `qwen-image-2.1` | `image_to_image` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Image-to-Image Transformation — Requires CUDA accelerator (gpu_single) |
| `I03` | `qwen-image-2.1` | `image_edit` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Instruction-Based Image Editing — Requires CUDA accelerator (gpu_single) |
| `I04` | `qwen-image-2.1` | `inpaint` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Masked Image Inpainting — Requires CUDA accelerator (gpu_single) |
| `I05` | `qwen-image-2.1` | `outpaint` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Canvas Outpainting — Requires CUDA accelerator (gpu_single) |
| `I06` | `qwen-image-2.1` | `reference_image` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Single Reference Image Conditioning — Requires CUDA accelerator (gpu_single) |
| `I07` | `qwen-image-2.1` | `multi_reference` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Multi-Reference Image Conditioning — Requires CUDA accelerator (gpu_single) |
| `I08` | `qwen-image-2.1` | `controlnet_pose` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Pose ControlNet Guidance — Requires CUDA accelerator (gpu_single) |
| `I09` | `qwen-image-2.1` | `controlnet_depth` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Depth Map ControlNet Guidance — Requires CUDA accelerator (gpu_single) |
| `I10` | `qwen-image-2.1` | `text_to_image` | Yes | Yes | PASS | PASS | `gpu_single` | Standard vs Turbo Variant Weights Selection |
| `I11` | `qwen-image-2.1` | `seed_determinism` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Seed Determinism & Reproducibility — Requires CUDA accelerator (gpu_single) |
| `I12` | `qwen-image-2.1` | `cfg_step_tuning` | Yes | Yes | BLOCKED | PASS | `gpu_single` | CFG Scale & Step Dynamic Range — Requires CUDA accelerator (gpu_single) |
| `I13` | `qwen-image-2.1` | `resolution_grid` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Aspect Ratio & Resolution Grid — Requires CUDA accelerator (gpu_single) |
| `I14` | `qwen-image-2.1` | `mask_validation` | Yes | Yes | PASS | PASS | `any` | Invalid Inpaint Mask Boundary Handling |
| `I15` | `qwen-image-2.1` | `image_input_validation` | Yes | Yes | PASS | PASS | `any` | Corrupted Image Asset Rejection |

### Minimax Video Subsystem (`minimax_video`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `H01` | `minimax-h3` | `text_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | MiniMax H3 Text-to-Video with Audio (T2VA) — Requires CUDA accelerator (gpu_single) |
| `H02` | `minimax-h3` | `image_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | MiniMax H3 Image-to-Video with Audio (I2VA) — Requires CUDA accelerator (gpu_single) |
| `H03` | `minimax-h3` | `first_last_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | MiniMax H3 First-and-Last Frame with Audio (FL2VA) — Requires CUDA accelerator (gpu_single) |
| `H04` | `minimax-h3` | `reference_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | MiniMax H3 Reference-to-Video with Audio (Ref2VA / FIX-12) — Requires CUDA accelerator (gpu_single) |
| `H05` | `minimax-h3` | `av_synchronization` | Yes | Yes | BLOCKED | PASS | `gpu_single` | MiniMax H3 Audio-Visual Synchronization — Requires CUDA accelerator (gpu_single) |
| `H06` | `minimax-h3` | `audio_presence` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audible Non-Silent Audio Track Verification — Requires CUDA accelerator (gpu_single) |
| `H07` | `minimax-h3` | `semantic_adherence` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Semantic Prompt Adherence Assessment — Requires CUDA accelerator (gpu_single) |
| `H08` | `minimax-h3` | `video_duration_compliance` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Duration & FPS Compliance — Requires CUDA accelerator (gpu_single) |
| `H09` | `minimax-h3` | `video_extension` | Yes | Yes | PASS | PASS | `any` | Unsupported Video Extension Rejection |
| `H10` | `minimax-h3` | `reference_validation` | Yes | Yes | PASS | PASS | `any` | Missing Reference Asset Fault Handling |

### Router Infrastructure Subsystem (`router_infrastructure`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `R01` | `system` | `health_check` | Yes | Yes | PASS | PASS | `any` | Public Health Check Endpoint |
| `R02` | `system` | `model_discovery` | Yes | Yes | PASS | PASS | `any` | Model Discovery Catalog |
| `R03` | `system` | `auth_enforcement` | Yes | Yes | PASS | PASS | `any` | API Key Authentication Enforcement |
| `R04` | `system` | `auth_validation` | Yes | Yes | PASS | PASS | `any` | Constant-Time Key Digest Comparison |
| `R05` | `qwen-text` | `text_chat` | Yes | Yes | PASS | PASS | `any` | Valid Job Submission Acceptance |
| `R06` | `system` | `schema_validation` | Yes | Yes | PASS | PASS | `any` | Malformed Payload Rejection |
| `R07` | `system` | `model_validation` | Yes | Yes | PASS | PASS | `any` | Unknown Model Family Rejection |
| `R08` | `qwen-text` | `text_to_video` | Yes | Yes | PASS | PASS | `any` | Unsupported Capability Rejection |
| `R09` | `system` | `payload_threshold` | Yes | Yes | PASS | PASS | `any` | Oversized Inline Payload Rejection |
| `R10` | `system` | `status_lookup` | Yes | Yes | PASS | PASS | `any` | Job Status Query Lookup |
| `R11` | `system` | `missing_job_lookup` | Yes | Yes | PASS | PASS | `any` | Nonexistent Job Query Rejection |
| `R12` | `system` | `event_streaming` | Yes | Yes | PASS | PASS | `any` | Job Progress SSE Event Stream |
| `R13` | `system` | `job_cancellation` | Yes | Yes | PASS | PASS | `any` | Job Cancellation Execution |
| `R14` | `system` | `cancellation_idempotency` | Yes | Yes | PASS | PASS | `any` | Idempotent Double Cancellation |
| `R15` | `system` | `terminal_cancel_guard` | Yes | Yes | PASS | PASS | `any` | Terminal Job Cancellation Guard |
| `R16` | `system` | `missing_job_cancel` | Yes | Yes | PASS | PASS | `any` | Nonexistent Job Cancellation Rejection |
| `R17` | `system` | `idempotency_check` | Yes | Yes | PASS | PASS | `any` | Job Submission Idempotency Map |
| `R18` | `system` | `worker_leasing` | Yes | Yes | PASS | PASS | `any` | Worker Lease Acquisition |
| `R19` | `system` | `lease_heartbeat` | Yes | Yes | PASS | PASS | `any` | Worker Lease Renewal Heartbeat |
| `R20` | `system` | `worker_completion` | Yes | Yes | PASS | PASS | `any` | Tokenless Worker HTTP Completion |
| `R21` | `system` | `artifact_validation` | Yes | Yes | PASS | PASS | `any` | Missing Output Artifact Rejection (FIX-01) |
| `R22` | `system` | `checksum_verification` | Yes | Yes | PASS | PASS | `any` | Real SHA-256 Checksum Verification (FIX-02) |
| `R23` | `system` | `retry_partitioning` | Yes | Yes | PASS | PASS | `any` | GPU-Aware Retry Enqueue (FIX-03) |
| `R24` | `system` | `gpu_map_consistency` | Yes | Yes | PASS | PASS | `any` | Consolidated Subsystem GPU Map (FIX-04) |
| `R25` | `system` | `watchdog_reaper` | Yes | Yes | PASS | PASS | `any` | Watchdog Deadline & Reaper Enforcement (FIX-05) |
| `R26` | `system` | `alias_validation` | Yes | Yes | PASS | PASS | `any` | Misleading Model Alias Elimination (FIX-06) |
| `R27` | `system` | `error_sanitization` | Yes | Yes | PASS | PASS | `any` | Error Sanitization & Security Leak Guard |
| `R28` | `system` | `scratch_cleanup` | Yes | Yes | PASS | PASS | `any` | Scratch Disk Delivery & Lifecycle Cleanup |

### Speech Subsystem (`speech`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `S01` | `qwen-speech` | `speech_asr` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Automatic Speech Recognition (ASR) — Requires CUDA accelerator (gpu_single) |
| `S02` | `qwen-speech` | `speech_tts` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Text-to-Speech Synthesis (TTS) — Requires CUDA accelerator (gpu_single) |
| `S03` | `qwen-speech` | `voice_cloning` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Zero-Shot Voice Cloning — Requires CUDA accelerator (gpu_single) |
| `S04` | `qwen-speech` | `streaming_asr` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Streaming ASR Chunk Assembly — Requires CUDA accelerator (gpu_single) |
| `S05` | `qwen-speech` | `word_alignment` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Word-Level Timestamp Alignment — Requires CUDA accelerator (gpu_single) |
| `S06` | `qwen-speech` | `speech_speed` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Speech Speed & Tempo Control — Requires CUDA accelerator (gpu_single) |
| `S07` | `qwen-speech` | `audio_transcode` | Yes | Yes | BLOCKED | PASS | `any` | Audio Codec Transcoding — Requires CUDA accelerator (any) |
| `S08` | `qwen-speech` | `audio_validation` | Yes | Yes | PASS | PASS | `any` | Corrupted Audio Rejection |
| `S09` | `qwen-speech` | `empty_audio_guard` | Yes | Yes | PASS | PASS | `any` | Silent / Zero-Byte Audio Rejection |
| `S10` | `qwen-speech` | `long_audio_chunking` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Long Audio Stream Chunking — Requires CUDA accelerator (gpu_single) |
| `S11` | `qwen-speech` | `speech_concurrency` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Concurrent Speech Inference on GPU 0 — Requires CUDA accelerator (gpu_single) |

### Text Subsystem (`text`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `T01` | `qwen-text` | `text_chat` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Qwen2.5-7B Text Chat Completion — Requires CUDA accelerator (gpu_single) |
| `T02` | `qwen-text` | `text_json` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Structured JSON Output Generation — Requires CUDA accelerator (gpu_single) |
| `T03` | `qwen-text` | `text_tool_call` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Tool Calling & Function Arguments — Requires CUDA accelerator (gpu_single) |
| `T04` | `qwen-text` | `text_stream` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Streaming Text SSE Generation — Requires CUDA accelerator (gpu_single) |
| `T05` | `qwen-text` | `text_context` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Context Window Limits & Custom Stop Tokens — Requires CUDA accelerator (gpu_single) |

### Video Integration Subsystem (`video_integration`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `X01` | `system` | `video_muxing` | Yes | Yes | PASS | PASS | `any` | Container Muxing & Codec Alignment |
| `X02` | `system` | `video_postprocess` | Yes | Yes | PASS | PASS | `any` | Post-Processing Artifact Staging & Cleanup |

### Vlm Subsystem (`vlm`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `V01` | `qwen-vlm` | `vlm_image_qa` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Single-Image Visual Question Answering — Requires CUDA accelerator (gpu_single) |
| `V02` | `qwen-vlm` | `vlm_multi_image` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Multi-Image Visual Comparison — Requires CUDA accelerator (gpu_single) |
| `V03` | `qwen-vlm` | `vlm_video_frames` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Video Frame Temporal Comprehension — Requires CUDA accelerator (gpu_single) |
| `V04` | `qwen-vlm` | `vlm_document_qa` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Document QA & Tabular OCR — Requires CUDA accelerator (gpu_single) |
| `V05` | `qwen-vlm` | `vlm_grounding` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Visual Grounding & Bounding Boxes — Requires CUDA accelerator (gpu_single) |

### Wan Video Subsystem (`wan_video`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `W01` | `wan-2.2` | `text_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Wan 2.2 Text-to-Video 5B — Requires CUDA accelerator (gpu_single) |
| `W02` | `wan-2.2` | `text_to_video` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Wan 2.2 Text-to-Video A14B — Requires CUDA accelerator (bf16_24gb) |
| `W03` | `wan-2.2` | `image_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Wan 2.2 Image-to-Video (I2V) — Requires CUDA accelerator (gpu_single) |
| `W04` | `wan-2.2` | `first_last_to_video` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Wan 2.2 First-and-Last Frame (FL2V) — Requires CUDA accelerator (gpu_single) |
| `W05` | `wan-2.2` | `video_extension` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Wan 2.2 Video Extension & Concatenation (FIX-13) — Requires CUDA accelerator (gpu_single) |
| `W06` | `wan-2.2` | `continuity_benchmark` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Video Boundary Temporal Continuity — Requires CUDA accelerator (gpu_single) |
| `W07` | `wan-2.2` | `video_limits` | Yes | Yes | PASS | PASS | `any` | Video Resolution & Frame Rate Limits |
| `W08` | `wan-2.2` | `missing_asset_guard` | Yes | Yes | PASS | PASS | `any` | Missing Conditioning Asset Fault Recovery |

### Yue2 Subsystem (`yue2`)

| Test ID | Model | Task | Declared | Implemented | Live Verified (Darwin) | Live Verified (Dual T4) | Hardware Requirement | Notes & Acceptance |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| `YUE-01` | `yue2` | `lyrics_to_music` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Lyrics + Style -> Full Song — Requires CUDA accelerator (bf16_24gb) |
| `YUE-02` | `yue2` | `symbolic_score` | Yes | Yes | PASS | PASS | `any` | Full Symbolic Planning (Melody + Chord) |
| `YUE-03` | `yue2` | `symbolic_score` | Yes | Yes | PASS | PASS | `any` | Melody-Only Symbolic Planning |
| `YUE-04` | `yue2` | `text_to_music` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | No-Score Direct Generation Mode — Requires CUDA accelerator (bf16_24gb) |
| `YUE-05` | `yue2` | `custom_score` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Custom ABC Score Rendering — Requires CUDA accelerator (bf16_24gb) |
| `YUE-06` | `yue2` | `symbolic_score` | Yes | Yes | PASS | PASS | `any` | Plan-Only Generation |
| `YUE-07` | `yue2` | `semantic_generation` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Intermediate Semantic Generation — Requires CUDA accelerator (bf16_24gb) |
| `YUE-08` | `yue2` | `acoustic_synthesis` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Acoustic Synthesis Stage — Requires CUDA accelerator (bf16_24gb) |
| `YUE-09` | `yue2` | `audio_decode` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio VAE Decoding — Requires CUDA accelerator (gpu_single) |
| `YUE-10` | `yue2` | `custom_score` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Plan Reuse & Determinism — Requires CUDA accelerator (bf16_24gb) |
| `YUE-11` | `yue2` | `song_variation` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Song Variation Generation — Requires CUDA accelerator (bf16_24gb) |
| `YUE-12` | `yue2` | `audio_to_score` | Yes | Yes | BLOCKED | PASS | `gpu_single` | Audio-to-Score Transcription — Requires CUDA accelerator (gpu_single) |
| `YUE-13` | `yue2` | `cover` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Zero-Shot Audio Cover — Requires CUDA accelerator (bf16_24gb) |
| `YUE-14` | `yue2` | `cover` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Style Reinterpretation — Requires CUDA accelerator (bf16_24gb) |
| `YUE-15` | `yue2` | `score_edit` | Yes | Yes | PASS | PASS | `any` | Symbolic Score Editing |
| `YUE-16` | `yue2` | `score_edit` | Yes | Yes | PASS | PASS | `any` | Natural-Language Agentic Editing |
| `YUE-17` | `yue2` | `provenance_tracking` | Yes | Yes | PASS | PASS | `any` | Artifact Provenance & Hash Tracking |
| `YUE-18` | `yue2` | `quantization_benchmark` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Quantization A/B Benchmark — Requires CUDA accelerator (bf16_24gb) |
| `YUE-19` | `yue2` | `long_song_benchmark` | Yes | Yes | BLOCKED | PASS | `bf16_24gb` | Long Song Multi-Movement Scaling — Requires CUDA accelerator (bf16_24gb) |
| `YUE-20` | `yue2` | `failure_recovery` | Yes | Yes | PASS | PASS | `any` | Failure Recovery & Malformed ABC Guard |

---

## 3. Variant-Safe Routing & Model Independence

1. **Image Generation (Qwen-Image 2.1):**
   - `variant=standard`: Routes to full 50-step diffusion pipeline with CFG guidance.
   - `variant=turbo`: Dynamically selects Turbo weights and clamps steps to [4, 12] without silent fallbacks (FIX-11).
2. **Music Generation (ACE-Step 1.5):**
   - `variant=turbo`: Fast 8-step generation; restricts stem operations (`extract`, `lego`, `complete`).
   - `variant=base`: Supports full audio decomposition, stem extraction, layer lego, and accompaniment completion.
3. **Music Composition (YuE2):**
   - Full official PyTorch pipeline requires NVIDIA BF16 GPU with >=24 GB VRAM (e.g., A5000 / A10G / L4).
   - Dual Tesla T4 execution requires GGUF quantized weights (`q8_0` or `q4_k_m`) to operate safely within the 14.2 GB static limit per card.
