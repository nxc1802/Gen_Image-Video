# Factory v4 Lite — Full Implementation & Benchmark Plan

> **Repository:** [nxc1802/Gen_Image-Video](https://github.com/nxc1802/Gen_Image-Video)  
> **Scope:** Bug fixing → model/runtime readiness → full capability implementation → ACE-Step 1.5 + YuE2 → live GPU verification → reproducible benchmark/reporting  
> **Hardware baseline:** Kaggle Dual Tesla T4 (2 × 16 GB VRAM); additional GPU profiles where required  
> **Document type:** Implementation plan, **not** a claim that the work or live tests have been completed  
> **Prepared:** 2026-11-07

---

## 1. Objectives and non-negotiable principles

The final Factory must let a user select **model + task + variant**, safely allocate GPU resources, execute **real model inference**, persist **verifiable artifacts**, and report reliable results per test case, checkpoint, GPU, and run.

**Definition of completion:**

1. **Correctness:** API, task routing, queue, storage, and worker execution behave correctly; no fabricated success.
2. **Capability completeness:** Every feature advertised as supported has an implemented and live-verified execution path, or is explicitly marked unsupported/unverified.
3. **Runtime reliability:** Model loading, unloading, GPU allocation, cancellation, retries, and worker recovery are safe and observable.
4. **Verified quality:** Each PASS has reproducible evidence: native checkpoint, inputs/config, artifacts, timings, peak VRAM, checksums, and relevant quality metrics.

**Implementation principles:**

- **Do not rewrite Factory.** Keep Control API, jobs, workers, manifests, ComfyUI/native backends, storage, and Studio Web.
- **Do not optimize before validating a minimal real execution path.** Correctness and end-to-end proof first; acceleration and SOTA comparisons later.
- **No simulation-as-certification.** MockTransport, random tensors, sine-wave audio, synthesized MP4 placeholders, and fallback output never count as live inference PASS.
- **No overstated capability.** Separate upstream/native support, Factory code implementation, and live verification per hardware profile.
- **No unjustified dependencies.** Avoid adding Redis, Kubernetes, distributed orchestration, multi-agent editing, or sophisticated LRU unless a measured requirement demands them.
- **No hardware assumptions.** Dual T4 is not equivalent to one 32 GB GPU; multi-GPU execution requires actual backend support and end-to-end testing.

### 1.1 Music model strategy

| Model | Intended role | Distinguishing strengths | Qualification |
|---|---|---|---|
| **ACE-Step 1.5** | Fast generation, lyrics, source-audio workflows | Cover/repaint/reference and Base-specific extract/lego/complete depending on checkpoint | Native capabilities vary by model variant; verify independently |
| **YuE2** | High-quality full-song and symbolic composition pipeline | Lyrics/style generation, intermediate melody/chord score, score-based control, cover/edit | Competitive open model, but not universally proven SOTA on every metric or hardware |

**YuE2 feasibility gate:** The official YuE2 PyTorch environment may require a BF16-capable GPU with approximately 24 GB VRAM. First verify native execution on a suitable GPU, then separately investigate quantized/GGUF operation and feature parity on Dual T4. If T4 support is not established, report `HARDWARE_BLOCKED`; do **not** fabricate support or stall other workstreams.

**Licensing gate:** Record the exact license and permitted use of YuE2 weights and every dependent model/asset before production deployment, particularly for commercial use. ACE-Step weights/dependencies also require their own license review.

---

## 2. Target architecture

```text
Studio Web / REST API
        |
        v
Control API + Capability Registry
  - model / task / variant / parameter validation
  - declared vs implemented vs live-verified capability
  - backend/weights/hardware readiness checks
        |
        v
Queue + ModelRuntimeManager
  - job priority / leases / cancellation / retries
  - GPU reservation and memory planning
  - load / ready / busy / unload / recovery
        |
        +---------------------+
        |                     |
        v                     v
GPU 0 preferred           GPU 1 preferred
Text / VLM / Decision     Video / Music
Image / Speech            Embedding / Enhancement
        |                     |
        +----------+----------+
                   |
                   v
             Real model backends
      (ComfyUI, vLLM, native music,
       ASR/TTS, enhancement engines)
                   |
                   v
Artifacts + Telemetry + Benchmark Reports
```

GPU-to-service assignments are **preferences**, not static exclusivity rules. A heavy runtime can take an idle GPU outside its preferred group when feasible. Jobs requiring both GPUs must reserve both atomically. Only a verified runtime can report `READY`.

### 2.1 Model runtime state machine

`UNLOADED → LOADING → READY → BUSY → READY → UNLOADING → UNLOADED`

Minimum runtime metadata:

- `model_id`, `variant_id`, `checkpoint_id`, `checkpoint_revision`, `backend_type`, `backend_version`.
- `gpu_ids`, estimated/reserved VRAM per GPU, actual peak VRAM, process ID, lock/lease ID.
- `last_used_at`, current state, health, previous failure and readiness timestamp.
- Dependency readiness: VAE, text encoder, LoRA, audio decoder, custom ComfyUI nodes, etc.

**Required methods:** `ensure_loaded`, `reserve`, `release`, `unload`, `health`, `recover`.

**Initial scheduling policy:** Prefer one active heavy runtime per GPU and explicit process unload/restart when switching models. Add more advanced caching/packing only after live VRAM data shows a measurable benefit.

---

## 3. Phase overview and gates

| Phase | Name | Core deliverable | Exit gate |
|---|---|---|---|
| **P0** | Baseline freeze & critical bug fixes | Real integrity, honest results, working fundamental flows | No false success; critical regression tests pass |
| **P1** | Registry, variant routing & GPU runtime | Runtime readiness and safe model switching | GPU-01–GPU-10 verified on supported hardware |
| **P2** | Finish existing subsystems | Text/VLM/Decision/Embedding/Speech/Image/Video/Enhancement | Each advertised capability either live-verified or explicitly not supported |
| **P3** | ACE-Step 1.5 full capability | Fast generation, source-audio operations, variant-aware native API | ACE capability matrix verified by variant |
| **P4** | YuE2 integration | Native proof, Factory generation, score, cover and editing | YuE2 functionality verified on declared hardware profiles |
| **P5** | Comprehensive live testing | Repeatable functional/cross-feature tests | Test outcomes and artifacts recorded without fake fallbacks |
| **P6** | Benchmark and reporting | Quality, time, VRAM, errors and A/B report | Reports can be regenerated from evidence |

**Dependency ordering:** P0 and minimum P1 safety precede heavy live multi-model testing. P2 and P3 can proceed in parallel after stable API/runtime contracts. P4 begins with independent YuE2 feasibility before extensive Factory integration. P6 is derived from P5 data, never hand-authored from expectations.

---

## 4. P0 — Fix existing bugs and integrity gaps

These issues arose from the prior source audit. Re-confirm each against the commit being changed; mark resolved issues as PASS with their regression-test evidence rather than reimplementing unnecessarily.

| ID | Priority | File/area | Required action | Acceptance criterion |
|---|---|---|---|---|
| FIX-01 | Critical | `factory/apps/control-api/src/queue/manager.py` | Reject success if required artifact is missing, empty, inaccessible or invalid | Missing output produces FAILED, not SUCCEEDED |
| FIX-02 | Critical | `queue/manager.py` | Remove fabricated all-zero SHA-256 fallback and compute/check real hashes | SHA-256 matches actual downloaded bytes |
| FIX-03 | Critical | `queue/manager.py` | Retry through the same GPU-aware enqueue logic as first submission | Retry cannot silently move a job to the wrong GPU |
| FIX-04 | Critical | Queue files: scheduler/leasing/manager/watchdog | Consolidate duplicated `SUBSYSTEM_GPU_MAP` into one policy | One source of truth, no mapping divergence |
| FIX-05 | Critical | `queue/watchdog.py` | Use configurable task/model/variant timeouts and heartbeat-aware deadlines | Slow valid video/music jobs are not killed prematurely |
| FIX-06 | Critical | `api/routers/jobs.py` | Remove misleading aliases (e.g. a 72B alias mapping to a 7B checkpoint) | Requested identity matches resolved weights |
| FIX-07 | Critical | Job admission / manifests | Check implemented capability, backend readiness, weights and hardware feasibility | Impossible/unserved jobs fail early, not indefinitely queue |
| FIX-08 | Critical | `services/speech/server.py`, `asr/engine.py` | Fix `audio_url`→Factory asset ingest→actual bytes→ASR handoff | Trusted uploaded audio reaches real ASR |
| FIX-09 | Critical | `services/music/launcher/run_acestep_api.sh` | Launch native ACE-Step instead of deprecated gateway | Launcher health checks native backend |
| FIX-10 | Critical | `services/music/acestep/client.py` | Align task type/payload with actual installed ACE-Step native API | Request maps to correct native task |
| FIX-11 | Critical | `workflows/image/parameter_mappings.py` | Choose the real Standard/Turbo checkpoint based on variant | Turbo selection loads Turbo weights, not just fewer steps |
| FIX-12 | Critical | `workflows/video/parameter_mappings.py` | Bind actual reference into MiniMax H3 `8_ref` | Ref2VA consumes provided asset |
| FIX-13 | Critical | `services/video/worker.py` | Concatenate original and generated segment for extension; validate codecs/timing | Final MP4 contains both segments in order |
| FIX-14 | Critical | `services/enhancement/` | Make backend health reflect model availability, not just HTTP process health | `/health` never signals infer-ready without backend |
| FIX-15 | Critical | `tests/e2e/conftest.py` | Split contract/mock and live-E2E suites, label both unambiguously | Default MockTransport tests never count as live inference |
| FIX-16 | Critical | `deployments/kaggle/` | Reclassify scripts/reports that use synthetic tensors/audio/video or fallback | Simulated report cannot carry LIVE_PASS |
| FIX-17 | High | Worker errors | Add actionable error taxonomy, stage and retry semantics | Errors not all reduced to generic `SERVICE_UNAVAILABLE` |
| FIX-18 | High | Studio Web streaming/inspector | Distinguish actual metrics from demonstration tokens/waveforms | Live UI contains no unlabeled fake telemetry |
| FIX-19 | High | CI/dependencies | Update vulnerable dependencies, enforce blocking critical security checks | CI fails for defined severe vulnerabilities |
| FIX-20 | High | Deploy/config/docs | Correct old launch paths and trim dead optional Redis/S3/legacy assumptions | One documented, actually working deploy path |

### 4.1 Error contract

| Code | Default retry | Policy |
|---|---|---|
| `INVALID_INPUT` | No | Invalid prompt, codec, mask, score, parameters |
| `UNSUPPORTED_CAPABILITY` | No | Task unsupported by this model/variant |
| `WEIGHTS_MISSING` | No automatic retry | Installer/admin action required |
| `MODEL_NOT_LOADED` | Conditional | Attempt runtime load if valid and allocated |
| `INSUFFICIENT_VRAM` | Conditional | Unload idle model, adjust supported profile or fail clearly |
| `BACKEND_UNAVAILABLE` | Limited | Retry with backoff after readiness recovery |
| `INFERENCE_TIMEOUT` | Conditional | Timeouts by actual workload and progress |
| `BACKEND_CRASHED` | Limited | Restart backend only with a controlled retry budget |
| `ARTIFACT_MISSING` | Do not blindly re-run inference | Investigate output/storage first |
| `LICENSE_RESTRICTED` | No | Deployment/use policy blocks use |

Every failure must log `job_id`, `request_id`, `model`, `variant`, `task`, `stage`, `error_code`, `retryable`, `attempt`, and correlated backend logs. Never leak internal stack traces, file paths, access tokens or secrets through public API responses.

### 4.2 P0 evidence

Deliver `bug_audit.md`: each FIX ID, original reproduction, failing test, fix commit, passing regression test, remaining risk, and evidence link. Record the exact repo commit used as the baseline.

---

## 5. P1 — Registry, model variants, GPU allocation, lifecycle

### 5.1 Capability truth model

Represent each capability through three distinct levels:

1. **Declared:** Upstream/model card says this feature exists.
2. **Implemented:** Factory has a real workflow/adapter, worker, backend and artifact path.
3. **Live verified:** It ran successfully on a named hardware+checkpoint profile with evidence.

Manifest fields should include model revision, required dependencies, task/variant matrix, validated hardware profiles, `installed`, `checkpoint_valid`, `backend_ready`, `validation_status`, `evidence_run_id`. Never infer real readiness from `coverage_percent: 100.0` or the presence of a workflow JSON file.

### 5.2 Variant-safe API contracts

```json
{
  "service": "image",
  "model": "qwen-image-2.1",
  "task": "text_to_image",
  "inputs": {
    "variant": "turbo",
    "prompt": "A cinematic coastline",
    "steps": 8,
    "seed": 42
  }
}
```

`variant=turbo` must select the actual Turbo checkpoint. Model-specific parameter validation should reject incompatible variants/steps/features instead of silently falling back.

### 5.3 GPU resource management

- Check available VRAM **before** reserving and load, and record actual post-load VRAM.
- Unload incompatible idle processes; verify memory returns near expected baseline.
- Start target backend, validate process, checkpoint, health, warmup and output-capable readiness.
- Maintain GPU lease until worker/child process and outputs are fully accounted for.
- If model uses two GPUs, reserve both atomically; never represent it as a normal single-GPU lease.
- Cancellation must reach the backend, avoid stale storage writes, and free leases.
- Model load failure should fail fast with an actionable error, not leave a perpetual QUEUED job.
- Initial strategy: one heavy active runtime/GPU; optimize caching only after benchmarks.

### 5.4 GPU switching test cases

| ID | Scenario | PASS condition |
|---|---|---|
| GPU-01 | Load model A, infer, unload | VRAM returns near idle baseline; process exited |
| GPU-02 | Switch A→B | Checkpoint and outputs belong to B |
| GPU-03 | Two jobs on distinct GPUs | Independent completion without allocation conflict |
| GPU-04 | Two heavy jobs on same GPU | Second waits or schedules valid alternate placement |
| GPU-05 | Dual-GPU model | Both reserved and released atomically |
| GPU-06 | Backend crash | Lease recovered; job terminates with useful error |
| GPU-07 | Cancel during load and inference | No stale busy GPU or ghost job |
| GPU-08 | Heartbeat, timeout, retry | No false timeout or infinite retry loop |
| GPU-09 | ACE-Step→YuE2→Video→ACE-Step | Correct backend/checkpoint per step, no OOM |
| GPU-10 | 10–20 repeated swaps | No meaningful cumulative VRAM leak or stuck queue |

**P1 exit:** Queue, model identity, backend readiness, and GPU lifecycle are safe enough for full live inference.

---

## 6. P2 — Complete existing Factory subsystems

Every implemented capability must cover: **manifest → API validation → worker/adapter → native backend → real artifact validation**.

### 6.1 Text, VLM, Decision and Embedding

| Subsystem | Current manifest baseline | Features to complete and verify |
|---|---|---|
| Text | Qwen2.5-7B-Instruct-AWQ | Chat, streaming, structured JSON, tool calling, context limits, stop tokens |
| VLM | Qwen2-VL-7B-Instruct-AWQ | Image QA, multi-image, video frames, document QA, grounded responses |
| Decision | Qwen2.5-3B-Instruct | Predicate evaluation, choice routing, score/ranking, output schema |
| Embedding | Qwen3-Embedding-0.6B | Single/batch embeddings, dimensions, normalization, retrieval |
| Reranking | Qwen3-Reranker-0.6B | Rerank multiple documents, batch score correctness |

**Tests:** Empty/invalid input, schema violations, long context, streaming chunk integrity, concurrent requests, unsupported modalities, real weights, real metrics. Do **not** equate separate Kaggle Qwen3.8-27B experimentation with the checkpoints declared by the Factory manifests. Keep multimodal embedding unsupported unless a real multimodal backend is installed.

### 6.2 Speech

| Capability | Implementation | Acceptance |
|---|---|---|
| ASR | Trusted asset ingestion, actual bytes, real Qwen3 ASR backend | Transcript is derived from uploaded audio |
| TTS | Real Qwen3 TTS backend and artifact upload | Audible, nonempty, correctly encoded waveform |
| Voice cloning | Reference audio + text, actual native cloning path | Input reference changes voice as expected |
| Streaming ASR | Chunk ordering, context/segment behavior, timestamp policy | Stable transcript assembly |
| Word alignment | Real forced aligner or mark unsupported | Token/word timestamps check against audio |
| Speed control | Actual speech rate transformation | Duration/speed changes without invalid artifact |
| Validation | Sample rate, codec, duration, asset security | Reject malformed/untrusted source |

Do not use fake voice embeddings or sine-wave audio as production output or live test evidence.

### 6.3 Image — Qwen-Image 2.1

| ID | Capability | Requirements |
|---|---|---|
| IMG-01 | Text-to-image | Real Standard/Turbo checkpoint selection and generation |
| IMG-02 | Image-to-image | Correct source image and denoise/strength mapping |
| IMG-03 | Instruction editing | Follow instruction and preserve unrelated content |
| IMG-04 | Inpaint | Real mask application, preserve unmasked regions |
| IMG-05 | Outpaint | Expanded canvas with preserved original content |
| IMG-06 | Reference image | Correct staged reference bound to workflow |
| IMG-07 | Multi-reference | Verify actual supported count and conditioning |
| IMG-08 | Pose control | Real pose model/preprocessor and conditioning |
| IMG-09 | Depth control | Real depth model/preprocessor and conditioning |
| IMG-10 | Variant switching | Standard↔Turbo uses distinct verified weights |

Cover seed/steps/CFG/sampler/resolution, source data, masks, output dimensions/MIME/hash/VRAM, and invalid input. Feature quality tests must assess prompt/edit fidelity, not merely file existence. Any controlnet claimed by a manifest must be wired to actual installed nodes/checkpoints.

### 6.4 Video — Wan 2.2 and MiniMax H3

| Model | Task | Required corrections / validation |
|---|---|---|
| Wan 2.2 | T2V | Select 5B/A14B explicitly; confirm model/workflow match |
| Wan 2.2 | I2V | Source image conditioning and stable motion |
| Wan 2.2 | FL2V | First/last frame binding and transitions |
| Wan 2.2 | Extension | Extract anchor, generate segment, concatenate original + segment |
| MiniMax H3 | T2VA | Generate actual video and synchronized audio |
| MiniMax H3 | I2VA | Correct image input and real audio track |
| MiniMax H3 | FL2VA | Real first/last frame binding and audio |
| MiniMax H3 | Ref2VA | Fix `8_ref` binding to user-provided reference |
| MiniMax H3 | Audio | Validate audible, non-silent, synchronized track |
| MiniMax H3 | Extension | Keep unsupported unless a genuine model/pipeline path is demonstrated |

Validate ComfyUI custom nodes, CLIP/VAEs, source staging, fps, frame count, dimensions, seed, duration, CUDA allocation, artifact upload and SHA. For extension, persist both `generated_segment.mp4` and `extended_video.mp4`; test codecs, fps, ordering and boundary continuity. A valid JSON workflow is **not** proof of working model inference.

### 6.5 Enhancement

Current manifest advertises four capabilities as false. Implement and enable them **one by one** when live proven.

| ID | Capability | Backend / workflow |
|---|---|---|
| EN-01 | Image super-resolution | Real-ESRGAN |
| EN-02 | Alternative image SR | BSRGAN |
| EN-03 | Video super-resolution | Chunk/frame pipeline using a real SR model |
| EN-04 | Frame interpolation | RIFE |
| EN-05 | Audio enhancement | DeepFilterNet |
| EN-06 | Cross-media pipeline | Video generation → interpolation/SR → final mux |

Preserve source duration, audio sync, timestamps, codecs; avoid uncontrolled CPU RAM use by batching and streaming chunks. Backend must fail with an honest status when weights/service are unavailable.

**P2 exit:** Each baseline model has at least one verified native inference path; each advertised capability has evidence or an explicit unsupported/unverified status.

---

## 7. P3 — Full ACE-Step 1.5 implementation

### 7.1 Variant-dependent feature matrix

Check the exact version and actual upstream model card/API before implementing; do not assume that a Turbo checkpoint includes specialized Base-only tasks.

| Feature | 2B Base | 2B Turbo/SFT | XL Base |
|---|---|---|---|
| Text / lyrics to music | Yes | Yes | Yes |
| Reference audio conditioning | Yes | Yes | Yes |
| Cover | Yes | Yes | Yes |
| Repaint | Yes | Yes | Yes |
| Extract | Yes | No | Yes |
| Lego / add layer | Yes | No | Yes |
| Complete / accompaniment | Yes | No | Yes |

**Suggested deployment:** Turbo for fast creation, Base for editing/extraction/arrangement. XL is optional and subject to actual VRAM feasibility. Mark every task's native compatibility and actual Factory verification separately.

### 7.2 Factory music API proposal

```json
{
  "service": "music",
  "model": "ace-step-1.5",
  "task": "cover",
  "inputs": {
    "variant": "base",
    "prompt": "Jazz arrangement with warm vocals",
    "source_audio_asset_id": "asset-source",
    "reference_audio_asset_id": "asset-style",
    "lyrics": "",
    "seed": 42,
    "output_format": "flac",
    "audio_cover_strength": 0.8
  }
}
```

This is a **proposed Factory contract**, not an exact native ACE-Step endpoint. The adapter maps to installed native task names/parameters (such as `task_type`, `src_audio_path`, `reference_audio_path`, `audio_duration`, `repainting_start/end`) after verifying the running upstream release.

### 7.3 ACE-Step test/feature checklist

| ID | Feature | PASS / implementation requirement |
|---|---|---|
| ACE-01 | Text-to-Music | Real audio from a style prompt |
| ACE-02 | Lyrics-to-Music | Lyrics sung with discernible verse/chorus |
| ACE-03 | Instrumental | Output conforms to instrumental instruction |
| ACE-04 | Prompt expansion/query rewrite | Model-derived prompt metadata rather than fake rewriting |
| ACE-05 | Thinking mode | Parameter maps correctly and affects native execution |
| ACE-06 | Musical controls | BPM/key/time signature/duration passed and validated |
| ACE-07 | Reference audio | Actual staged reference impacts style |
| ACE-08 | Cover | Uses source audio; audible cover/arrangement output |
| ACE-09 | Repaint | Edits intended interval, preserves outside interval as appropriate |
| ACE-10 | Remix/style transfer | Real cover/reference implementation and documented constraints |
| ACE-11 | Extract | Extract specified stem with compatible Base checkpoint |
| ACE-12 | Lego | Add target instrument/layer to existing source |
| ACE-13 | Complete | Generate accompaniment/complete source input |
| ACE-14 | Music extend | Verify actual continuation path; do not relabel unrelated task |
| ACE-15 | Multilingual lyrics | Vietnamese/English/Japanese pronunciation and adherence |
| ACE-16 | Batch generation | Independent seeds, artifacts and timings |
| ACE-17 | Variant switching | Base/Turbo/SFT/XL map to real checkpoints and feature flags |
| ACE-18 | Audio format export | MP3/WAV/FLAC valid where supported/conversion provided |
| ACE-19 | Audio understanding | Native metadata/caption feature if supported/installed |
| ACE-20 | Lyrics timing/LRC | Real timestamps where supported; else mark unsupported |
| ACE-21 | Quality scoring | Scores from actual scoring model with provenance |
| ACE-22 | LoRA inference | Load compatible trained LoRA and verify its effect |
| ACE-23 | LoRA training | Separate optional advanced training workflow with verified reload |
| ACE-24 | Failure/recovery | Invalid asset, cancel, crash, OOM, timeout, retries |

**Important semantics:** `cover` or `style transfer` is **not** automatically precise identity-preserving voice conversion. If the project requires swapping a named singing voice independently from accompaniment, add a separately validated voice-conversion/source-separation pipeline rather than overstating ACE-Step.

### 7.4 Files to update

- `services/music/acestep/adapter.py`: task/variant-specific mapping and input validation.
- `services/music/acestep/client.py`: source and reference audio handling, native task calls, polling/download, metadata.
- `services/music/worker.py`: asset staging, backend selection, multi-artifact upload, heartbeat/cancel/error status.
- `services/music/launcher/`: one official native launcher and readiness check.
- `manifests/models/ace-step-1.5.yaml`: task/variant/hardware truth matrix.
- `services/music/tests/`: unit, integration and live native tests.

**P3 exit:** Baseline creation, cover, repaint, and Base-specific operations validated end-to-end on declared hardware; unsupported or advanced-only functions marked precisely.

---

## 8. P4 — YuE2 integration

### 8.1 Runtime feasibility alternatives

**Track A — Official Python runtime:** Install official YuE2 and its real weights on a supported BF16 GPU; validate the planning/semantic/synthesis/decode pipeline, then integrate through an adapter.

**Track B — Quantized/GGUF runtime:** Independently test candidate quantizations (e.g. Q8_0, Q6_K, Q5_K_M where available) on Dual T4; record exactly which stages and editing/score features remain available. A working GGUF basic generation does not certify full Python feature parity.

**Hard gate Y0:** Before large implementation, create one real song using original lyrics and style, validate WAV/audio duration, elapsed time, peak VRAM, device placement, logs, and fallback flag. If Y0 fails, document the specific hardware/runtime blocker and continue the other phases.

### 8.2 YuE2 dependency plan

| Dependency | Purpose | Policy |
|---|---|---|
| YuE2-3B | Main song generation | Required |
| YuE2-VAE | Acoustic decode | Required |
| YuE2 legacy VAE | Reference compatibility/benchmark | Optional specific benchmark |
| SheetSage2 | Audio recording to symbolic score | Needed for source-audio cover route |
| MERT-v2-FullSong | Encoder in score-transcription route | Required if selected transcription pipeline requires it |
| ABC parser/editor | Validate and modify symbolic score | Needed for score control/editing |

Check official compatibility, weights, licenses and the exact pipeline before downloading all dependencies; load optional components on demand.

### 8.3 YuE2 full capability checklist

| ID | Feature | PASS / implementation requirement |
|---|---|---|
| YUE-01 | Lyrics + Style → Song | Real coherent audio, lyrics and style adherence |
| YUE-02 | Full symbolic planning | Export valid melody/chord score |
| YUE-03 | Melody-only planning | Generate valid melody-only plan |
| YUE-04 | No-score generation | Verify actual no-score mode |
| YUE-05 | Custom score | Accept/validate user ABC and render according to score |
| YUE-06 | Plan-only | Save score without requiring audio render |
| YUE-07 | Semantic generation | Generate real intermediate semantic representation |
| YUE-08 | Acoustic synthesis | Native acoustic synthesis stage succeeds |
| YUE-09 | Audio decode | Nonempty, validated stereo output where supported |
| YUE-10 | Plan reuse | Re-render from same composition deterministically enough to inspect |
| YUE-11 | Song variation | Multiple independent seeds/configs and artifacts |
| YUE-12 | Audio-to-score | Actual transcription backend produces valid symbolic score |
| YUE-13 | Zero-shot cover | Source audio→score→new arrangement with real artifacts |
| YUE-14 | Style reinterpretation | Preserve intended melody while changing arrangement |
| YUE-15 | Score editing | Edits notes/chords/tempo/form and regenerates audio |
| YUE-16 | Agentic editing | Natural-language edit→validated score diff→render; no multi-agent prerequisite |
| YUE-17 | Artifacts/provenance | Audio, ABC, metadata, optional intermediate tensors/tokens, hashes |
| YUE-18 | Quantization A/B | Native and quantized compare with hardware/feature caveats |
| YUE-19 | Long song | Duration/structure and GPU scaling tested |
| YUE-20 | Failure recovery | OOM, malformed ABC, invalid reference, timeout, cancel |

**Semantic distinction:** YuE2 score editing modifies the symbolic composition and re-renders it. ACE-Step repaint targets a selected audio region. Benchmark them against their intended editing objectives rather than treating them as identical operations.

### 8.4 Proposed source layout

```text
services/music/
├── acestep/
│   ├── adapter.py
│   ├── client.py
│   └── runtime.py
├── yue2/
│   ├── adapter.py
│   ├── client.py
│   ├── runtime.py
│   ├── score.py
│   └── transcription.py
├── assets.py
├── contracts.py
├── worker.py
└── tests/

manifests/models/
├── ace-step-1.5.yaml
└── yue2.yaml
```

This is a **target** directory proposal, not an assertion these files already exist. Reuse the Factory queue and common storage contract; do not create a second incompatible Music job system.

### 8.5 YuE2 incremental milestones

1. **Y1 — Native proof:** Supported environment, verified weights, licensing, one actual song.
2. **Y2 — Basic Factory integration:** Register model and generation tasks; real progress, audio and job artifacts.
3. **Y3 — Symbolic score:** Plan-only, ABC validation, custom score, re-render, variations.
4. **Y4 — Cover:** Source audio→SheetSage2 (or verified alternative)→score→cover; also accept existing ABC.
5. **Y5 — Editing:** Score diff, validation, render; optional natural-language editor.
6. **Y6 — Quantization/hardware:** Per-variant GPU compatibility and feature-parity matrix.
7. **Y7 — Live acceptance:** Generation, cover and edit verified on each explicitly supported hardware profile.

---

## 9. P5 — Full test strategy and 163-case matrix

### 9.1 Four separate testing tiers

| Tier | Purpose | Real model/GPU required? | Can certify inference? |
|---|---|---|---|
| T1 — Unit/contract | Parameter validation, mapping, state transitions, API responses | No | **No** |
| T2 — Integration | API↔worker↔backend↔storage coordination | Optional | Only if actual model inference is used |
| T3 — Live model | Real checkpoint, backend, inference, artifact | **Yes** | **Yes** |
| T4 — Quality/benchmark | Performance, reliability and quality comparison | **Yes** | **Yes**, subject to metrics |

Default `tests/e2e/conftest.py` currently uses mock transport when `FACTORY_E2E_LIVE` is unset. Relabel simulated tests and build a genuinely asynchronous live runner which waits for job terminal state and downloads/verifies artifacts.

### 9.2 Complete experiment inventory

| Group | IDs | Cases | Content |
|---|---|---:|---|
| Router/infrastructure | R01–R28 | 28 | Authentication, submission, states, leases, retries, cancellation, assets, security, watchdog |
| GPU runtime | GPU-01–GPU-10 | 10 | Load/unload, model switching, dual GPU, crash recovery |
| Text | T01–T05 | 5 | Chat, JSON, tool calls, stream, context |
| VLM | V01–V05 | 5 | Image/multi-image/video/document/grounding |
| Decision | D01–D03 | 3 | Predicate, routing, ranking |
| Embedding | E01–E04 | 4 | Single/batch embedding, retrieval, reranking |
| Speech | S01–S11 | 11 | ASR, TTS, cloning, alignment, streaming, errors |
| Image | I01–I15 | 15 | T2I, edit/inpaint/outpaint, references, pose/depth, variants, errors |
| Wan 2.2 | W01–W08 | 8 | T2V, I2V, FL2V, extension, limits, faults |
| MiniMax H3 | H01–H10 | 10 | T2VA, I2VA, FL2VA, Ref2VA, audio and faults |
| Video integration | X01–X02 | 2 | Artifact/mux/postprocessing integration |
| ACE-Step | ACE-01–ACE-24 | 24 | Generation, reference, cover, edits, multi-track, variants, LoRA |
| YuE2 | YUE-01–YUE-20 | 20 | Generate, score, cover, edit, quantization |
| Enhancement | EN01–EN06 | 6 | Image/video SR, interpolation, denoise, cross-media |
| Cross-feature | P01–P12 | 12 | Model chaining, asset handoff, mixed workload |
| **TOTAL** | | **163** | **Planned cases — not 163 completed tests** |

The 163 cases define the **minimum tracked matrix**, with each case expanded by relevant checkpoint/variant/hardware profiles as needed. A single logical ID may have multiple measured runs. Regression tests discovered during development increase the total.

### 9.3 Cross-feature pipeline scenarios

| ID | Pipeline | Acceptance focus |
|---|---|---|
| P01 | Text → Image | Accurate prompt-to-image generation |
| P02 | Image → Image Edit | Transformation and preservation correctness |
| P03 | Image → Video | Image conditioning and motion consistency |
| P04 | Text → MiniMax H3 | Real synchronized audio/video |
| P05 | Wan Video → Extend | Correct concatenation and transition |
| P06 | Video → Frame Interpolation | Target FPS with preserved duration/audio |
| P07 | Image → SR | Larger valid image with inspectable detail |
| P08 | Speech ASR → Text → TTS | Content preserved across modalities |
| P09 | ACE-Step → Cover → Repaint | Actual source artifact passed to each next stage |
| P10 | YuE2 → Score Edit → Re-render | Score diff is reflected in new audio |
| P11 | YuE2 → ACE-Step/Enhancement | Audio format/codec/sample rate handoff |
| P12 | Mixed model workload | VRAM swaps, queue integrity, no OOM/stuck jobs |

### 9.4 Test statuses

| Status | Meaning |
|---|---|
| `PASS` | All criteria met with verifiable evidence |
| `FAIL` | Executed but output/behavior violates criteria |
| `ERROR` | Runtime/system error prevented valid execution |
| `BLOCKED` | Dependency/configuration prerequisite missing |
| `HARDWARE_BLOCKED` | Hardware cannot execute the requested profile |
| `NOT_IMPLEMENTED` | Factory execution path is absent |
| `SKIPPED` | Intentionally excluded from a documented test profile |
| `NOT_RUN` | Not yet executed |

`PASS` in T3/T4 requires real weights, verified backend, non-placeholder output, actual artifact and checksum, and `mock_used=false`, `fallback_used=false`.

### 9.5 Hardware matrix

| Profile | Scope |
|---|---|
| Dual Tesla T4 (16 GB × 2) | Main production-constrained target |
| Single T4 (16 GB) | Graceful failure and minimal viable support |
| NVIDIA BF16 ≥24 GB | Official YuE2 reference-like configuration |
| CPU offload / GGUF | Optional constrained model alternative |
| Dual GPU split | Only for backend expressly supporting sharding/split |

Track `gpu_name`, exact available memory, CUDA/toolchain, driver, framework, kernel/runtime settings, backend and model revision per measured run.

---

## 10. P6 — Reproducible benchmark and reports

### 10.1 Common metrics

| Area | Required measurements |
|---|---|
| Outcome | Status, expected/actual, errors, retry count |
| Latency | Queue wait, cold load, warmup, inference, postprocess, end-to-end |
| GPU | Peak/idle VRAM **per GPU**, utilization, allocated device IDs |
| Host | Peak RAM, CPU if measurable |
| Artifact | Path, MIME, bytes, SHA-256, duration/resolution, validation |
| Reliability | Success/OOM/timeout/cancel rates, memory recovery |
| Provenance | Git SHA, weights hash/revision, backend version, prompt/config/seed |
| Integrity | Mock/fallback flags, source asset IDs, log and evidence IDs |

Do not substitute zero for unmeasured metrics; use `null` with a status/reason. Separate load time and inference time; benchmarks of warm inference should not hide cold-start overhead.

### 10.2 Domain metrics

| Domain | Useful metrics |
|---|---|
| Text | TTFT, tokens/s, schema adherence, tool-call success |
| VLM | Grounding and task accuracy, image/video understanding |
| Decision | Choice accuracy, ranking agreement, invalid output rate |
| Embedding | Recall@K, MRR, nDCG@K |
| ASR | WER, CER per language |
| TTS/voice cloning | Intelligibility, speaker similarity, human listening/MOS |
| Image | Prompt adherence, edit preservation, visual quality; reference metrics if available |
| Video | Temporal consistency, motion, audio presence/sync, sec/frame |
| Music | Lyric adherence, coherence, genre fidelity, vocal quality, RTF, listener preference |
| Enhancement | PSNR/SSIM/LPIPS when ground truth exists, perceptual scoring otherwise |

For audio, **RTF = generation wall-clock seconds / output audio seconds**. A smaller RTF is faster. Record whether generation includes planning/selection or only synthesis.

### 10.3 ACE-Step 1.5 vs YuE2 benchmark suite

| ID | Prompt family | Evaluation focus |
|---|---|---|
| B-M01 | English pop | Lyrics, structure, vocals |
| B-M02 | Vietnamese ballad | Vietnamese pronunciation, melody |
| B-M03 | Japanese pop | Multilingual singing |
| B-M04 | Cinematic instrumental | Timbre, arrangement, realism |
| B-M05 | EDM/electronic | Rhythm, progression, energy |
| B-M06 | Acoustic/jazz | Harmony, musicality, instrumentation |
| B-M07 | Complex lyrics | Verse/chorus consistency |
| B-M08 | Long song | Long-range coherence and memory |
| B-M09 | Style reference | Prompt/reference adherence |
| B-M10 | Cover | Melody preservation vs arrangement change |
| B-M11 | Editing | ACE audio repaint vs YuE2 composition edit, using task-specific rubrics |
| B-M12 | Repeated generation | Diversity, stability, latency variance |

**Fairness rules:**

- Use shared lyrics/style descriptors where supported, but do not force semantically non-equivalent features.
- Generate **at least three independent samples per condition** for the core comparison; more for meaningful statistical claims.
- Compare same-hardware latency only if both models actually run on that hardware. Otherwise publish **hardware-normalized context**, not misleading absolute rankings.
- Report `best-of-N` and full cumulative compute cost if any candidate selection is used; never compare best-of-8 to the other's one-shot output without labeling.
- Use blind A/B listening when possible; save independent rater scores and notes.
- Distinguish native model quality from quantization trade-offs and postprocessing.
- Do not claim universally superior SOTA performance from one limited private benchmark.

### 10.4 Human listening rubric (score 1–5)

Evaluate each output on: lyric intelligibility, melody/harmony, arrangement/instrumentation, vocal quality, style adherence, overall song coherence, audible artifacts/distortion, and overall preference. Store individual raters, blinded sample IDs, scores, notes, and aggregate mean/dispersion. Subjective listening is not an objective accuracy score.

### 10.5 Proposed report and results directory

```text
benchmarks/
├── configs/
│   ├── hardware_profiles.yaml
│   ├── evaluation_cases.yaml
│   └── music_comparison.yaml
├── datasets/
│   └── test_manifest.json
├── runners/
│   ├── contract_runner.py
│   ├── live_runner.py
│   └── report_builder.py
├── results/
│   └── <run_id>/
│       ├── metadata.json
│       ├── results.jsonl
│       ├── metrics.csv
│       ├── failures.json
│       ├── gpu_telemetry.csv
│       ├── audio_reviews.csv
│       ├── artifacts/
│       └── logs/
└── reports/
    ├── executive_summary.md
    ├── capability_matrix.md
    ├── model_comparison.md
    ├── latency_vram.md
    ├── failure_analysis.md
    └── final_report.md
```

### 10.6 Canonical live-test result schema

```json
{
  "test_id": "YUE-01",
  "suite": "live_model",
  "status": "NOT_RUN",
  "model": "yue2",
  "variant": "q8_0",
  "task": "lyrics_to_music",
  "environment": {
    "git_commit": null,
    "model_revision": null,
    "backend_version": null,
    "gpu_names": [],
    "gpu_ids": []
  },
  "parameters": {
    "seed": 42,
    "prompt_id": "B-M02",
    "target_duration_seconds": 120
  },
  "measurements": {
    "queue_seconds": null,
    "load_seconds": null,
    "inference_seconds": null,
    "end_to_end_seconds": null,
    "peak_vram_mb_per_gpu": [],
    "actual_duration_seconds": null,
    "rtf": null
  },
  "artifact": {
    "audio_path": null,
    "score_path": null,
    "sha256": null,
    "validated": false
  },
  "integrity": {
    "mock_used": false,
    "fallback_used": false,
    "weights_verified": false
  },
  "error": null
}
```

This is an **example schema with no measured results**, not a completed test.

### 10.7 Final report structure

1. **Executive summary:** Totals and pass/fail/blocked breakdown; no hidden excluded tests.
2. **System infrastructure:** API, worker, queue, storage, GPU runtime and reliability.
3. **Model readiness:** Exactly which checkpoint/tasks/variants run on each hardware profile.
4. **Capability matrix:** Declared vs implemented vs live-verified, with evidence ID.
5. **Functional results:** Every test ID, expected/actual, status, artifact and failure explanation.
6. **Performance:** Cold load, inference, E2E, throughput, VRAM per GPU, reliability.
7. **Quality:** Text, VLM, Speech, Image, Video, Music, Enhancement assessments.
8. **ACE-Step vs YuE2:** Native capabilities, quality, cost/speed/hardware, caveats.
9. **Failure analysis:** Root causes, affected IDs, fix commits, retest outcomes.
10. **Recommendations:** High-impact improvements derived from real measurements.
11. **Reproduction:** Commands, environments, hashes, weights, prompts, seeds, dataset licensing.

Provide both summary tables and **per-case results**; a single overall PASS percentage or polished chart is insufficient.

---

## 11. Acceptance gates and definition of done

| Gate | Condition |
|---|---|
| G0 — Bug Fix | Critical FIX cases resolved with regression evidence |
| G1 — Runtime | Correct model identity, safe switching, honest GPU admission, artifact integrity |
| G2 — Core Models | Each subsystem has at least one real, verified end-to-end path |
| G3 — Capabilities | Every advertised feature is live-verified or transparently unsupported/not verified |
| G4 — ACE-Step | All selected native tasks verified against correct model variants |
| G5 — YuE2 | Generate, plan, score/edit, cover where dependencies exist; hardware precisely labeled |
| G6 — Reliability | No false success, undeclared fallback, mock contamination or infinite queue jobs |
| G7 — Benchmark | Metrics, artifacts, logs, A/B comparisons and reproducible reports complete |

A model/task marked `HARDWARE_BLOCKED` must **not** be recorded as PASS for that hardware. An optional advanced feature (e.g. LoRA training) may remain explicitly `NOT_IMPLEMENTED` without masquerading as a failed basic music generation capability. Public capability advertisements must reflect that status.

---

## 12. Recommended PR sequence

| PR | Scope | Main dependencies |
|---|---|---|
| PR-01 | Baseline audit, test classification, integrity fixes | None |
| PR-02 | Queue, retry, timeout, aliases, error taxonomy | PR-01 |
| PR-03 | ModelRuntimeManager, variant/readiness, safe GPU switching | PR-02 |
| PR-04 | Text/VLM/Decision/Embedding and Speech fixes | PR-02/03 contracts |
| PR-05 | Qwen-Image variants, control/edit workflows | PR-03 |
| PR-06 | Wan/H3 workflow and video extension fixes | PR-03 |
| PR-07 | Real enhancement backends | PR-03 |
| PR-08 | ACE-Step full native source-audio feature expansion | PR-03 |
| PR-09 | YuE2 feasibility, native/quantized track, basic generation | PR-03; Y0 native proof |
| PR-10 | YuE2 score, transcription, cover and editing | PR-09 |
| PR-11 | Full real-GPU test suite and cross-feature coverage | Relevant model PRs |
| PR-12 | Benchmark analysis, final report and Studio readiness UI | PR-11 |

Keep PRs reviewable and independently testable. PR-09's hardware feasibility should not block unrelated subsystem work.

---

## 13. Execution checklist

### Immediate first steps

- [ ] Record baseline `git commit` and verify repo tree against this plan.
- [ ] Open FIX-01 through FIX-20 as traceable issues/tasks.
- [ ] Separate mock/simulation and live model suites before further certification claims.
- [ ] Fix artifact validation, SHA-256, alias identity, retry GPU mapping, watchdog timeout.
- [ ] Implement the minimum viable GPU Runtime Manager and switching tests.
- [ ] Prove one real generation per subsystem, with model revision, weights and artifacts.
- [ ] ACE-Step: prove native `text_to_music`, `lyrics_to_music`, then staged source-audio task.
- [ ] YuE2: prove official native song creation on supported GPU **before** promising Dual-T4 execution.
- [ ] Only then widen capability matrices, run the full 163 planned cases, and report.

### Completion checklist

- [ ] All critical bugs tracked to resolution, evidence or documented blocker.
- [ ] Registry clearly distinguishes declared/implemented/live capabilities.
- [ ] Correct checkpoint selected for each model variant and task.
- [ ] Load/unload/GPU reservation/dual-GPU locks/cancel/retry tested.
- [ ] Speech, Image, Video, Enhancement have honest real-model paths.
- [ ] ACE-Step task families and native variants implemented/verified.
- [ ] YuE2 integrated with score/cover/edit where hardware/dependencies allow.
- [ ] 163 planned logical tests have explicit statuses and run evidence.
- [ ] Zero synthetic/fallback outputs counted as live PASS.
- [ ] Artifacts have validated hashes and meaningful telemetry.
- [ ] Final markdown reports and machine-readable result files generated from test evidence.

---

## 14. References and validation notes

- Factory source: [nxc1802/Gen_Image-Video](https://github.com/nxc1802/Gen_Image-Video).
- Existing baseline source paths include `factory/apps/control-api/src/queue/`, `services/`, `workflows/`, `manifests/models/`, `tests/e2e/`, `deployments/kaggle/`.
- ACE-Step upstream and its exact current task/variant matrix **must be checked against the installed release** before implementing task payloads; official upstream: [ACE-Step](https://github.com/ace-step/ACE-Step).
- YuE2 model/runtime, dependencies, GPU requirements, quantization variants and licenses **must be re-verified against the chosen release** during Y0; model reference: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B).
- All live experiment claims must cite their own artifact+metrics+log run ID. This plan is based on prior code audit/discussion and **does not certify that the latest branch has already implemented or passed any test**.

---

**Final acceptance criterion:** Factory operates as a trustworthy multimodal generation system with ACE-Step 1.5 and YuE2 coexisting, safe model/GPU switching, a clear capability matrix, real artifacts, an auditable **163-case test plan**, and reproducible quantitative/qualitative reports. Implementation comes first; optimization follows evidence.
