# Factory v4 Lite — Simplification, Correctness & Full Kaggle Benchmark Plan

> **Repository:** https://github.com/nxc1802/Gen_Image-Video  
> **Baseline reviewed:** `main` — `145a16f218dbe828032c74ae8a8293dbf5946219` (2026-11-08)  
> **Target:** Personal Production, inference-first, Kaggle Dual T4 benchmark  
> **Document type:** Implementation plan; **not** an assertion that the listed changes have been implemented.

## 1. Objectives and non-negotiable principles

Transform Factory v4 Lite into a lightweight **Personal AI Production Platform** that delivers authentic model inference through a usable API and Studio Web, with a single reproducible benchmark path on Kaggle.

1. **Simplification:** Remove unused code, redundant documents, CI/CD, duplicated test suites, historical scripts, and unjustified abstractions. Reduce actual runtime complexity, not merely the number of files.
2. **Correctness:** Verify model identity, API contracts, job lifecycle, GPU scheduling, worker execution, output integrity, errors, and security. Never count fabricated outputs or simulated model state as real inference.
3. **Full Kaggle Benchmark:** Evaluate every declared model, capability, and applicable variant using real weights and the native backend. Record explicit blockers when a case cannot run.

**Scope boundaries:** Do **not** rebuild the whole project, add new models, perform fine-tuning, migrate to Kubernetes, introduce enterprise RBAC, implement sophisticated autoscaling, or replace working inference engines only to improve benchmark scores. Do not optimize deeply before basic correctness and feasibility are demonstrated.

**Integrity contract:**

```text
REAL BACKEND AVAILABLE   -> real inference -> validated result
REAL BACKEND UNAVAILABLE -> explicit unavailable/error status
NEVER                    -> fake images, videos, waveforms, embeddings,
                            transcripts, fabricated telemetry or mock PASS
```

## 2. Target architecture

```text
Studio Web / API Client
          |
          v
FastAPI Control Plane
  - Single API key authentication
  - Model registry and capability validation
  - In-memory queue / job lifecycle / worker leases
  - Discrete GPU scheduling
  - Streaming progress and error handling
  - Ephemeral artifact storage
          |
          +--> GPU 0: Text / VLM / Decision / Image / Speech
          |
          +--> GPU 1: Video / Music / Embedding / Enhancement
                          |
                          v
               Native inference engines
               (vLLM, ComfyUI, ACE-Step, etc.)
                          |
                          v
              Real artifacts + telemetry
```

The mapping above is the **current intended service placement**, not proof that every model can fit or run concurrently. Kaggle Dual T4 provides two distinct GPUs: VRAM is **not** automatically pooled. GPU compatibility and model switching must be validated by live execution.

Retain the existing structure where useful: `factory/apps/control-api`, `factory/apps/studio-web`, `packages`, `services`, `workflows`, `manifests`, and `deployments/kaggle`. Avoid cosmetic renaming and unnecessary interface changes.

## 3. Phase 0 — Freeze, inventory, and recovery

### Actions

1. Record baseline commit and create a backup tag/branch before deletions.
2. Build an inventory of source files, runtime entrypoints, imports, manifests, scripts, datasets, checkpoints, and environment variables.
3. Classify every candidate as `REQUIRED_RUNTIME`, `REQUIRED_BENCHMARK`, `REDUNDANT`, or `CONDITIONAL`.
4. Trace dependencies of files proposed for removal, including imports, CLI entrypoints, workflow references, documentation commands, and deployment scripts.
5. Map 11 model manifests to the **actual** checkpoint, runtime/backend, environment requirements, supported tasks and variants.
6. Preserve any unique Kaggle inference evidence and previous benchmark artifacts before cleanup.

**Exit gate:** A deletion inventory, model–engine map, working baseline and recovery point exist. No candidate is deleted while relevant references remain unresolved.

## 4. Phase 1 — Aggressive but safe simplification

### 4.1. Repository cleanup

| Current component | Target action | Safeguard |
|---|---|---|
| `.github/workflows/` | Remove CI workflows | Keep manual preflight and benchmark commands; resolve known dependencies/security issues before release |
| `legacy/` | Remove from production branch | Preserve in history/tag if valuable |
| `tests/` | Drastically consolidate | Retain minimal correctness/security preflight; delete duplicated historic and Mac-specific suites |
| `Factory_v4_Lite_Full_Implementation_and_Benchmark_Plan.md` | Archive/delete after extraction | Transfer essential model/task requirements into canonical benchmark config |
| `Refactor.md`, `PROJECT.md`, `ORIGINAL_REQUEST.md` | Archive/delete | Keep only information needed to run or maintain production |
| `TEST_INFRA.md`, `TEST_READY.md`, `LIVE_VALIDATION.md`, `bug_audit.md` | Replace with concise run/verification guide | Do not lose unresolved safety or correctness findings |
| `benchmarks/reports/` | Remove stale or redundant reports | Retain reproducible verified evidence and summary |
| `deployments/kaggle/` scripts | Consolidate to one bootstrap and one benchmark path | Preserve unique model support and real assets |
| Empty service/workflow README files | Delete | No behavior change |
| `manifests/` | Keep | Single source of truth for models and capabilities |
| `workflows/` | Keep working graphs; remove duplicates | Cross-check every task/variant |
| `factory/apps/studio-web/` | Keep; simplify demo-only or redundant UI | Preserve API Explorer, jobs, results, streaming as needed |

**Important:** Do not remove all automated checks merely because GitHub CI is unnecessary for a personal project. Security invariants and basic lifecycle behavior still need a small, executable regression gate.

### 4.2. Runtime simplification

- **Authentication:** Prefer one production bearer API key. Remove JWT/RBAC only after verifying no necessary worker authentication, session, or API route depends on it. Preserve safe worker ownership and access control.
- **Storage:** Default to local ephemeral scratch storage. Remove S3 adapter/config only if no cross-host artifact persistence, backup or transfer is needed.
- **Observability:** Keep structured logs, error codes, model/checkpoint ID, job timings, real GPU utilization/VRAM, queue state, and artifact hashes. Remove unused tracing/export infrastructure.
- **Queue/scheduling:** Retain a single well-defined GPU assignment and lease mechanism. Eliminate duplicate mappings/state sources when safe. Preserve cancellation, retries, watchdog, timeout and concurrency safety.
- **Registry:** Retain authoritative capability/variant declarations. Simplify unused schema fields, but never conflate desired capability, backend availability, and live-verified readiness.
- **Studio:** Remove redundant demo scaffolding where appropriate. Clearly separate live streaming from synthetic UI preview.
- **Environment:** Reduce `.env.example` to genuinely used variables with correct defaults and an explicit minimal launch command.

### 4.3. Replace local Mac-centric test system

Remove historical rounds of adversarial testing, duplicated contract suites and Mac-specific GPU tests, **after** transferring high-value assertions to a minimal Kaggle preflight suite. Proposed preflight coverage:

1. Imports, startup, manifest loading and environment configuration.
2. Authentication, error format and unsupported capability rejection.
3. Job creation, queue/lease ownership, completion and failure.
4. GPU allocation/release, timeout, retry and cancellation.
5. Real artifact presence, MIME, size, hash and download.
6. URL/path containment, upload limits and SSRF prevention.
7. Backend unavailable -> explicit failure; no fake inference fallback.

Aim for roughly **10–15 critical preflight scenarios**, adjusting to actual coverage, rather than preserving thousands of low-value assertions. Execute on Kaggle when feasible; basic CPU-only assertions may also be runnable manually for debugging, but there is no requirement to maintain a separate Mac CI/test matrix.

### 4.4. Suggested production tree

```text
Factory/
├── factory/
│   └── apps/
│       ├── control-api/
│       └── studio-web/
├── packages/
│   ├── contracts/
│   ├── auth/
│   ├── model-registry/
│   └── client-sdk/
├── services/
│   ├── text/
│   ├── vlm/
│   ├── decision/
│   ├── embedding/
│   ├── speech/
│   ├── image/
│   ├── video/
│   ├── music/
│   └── enhancement/
├── workflows/
├── manifests/
├── benchmarks/
│   ├── cases.yaml
│   ├── runner.py
│   ├── validators.py
│   └── report.py
├── deployments/
│   └── kaggle/
│       └── bootstrap.py
├── README.md
├── BENCHMARK.md
├── .env.example
├── pyproject.toml
└── uv.lock
```

This is a **target organization**, not an instruction to rename functioning modules without reason.

**Exit gate Phase 1:** API starts, accepts a job, communicates with a real backend (where present), delivers artifacts, and all core imports resolve. No active command relies on deleted files.

## 5. Phase 2 — End-to-end correctness

### 5.1. Mandatory correctness matrix

| Area | Correctness requirement |
|---|---|
| API | Request schema and model/task/variant validation are consistent |
| Model | Exact intended checkpoint/revision and backend execute, without silent fallback |
| Runtime | `READY` reflects real engine/model availability rather than only bookkeeping |
| GPU | Real GPU identity and allocation; no stale leases or fabricated VRAM usage |
| Queue | Correct state transitions, no unintended double execution, safe retry and cancellation |
| Worker | Does not mark `SUCCEEDED` before valid output and artifact completion |
| Artifact | Correct MIME, dimensions/duration where relevant, non-empty bytes, checksum, usable download |
| Security | Bearer authentication, worker ownership, SSRF/path defense, upload boundaries |
| Errors | Unavailable backend/model gives explicit non-success, never synthetic output |
| Studio | Live capability and status are not misrepresented by demo-only streams |

### 5.2. Concrete issues found in baseline

**C1 — ModelRuntimeManager truth.** Distinguish `estimated_vram_gb`, planned assignment and in-memory state from confirmed live checkpoint loading. Readiness should be validated by native inference engine or trustworthy probe.

**C2 — Capability truth.** Align `capabilities`, variants, `installed`, `backend_ready`, `validation_status` and `live_verified`. The manifest declaration alone is not proof of runnable functionality.

**C3 — Storage config.** `.env.example` uses `STORAGE_BACKEND=ephemeral_disk`, whereas current configuration uses a `local`/`s3` selector. Harmonize the accepted value and documentation.

**C4 — Launch entrypoint.** README describes `factory_control_api.server:app`; inspect and verify the actual exported `factory_control_api:app` entrypoint, fix documentation/packaging and execute a startup smoke test.

**C5 — Artifact deletion.** README asserts automatic unlink after successful download, but `cleanup_on_download` is disabled by default. Choose a policy and ensure implementation, settings and docs agree.

**C6 — Secret hygiene.** Remove a hardcoded API key string in `benchmarks/runners/live_runner.py`. Use Kaggle Secrets/environment. Rotate any credential if it was ever live.

**C7 — Studio live/demo distinction.** Replace demo-only token/audio streams with live data if in scope, or label them clearly. Never report synthetic visualizations as actual model telemetry.

**C8 — Hardware-profile readiness.** Capability may be available only with certain checkpoint quantization, offload or host GPU. Attach evidence and hardware profile to readiness.

**C9 — Existing CI regressions.** The latest baseline failed Python tests due to an absolute Mac filesystem path in a YuE2 test, and security/dependency checks reported issues. Removing CI is not a substitute for resolving actual security exposure or runtime vulnerabilities. Remove the broken test architecture only after preserving essential checks.

**C10 — Live certification integrity.** Existing Kaggle synthetic hardware reports cannot certify real model weights. Tighten smoke tests so a callable result plus NVML reading is insufficient: require real checkpoint provenance, inference evidence, output validation and known engine identity.

**Exit gate Phase 2:** All critical preflight checks pass. Failures are explicit and attributable to request, backend, hardware or execution. No silent mock result, false ready state or incorrect artifact success.

## 6. Phase 3 — Unified Kaggle benchmark

### 6.1. Benchmark modes

| Mode | Goal | Execution |
|---|---|---|
| `smoke` | Verify feasibility | A small authentic inference per feasible model/variant |
| `full` | Verify capability correctness | Every declared supported task/variant, including API -> queue -> worker -> native backend -> output |
| `perf` | Measure performance | Reproducible runs for every passing supported task; repeat light cases and report median/variability |

The benchmark should be **model-oriented, not merely a 163-case legacy contract matrix**. Preserve useful cases from `benchmarks/configs/evaluation_cases.yaml`, but replace historical duplicates with concrete model/capability coverage and robust input/output validation.

### 6.2. Mandatory model coverage (11 manifests)

| Manifest/model | Supported feature groups to benchmark | Additional caution |
|---|---|---|
| Qwen Text | Chat, structured response, tool calls, streaming | Valid JSON/schema and token stream |
| Qwen VLM | Image QA, video understanding, document/image grounding | Real multimodal asset ingestion |
| Qwen Decision | Predicate, choice, scoring/ranking | Valid decision schema, no invented probabilities |
| Qwen Embedding + Reranker | Text embedding, semantic similarity, reranking | Correct vectors/dimensions; multimodal embedding not currently supported |
| Qwen Speech | ASR, TTS, voice cloning | Forced alignment/timestamps currently unsupported in manifest |
| Qwen-Image 2.1 | T2I, I2I, inpainting, outpainting, references, pose/depth, variants | Verify individual workflow graph and GGUF checkpoints |
| Wan 2.2 | T2V, I2V, first/last frame, extension | Respect GPU VRAM and output video continuity |
| MiniMax H3 | T2V, I2V, first/last frame, audio/video, reference | Test exact variant; audio mux and duration |
| ACE-Step 1.5 | Music generation, lyrics, reference, cover, repaint, extract, lego, complete, extend | Check variant-specific capability gates |
| YuE2 | Text/lyrics music generation, symbolic score, score edits and variations | Confirm actual checkpoint identity and hardware feasibility |
| Enhancement Suite | Image/video SR, frame interpolation, audio enhancement | Manifest/backend currently unavailable: record an explicit blocker until real engines exist |

For every model, enumerate task–variant pairs from the authoritative manifest, include both positive and negative cases (unsupported task must reject), and do not silently omit a pair due to resource constraints.

**Dual-T4 constraint:** Two 16 GB T4 devices do not form one shared 32 GB heap. Benchmark only documented sharding/offload/quantization configurations. If impossible within available hardware, mark `HARDWARE_BLOCKED` and specify the estimate and actual resource limit.

### 6.3. Runtime path

```text
Kaggle bootstrap + secrets
      -> detect GPU / software versions / disk / CUDA
      -> validate manifest + checkpoint revision/hash
      -> start native backend + Factory control/worker
      -> submit real Factory API job
      -> await job, capture progress and worker logs
      -> validate bytes + media/task semantics
      -> capture NVML time series, latency and errors
      -> save JSON/CSV/evidence + report
```

Avoid accepting a synthetic direct Python callable as full Factory E2E evidence. If a standalone model proof is useful, classify it separately from `FACTORY_E2E_PASS`.

### 6.4. Metrics

**All models:** startup/load time, end-to-end latency, inference duration, GPU utilization, sampled peak VRAM, system RAM, throughput, success/error rate, artifact integrity, model/checkpoint provenance and GPU profile.

**By modality:**

- **Text / Decision:** TTFT, tokens/second, structured-output validity, answer/decision correctness.
- **VLM:** correctness on small labeled QA/grounding sets; actual media input path.
- **Embedding / Rerank:** vector dimension/norm, similarity ordering, ranking behavior.
- **ASR / TTS:** ASR WER on available labeled sample; real-time factor, valid audio, duration and listening spot checks.
- **Music:** valid audio, duration, prompt/lyrics adherence, meaningful listening rubric with retained samples.
- **Image:** seconds per image, dimensions, decode validity, visual prompt/edit fidelity with saved examples.
- **Video:** seconds per output second, frame count/FPS, continuity, playable output, sound synchronization where supported.
- **Enhancement:** input/output size and validity, FPS/resolution changes, quality comparison against baseline where available.

Sample VRAM over the whole execution window: never claim peak VRAM from a single post-inference snapshot or a manifest estimate. For costly long-video/music tasks, start with at least one true inference; make further repetitions conditional on budget and report the number of repetitions honestly.

### 6.5. Status semantics

| Status | Meaning |
|---|---|
| `PASS` | Actual checkpoint/backend ran, output passes validation and evidence exists |
| `FAIL` | Execution completed but correctness, contract or artifact validation failed |
| `ERROR` | Unexpected runtime/infra failure |
| `HARDWARE_BLOCKED` | Insufficient/incompatible GPU, memory or hardware configuration |
| `BACKEND_UNAVAILABLE` | Missing checkpoint, engine or dependency |
| `NOT_IMPLEMENTED` | No complete implementation exists |
| `NOT_RUN` | Case not executed; must include reason |

Do not misclassify `HARDWARE_BLOCKED` as passed, or count contract-only tests or synthetic tensor benchmarks as live model inference.

### 6.6. Evidence schema and deliverables

Each case should record:

```yaml
run_id: unique-run-id
git_commit: exact-commit
hardware: {gpu_names: [], gpu_count: 2, cuda_version: null}
model_id: manifest-model-id
checkpoint: {repo: null, revision: null, hash: null}
variant: default
task: task-id
backend: native-engine-name
mode: full
status: NOT_RUN
input_reference: fixture-or-hash
load_time_ms: null
inference_time_ms: null
e2e_latency_ms: null
peak_vram_mb: null
output: {path: null, mime: null, sha256: null}
evidence: {logs: null, nvml_trace: null}
error: null
```

Output structure:

```text
benchmarks/
├── cases.yaml
├── runner.py
├── validators.py
└── report.py

Kaggle Output/
└── run_<timestamp>/
    ├── results.json
    ├── metrics.csv
    ├── summary.md
    ├── logs/
    └── artifacts/
        ├── text/
        ├── image/
        ├── video/
        ├── speech/
        └── music/
```

Store large media in Kaggle Outputs/Dataset or a separate artifact location. Keep only verified summaries and reproducible scripts in the production repository.

**Exit gate Phase 3:** 100% of declared supported model/task/variant combinations have a real result or an explicit, evidence-based blocking status. Each `PASS` has true inference evidence.

## 7. Phase 4 — Fix, retest, harden, release

1. Triage benchmark results into correctness, configuration/dependency, GPU feasibility, native backend and artifact failures.
2. Fix correctness and unsafe behavior first, then runtime integration, then performance issues.
3. Retest affected tasks plus a minimal regression sample of previously passing features.
4. Update model manifests so unsupported/unverified features fail closed, not displayed as available.
5. Align Studio Web, SDK, README and benchmark docs with actual capabilities.
6. Remove superseded scripts, old reports and temporary outputs only after their important evidence is preserved.
7. Produce one reproducible deploy/benchmark path and a clean release tag.

Do not replace models or redesign infrastructure solely because inference is slow; first investigate cold start, checkpoint/quantization, memory placement, backend launch flags, I/O and offload.

## 8. Final acceptance checklist

- [ ] A clean production branch, baseline recovery tag and reviewed deletion list exist.
- [ ] Redundant CI, historical documents, legacy code and Mac-centric test suites are removed or archived.
- [ ] Minimal core/security preflight exists and passes on the target environment.
- [ ] Factory API, Studio and necessary workers launch with one reproducible configuration.
- [ ] Auth, job lifecycle, GPU lease, artifact delivery and failure semantics are correct.
- [ ] No fabricated inference output or telemetry is represented as model success.
- [ ] All 11 model manifests have a benchmark status and task-level/variant-level coverage.
- [ ] Every `PASS` has an identifiable checkpoint, real backend execution, valid output and stored evidence.
- [ ] All blocked/unimplemented cases have clear reasons; nothing is silently skipped.
- [ ] GPU/latency measurements are real, sampled and traceable to a hardware profile.
- [ ] `README.md` and `BENCHMARK.md` are concise and agree with the implementation.
- [ ] Kaggle benchmark can be rerun from the same commit, config and assets.

**Definition of done:** Factory is small enough to maintain as a personal production system and honest enough to identify exactly which models/features work, which do not, and why. A working API/worker path and credible live Kaggle evidence matter more than a large test count or a “100% green” headline.

## 9. Execution order

```text
Freeze baseline
  -> Inventory and dependency map
  -> Simplify source and deployment
  -> Correctness hardening + minimal preflight
  -> Kaggle smoke: feasibility by model/variant
  -> Kaggle full: all supported capabilities
  -> Kaggle perf: latency/VRAM/quality
  -> Fix failures and targeted reruns
  -> Update registry/docs and release
```

### Out of scope

New model acquisition, broad model-to-model competition, fine-tuning, multi-tenant enterprise infrastructure, Kubernetes, distributed serving, extensive production observability stacks and premature GPU optimization.

---

**Key source folders in baseline:** `factory/apps/control-api/`, `factory/apps/studio-web/`, `services/`, `workflows/`, `manifests/models/`, `benchmarks/`, `deployments/kaggle/`, `tests/`. The present document is a proposed implementation plan based on the inspected repository, not a completed implementation report.
