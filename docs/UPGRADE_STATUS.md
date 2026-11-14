# Factory upgrade status

Authority: [MASTER_PLAN.md](MASTER_PLAN.md), starting from main commit `e74f674fbfb6876b7b5899a7b64458801e3a683d`. This file records progress; it does not replace the plan. Updated 2026-11-14 (Asia/Ho_Chi_Minh).

## Phase 0 — Truth Baseline: PASS

- Frozen [baseline.json](../validation/baseline.json): 12 models, 69 model/capability pairs (52 BLOCKED, 17 DISABLED), original manifest claims, checkpoint/backend/GPU requirements, 82 static route declarations, 86 test/conftest files, test configuration, and 18 historical evidence files. All snapshots refer to the immutable starting commit and include source hashes.
- Generated [baseline_matrix.md](../validation/baseline_matrix.md) describes implementation observations and evidence decisions. **Zero pairs have admissible V3 + V4 + semantic certification.** BLOCKED records absent proof; it does not report a measured inference failure.
- Supplemental [local evidence review](../validation/local_evidence_audit.json) records three previously ignored reports and two WAV hashes. The stored Qwen/XTTS WAVs correlate above 0.99999999 with their runner's sine-wave formulas (RMSE about 0.0000126), confirming synthetic output within PCM quantization error. Historical native visual outputs remain candidates, not V3/V4 evidence.
- Revoked model/capability `live_verified` claims, evidence run references and unproven validated hardware profiles. Qwen Speech requires particular care: the historical audio runner emits sine-wave WAVs and unrelated CUDA allocations on backend failure; XTTS cannot certify Qwen voice cloning.
- Removed schema behavior that promoted all capabilities and installation/backend/checkpoint flags from a model-wide verification label. The dual-GPU reservation test now declares its fixture readiness explicitly and is marked `contract_mock`.
- Software verification: default suite **1001 passed**, service/workflow suites **65 passed**. Frozen pre-change default collection/execution: **992 tests**. These are software checks only; no GPU inference or official-SDK acceptance was executed.
- No engine restructuring, request routing redesign, Studio migration or large cleanup was performed.

Reproduce the baseline without running a backend:

```bash
.venv/bin/python benchmarks/truth_baseline.py --check
.venv/bin/python -m pytest -q -o addopts=''
.venv/bin/python -m pytest services workflows -q -o addopts=''
```

The baseline is a migration audit, not a second runtime catalog. Its code observations are intentionally frozen. ModelManifest remains the intended runtime source of truth.

## Phase 1 — Canonical Model Truth: PASS

Exit Gate: [model_truth_gate.json](../validation/model_truth_gate.json). Every one of **20 canonical models** can be inspected through one command, including exact checkpoint files, revision, SHA-256/size, engine, adapter/backend, declared capabilities, parameters, aliases, runtime requirements and validation disposition. The gate compares the complete current pinned checkpoint graph against the upstream probe; a stale checkpoint identity or incomplete graph fails. Capability/parameter migrations may reuse the unchanged graph observation. The gate records the original observed manifest hash separately from the current manifest hash.

```bash
factory models list
factory models inspect qwen-text
factory models inspect minimax-h3-turbo
factory models inspect qwen-asr
```

The CLI reads the ModelManifest catalog directly. `canonical_truth.complete=true` describes complete static identity; `current_session.attested=false` and `production_ready=false` remain explicit. Unknown identities fail. The retired composite `qwen-speech` reports explicit ASR/TTS replacements and does not choose one automatically.

- All production model YAML is now `factory.ai/v2`. Its complete `weights.files` graph owns immutable provider/repository/revision/path/SHA-256/size and destination identities. Legacy reader fields are derived; conflicting fields, checkpoint variants and static readiness claims fail validation.
- Different checkpoint bundles have different IDs: Standard/Turbo/Uncensored Image; MiniMax base/Turbo/Reference; ACE-Step Base/SFT/Turbo/XL Base; ASR/TTS; Embedding/Reranker. No different-checkpoint `task_variant_matrix` remains in the canonical catalog. The old composite speech identity is recorded under `manifests/retired` for migration.
- Model aliases and families, legacy service model bindings, dataset preparation and bootstrap checkpoint specifications now read the shared catalog. Qwen Text selects the actual Qwen3.8 GGUF with llama.cpp; it cannot silently become Qwen2.5 AWQ/vLLM. New identity/alias tests add a model by manifest alone.
- The [current upstream probe](../validation/checkpoint_metadata_current.json) resolves **20/20 primary identities and 227/227 bundle file identities**. LFS checks compare upstream SHA/size declarations; small configuration/tokenizer files are content-hashed. Neural weights were not downloaded by this probe. Transient HTTP errors retry only the identical source; unresolved errors fail the probe.
- Bootstrap and generated dataset downloaders verify exact SHA-256 and byte size. File existence, a Kaggle dataset slug or API health cannot claim inference verification. Existing datasets are not accepted as interchangeable checkpoints. Canonical bootstrap explicitly refuses legacy hardware/variant heuristics and health-only readiness; active model/backend attestation remains Phase 3/4 work.
- Current catalog dispositions: **17 EXPERIMENTAL, 1 BLOCKED (YuE2), 2 DISABLED (Uncensored Image and Enhancement)**. All capability live verification and deployment-readiness flags remain false. At the Phase 1 exit, native inference and Full Factory E2E had not run; the later isolated Enhancement inference probe is recorded under Phase 5.

### Recovered sources and hardware limits

The initial [metadata probe](../validation/checkpoint_metadata_probe.json) is preserved: all 12 original declared pins failed anonymous resolution (nine invalid revisions and three HTTP 401). A 401 alone cannot distinguish private/gated/nonexistent sources. Source recovery is recorded separately in [checkpoint_sources.json](../validation/checkpoint_sources.json).

The completed [Kaggle CPU probe](https://www.kaggle.com/code/cuongnguyen1802/factory-phase-1-weight-identity) hashed mounted files with GPU and internet disabled. Its immutable [observation](../validation/weight_identity_observation.json) and [subsequent source comparison](../validation/weight_identity_comparison.json) remain separate. Five MiniMax hashes match the pinned Comfy-Org bundle. Three image files were originally `OBSERVED_UNBOUND`, so that original run deliberately exited 1. Later source lookup identified the old Standard dataset's primary GGUF as Uncensored weights. The intended Unsloth Standard and the disabled Uncensored model now have different IDs and hashes; the old dataset cannot pass Standard integrity checks.

YuE2 now points to the real `m-a-p/YuE2-3B` plus `YuE2-Vae`, rather than YuE1. Its selected native BF16 runtime is incompatible with T4 and remains **BLOCKED**. Proposed Phase 4/5 work: validate an explicit compatible runtime/precision on the same weights, or declare a separately identified quantized checkpoint and select it through CLI. No such alternative or automatic fallback has been applied.

The legacy Text and Speech launchers reject their mismatched backend selections; Music rejects the v2 bundle until its explicit lifecycle is implemented. This prevents a legacy default from silently choosing a different backend or ACE-Step Turbo checkpoint. Their source remains for the ordered Phase 4 migration.

### Reproduction and software validation

```bash
.venv/bin/python benchmarks/checkpoint_probe.py --output validation/checkpoint_metadata_current.json
.venv/bin/python benchmarks/model_truth_gate.py
.venv/bin/factory-manifest validate
.venv/bin/python benchmarks/truth_baseline.py --check
.venv/bin/python -m pytest -q -o addopts=''
.venv/bin/python -m pytest services workflows -q -o addopts=''
```

The gate verifies static identity and complete inspection. Current software checks: **1,025 default tests + 65 service/workflow tests PASS**, manifest validation, frozen-baseline reproduction and diff checks PASS. Counts and report hashes are recorded in [phase_status.json](../validation/phase_status.json); tests marked `contract_mock` use explicit legacy fixtures with simulated readiness. They provide no model inference evidence. No architecture alternative to the Master Plan has been applied.

## Phase 2 — Capability Canonicalization: PASS

Exit Gate: [capability_gate.json](../validation/capability_gate.json). One immutable [registry](../packages/contracts/src/factory_contracts/capabilities.py) owns **55 canonical capability IDs and request aliases** across the eight target engines. ModelManifest continues to own which capabilities each exact model supports; registry membership never grants implementation or readiness.

- Migrated all 20 model capability dictionaries, their implementation-truth projections and capability-dependent parameter requirements to canonical IDs. The 12 existing capability parameter specifications and service task configuration now reference that vocabulary. The v2 schema rejects noncanonical or wrong-engine capability declarations.
- `JobSubmissionRequest`, worker registration, legacy catalog/config readers and persisted-job deserialization normalize their ingress data. Queue entities, admission, lease queries, timeout policy, worker advertisements, workflow bindings and music/decision adapters use canonical IDs. Leasing no longer expands task aliases.
- Removed independently maintained alias maps from admission, leasing, workflow registries and music adapters. Compatibility imports expose derived read-only views of the central registry. Native adapter bindings translate canonical IDs into backend protocol operations; they do not accept request aliases.
- `t2i`, `image.t2i` and `text2image` normalize to `image.text_to_image` before persistence/dispatch. Ambiguous short names such as `reference`, `edit` and `score` require service/engine context. Image, Video and Music reference capabilities stay distinct; multiple image references become `image.multi_reference` at ingress. Pose and depth remain separate capabilities, and an explicit control capability cannot be changed by a conflicting `control_type` parameter.
- Contract checks cover alias request → persistence → canonical lease query, no repeated alias resolution for canonical jobs, ambiguous contexts, multiple references, pose/depth isolation, native operation bindings and rejection of alias handling inside workflow dispatch. These tests do not execute inference.
- Static audit of **186 runtime Python sources** finds zero independent capability alias tables or alias resolution in dispatch. It also checks canonical model/config/parameter-spec IDs. The unchanged **20 checkpoint graphs / 227 pinned file identities** continue to pass the Phase 1 gate using the original upstream observation; this phase did not re-download or re-certify weights.
- Software checks: **1,089 default tests + 65 service/workflow tests PASS**. Manifest validation, Phase 0 baseline reproduction and diff checks PASS. Counts and report hashes are in [phase_status.json](../validation/phase_status.json). There is still no native GPU, semantic or Full Factory E2E certification.

```bash
.venv/bin/python benchmarks/capability_gate.py
.venv/bin/python benchmarks/model_truth_gate.py
.venv/bin/factory-manifest validate
.venv/bin/python benchmarks/truth_baseline.py --check
.venv/bin/python -m pytest -q -o addopts=''
.venv/bin/python -m pytest services workflows -q -o addopts=''
```

## Phase 3 — Deployment Registry + CLI: PASS (configuration exit gate)

Exit Gate: [deployment_gate.json](../validation/deployment_gate.json). Both required layouts are expressed and validated through YAML/CLI without modifying Python:

- [Shared layout](../deployments/examples/dual-t4-shared.yaml): GPU0 hosts Qwen Decision + Embedding; GPU1 hosts Reranker. The summed manifest requirements are 10 GB and 4 GB respectively, checked independently against each GPU's 14.2 GB budget. These are metadata estimates, not measured residency or an OOM guarantee.
- [Exclusive layout](../deployments/examples/dual-t4-exclusive.yaml): Qwen Text explicitly selects llama.cpp on devices `[0, 1]`, with exclusive access. Its declared per-device requirement is 11 GB. No 32 GB pooled-memory assumption is used.

### Implemented controls

`ModelCatalog` owns the frozen catalog snapshot; `DeploymentConfig` owns explicit model/backend/device/port/weight-root configuration; `DeploymentRegistry` owns immutable session bindings; `DeploymentValidator` separates configuration plausibility from current-host preflight. GPU IDs are mandatory, aliases resolve before configuration is frozen, and different model IDs cannot alias into duplicate deployments.

```bash
factory --deployment session.yaml deployment init --models-dir /kaggle/working/models
factory --deployment session.yaml deployment add qwen-decision --gpu 0
factory --deployment session.yaml deployment add qwen-embedding --gpu 0
factory --deployment session.yaml deployment add qwen-reranker --gpu 1
factory --deployment session.yaml deployment show
factory --deployment session.yaml deployment validate --config-only
factory --deployment session.yaml deployment validate
factory --deployment session.yaml serve --check
factory --deployment session.yaml status
factory --deployment session.yaml status --live
factory --deployment session.yaml deployment remove qwen-reranker
```

`deployment add` freezes one backend implementation, port and explicit GPU list before serving. A rejected mutation leaves the existing file unchanged. `deployment init` refuses overwrite. `show` and offline `status` are configuration views; `status --live` queries the authenticated local session and never forwards credentials through redirects.

- Configuration validation checks canonical models/capabilities, model disposition, selected backend, GPU IDs/count, per-device aggregate VRAM plausibility, API/backend port conflicts and exclusive-device conflicts. BLOCKED/DISABLED/DEPRECATED entries cannot pass.
- Full local preflight additionally checks current-host Dual-T4 inventory, loadable selected backend/adapter providers, exact SHA-256/size for every pinned file and local port availability. It never infers runtime load or semantic inference from those checks. Adapter/backend provider discovery uses package entry points rather than a second model-specific routing table.
- Removed `SUBSYSTEM_GPU_MAP` and subsystem/default-device routing from production scheduling. Submission and retry bind a job to the same deployment fingerprint, session ID and device list. Leasing reserves the configured devices, rejects worker attempts to choose another GPU, and reserves all devices atomically for a dual-GPU model. Request inputs cannot choose a backend/GPU. Retry/idempotency cannot silently change session, model or capability.
- Canonical models cannot use the legacy runtime's load probe to become READY or trigger automatic model swapping. Legacy runtime contracts cannot authorize production readiness. Legacy queue fixtures retain explicit test-only bindings for migration; they are not valid co-resident production deployments.
- `/v1/models` now returns only the active-session discovery envelope; the full catalog is separate at `/v1/registry/models`. Deployment status is `/v1/registry/deployment`. No configured model is automatically public. The envelope was checked against [official OpenAI documentation](https://developers.openai.com/api/reference/resources/models/methods/list); official SDK acceptance and the rest of OpenAI compatibility remain Phase 6 work.

### Runtime boundary and evidence

At the Phase 3 exit, backend launch/health remained pending and plain `serve` stopped fail-closed after preflight. Phase 4 below replaces that boundary with owned backend lifecycle. `serve --check` still performs preflight without starting processes. There is no boolean/YAML switch that promotes a configured model to active.

The [actual local environment report](../validation/deployment_environment.json) has `preflight_passed=false` (CLI exit 2): this host has no `nvidia-smi`/Dual-T4 inventory, canonical provider registrations or configured example weight store. This is a local-host observation, not a failed Kaggle inference attempt. No new Kaggle GPU session was launched during this phase. The deployment configuration gate passes independently; it does not represent running backends or production readiness.

Software validation: **1,114 default tests + 65 service/workflow tests PASS**. Counts and report hashes are recorded in [phase_status.json](../validation/phase_status.json). Tests cover configuration edits, conflicts, same-size corrupted files, missing environment/providers, canonical API visibility, configured GPU routing, atomic dual-GPU reservation, session replay rejection, idempotency model isolation and startup refusal. Mocked inventory/tiny file fixtures are software contracts only, never checkpoint/model/GPU evidence.

```bash
.venv/bin/python benchmarks/deployment_gate.py
.venv/bin/python benchmarks/capability_gate.py
.venv/bin/python benchmarks/model_truth_gate.py
.venv/bin/python benchmarks/truth_baseline.py --check
.venv/bin/factory-manifest validate
.venv/bin/python -m pytest -q -o addopts=''
.venv/bin/python -m pytest services workflows -q -o addopts=''
```

## Phase 4 — Backend Lifecycle: PASS (lifecycle only)

Exit Gate: [backend_lifecycle_gate.json](../validation/backend_lifecycle_gate.json), **PASS** against the current source archive and kernel hash. The [v10 observation](../validation/backend_lifecycle_observation_v10.json) and [collection receipt](../validation/backend_lifecycle_receipt_v10.json) bind the current-source Kaggle regression. Earlier successful v5–v9 observations are retained.

- Added `Backend.prepare / launch / health / stop / runtime_info`, eight explicitly selected backend provider registrations and thirteen launch adapters. vLLM, llama.cpp, ComfyUI, vLLM-Omni and native SDK launch paths are implemented. Availability and launch bindings are not claims that every provider/model was executed or that inference adapters are complete.
- `factory serve` carries the exact compiled registry/preflight into the API lifespan. The API listens only after selected backends finish startup checks. Failed backends are isolated, stopped and excluded from healthy session status; no restart, backend swap or checkpoint fallback occurs. If every configured backend fails, API startup fails.
- Each backend sees a private symlink view containing only the manifest-pinned files. Preparation rehashes every file; runtime checks reject file identity changes. CUDA visibility is fixed from the deployment devices. Health requires an owned process/listener, selected model identity and observed GPU processes on exactly the assigned physical devices. Actual resident memory is checked per GPU against the hardware budget after load.
- Shutdown and startup cancellation stop owned process groups, including stubborn children; API resources are cleaned up on startup failure too. Runtime status records PID/birth, backend version or source revision, executable hash, checkpoint graph, devices and GPU memory. Internal health never emits model inference output.
- Healthy backend residency is distinct from public authorization. As required by Master Plan §§4.3/22, `/v1/models` remains empty until evidence authorizes READY capabilities. Canonical job admission and production Chat/Responses reject unvalidated models before reaching a legacy upstream. No model/capability is production-ready.

### Live witness and debugging record

The private [Kaggle lifecycle kernel](https://www.kaggle.com/code/cuongnguyen1802/factory-phase-4-backend-lifecycle) explicitly requests `NvidiaTeslaT4` and checks the observed inventory for exactly two Tesla T4 devices. It downloads the original manifest-pinned Qwen3-ASR-1.7B bundle, then runs the actual `factory serve` command with GPU0 selected. All **10 files / 4,703,055,333 bytes** matched SHA-256 and size in runs 2–10. These are exact-weight observations, not inference evidence.

Failed observations are retained without rewriting them: [v1](../validation/backend_lifecycle_observation_v1.json) exposed a bootstrap `sys.path` ordering bug (`queue` shadowed the standard library); [v2](../validation/backend_lifecycle_observation_v2.json) exposed missing `services` modules in the source archive; [v3](../validation/backend_lifecycle_observation_v3.json) reached lifecycle startup but no backend became healthy. Its environment had psutil 5.9.5, incompatible with the `Process.net_connections()` API used here; the dependency minimum is now 6.0, as documented in the [psutil migration guide](https://psutil.io/migration/#migrating-to-6-0). Startup failures now log the selected provider's error. No architecture or model substitution was made.

The source archive is deterministic and includes the API's retained Text/VLM imports. A separate isolated-directory import check passed before the rerun. Collection keeps immutable versioned observations and artifact SHA-256 receipts. The gate checks current source/kernel hashes, all checkpoint files, actual runtime identity/GPU residency, public rejection, and graceful cleanup. A stale run cannot satisfy the current gate.

```bash
.venv/bin/python -m deployments.kaggle.backend_lifecycle_probe --run
.venv/bin/python -m deployments.kaggle.backend_lifecycle_probe --collect
.venv/bin/python benchmarks/backend_lifecycle_gate.py
```

The witness deliberately has no inference operation: `native_inference_executed=false`, `semantic_validation_executed=false`, `full_factory_e2e_executed=false`, `production_ready=false`. The successful lifecycle gate did not change these flags. The old Qwen Text dataset has different bytes from the canonical UD-Q4_K_M checkpoint and was not reused as a substitute.

### Completed real lifecycle witness

Versions 5–10 completed successfully on two Tesla T4 devices. Version 8 covered the final Phase 5 changes, version 9 the Phase 6 API implementation, and version 10 the Phase 7 core simplification with the current archive hash. `factory serve` loaded the selected `qwen-asr` backend on physical GPU0. The original SDK exposes **2,038,052,480 actual parameters**, all on `cuda:0`, all FP16; nvidia-smi attributed **4,638 MiB** to the owned backend PID. Exact checkpoint identity, current-session nonce, authenticated health, listener/process ownership and per-GPU runtime budget checks passed. GPU1 was not assigned to this model.

The API returned `{"object":"list","data":[]}` and rejected both unvalidated Chat/Responses calls with 503. Shutdown recorded the matching session/config fingerprint, every backend `STOPPED`, no cleanup errors and no surviving backend PID. Uvicorn returns `-15` after handling SIGTERM in this environment; the gate requires the cleanup callback's actual receipt, rather than inferring graceful shutdown from that exit code alone. [Version 4](../validation/backend_lifecycle_observation_v4.json) is retained with its original NOT_MET result for the earlier exit-code-only probe.

Runtime versions: qwen-asr 0.0.6, Torch 2.11.0+cu130, Transformers 4.57.6, huggingface-hub 0.36.2, psutil 7.2.2, FastAPI 0.141.1 and Uvicorn 0.54.0. This validates this selected backend/checkpoint's **load and lifecycle only**; the other provider launch plans have no equivalent live proof yet.

Software checks: **1,128 default tests + 65 service/workflow tests PASS**. Manifest validation, frozen baseline, Phase 1/2/3 gates, dependency compatibility, compilation and diff checks passed. CPU process tests and mocked SDK metadata remain explicitly software-only. No audio was submitted, transcribed or semantically validated.

## Phase 5 — Engine Refactor: PASS

[Exit Gate](../validation/engine_refactor_gate.json) passes for all eight generic engines and all twenty Manifest-selected model entries. The executable gate reruns **eight explicit mock extensibility contracts**: each adds a new Manifest identity plus Adapter module, invokes validation/preparation/inference-hook/normalization through its Engine and verifies the Engine source is unchanged. This satisfies the Phase 5 gate; those contracts provide no model inference certification. Phase 0–4 regression gates also pass against the current source.

| Engine | Adapter responsibility and implemented scope |
|---|---|
| Vector | Separate Qwen embedding/reranker adapters own vLLM requests, dimensions, native scores and output validation. No fabricated vectors/usage/ranking. |
| Decision | Qwen adapter owns native typed questions/answers, refusals, genuine usage and separately labelled numeric score conversion. No fabricated calibration metrics. |
| Responses | Text and VLM use one Engine/request contract. Separate adapters own llama.cpp/vLLM protocol, inline multimodal input, tools/structured/stream capability checks and strict native output/SSE validation. |
| Speech | Qwen ASR owns real WAV decoding and SDK inference. Qwen TTS Base owns vLLM-Omni reference-audio cloning; reference audio plus transcript are required, with no preset-voice checkpoint fallback. |
| Image | Qwen adapter owns exact GGUF/CLIP/VAE workflow, 64-channel input latent node, editing/reference/mask graphs and native output retrieval. Missing nodes or unpinned selectors fail. |
| Video | MiniMax owns AV conditioning, frame grid, exact Turbo LoRA and native MP4 graphs. Wan owns SDK text/image wrappers over the same exact components, first/last frame conditioning and actual extension concatenation. |
| Music | ACE-Step owns native GenerationParams/task instructions and selected Base/SFT/Turbo/XL bundles. YuE2 owns its actual SDK Pipeline/VAE/ABC contract; selected BF16 runtime remains BLOCKED on T4. |
| Enhancement | Actual ComfyUI RealESRGAN x4plus graph and decoded image output. The suite remains DISABLED; absent RIFE/audio/video-SR checkpoints are unsupported. |

The shared `ModelAdapter` interface owns request validation, preparation, inference, normalization and health. Engines select only the Manifest adapter and the precompiled deployment binding. Exact-session transport fixes localhost port/model/private credential, rejects redirects/environment proxies and refuses request-level model/backend/device/checkpoint changes. Shared per-device locks serialize co-resident engine execution. Public execution still requires registry evidence authorization before invoking any adapter.

Declared supported capabilities now have static `implemented_capabilities` metadata for their code paths. This does not assert that every path has run against its SDK or GPU: **all live verification and production-readiness flags remain false**, with **17 EXPERIMENTAL / 1 BLOCKED / 2 DISABLED** models. All **20 checkpoint graphs / 227 pinned identities** remain unchanged. Only the Enhancement path below has new native inference evidence. Exact-checkpoint V3, semantic quality and Full Factory E2E remain Phase 8/9 work.

### Actual Enhancement backend witness

The private [Kaggle kernel](https://www.kaggle.com/code/cuongnguyen1802/factory-phase-5-enhancement-native), [v6 observation](../validation/enhancement_observation_v6.json) and [artifact receipt](../validation/enhancement_receipt_v6.json) bind the current source archive to a real backend execution on the observed Dual-T4 host, statically assigned GPU0. This isolated internal Adapter probe does not activate the disabled model through `factory serve` or bypass public Engine authorization.

- Exact Manifest checkpoint: `RealESRGAN_x4plus.safetensors`, 66,857,836 bytes, SHA-256 `37f9a931c215f040aa6d50f711f2cb115f713c46df1d0d6469a8bd7bfe9a60bb`. ComfyUI revision `c5751c81abe131202eea9e4c66d000589013dd0b`.
- Input is the pinned scikit-image v0.25.2 astronaut photograph, **512×512**, with recorded source/hash. The executed `UpscaleModelLoader → ImageUpscaleWithModel → SaveImage` graph and actual native history prompt ID are stored in the observation.
- Output is an actual decoded **2048×2048** PNG, with a different hash, matching collection receipt and byte count. Owned GPU0 telemetry reached **4,526 MiB** during inference; process shutdown was verified. GPU1 was not assigned.
- `native_inference_executed=true` applies to this image-upscaling probe only. `v3_certified=false`, `semantic_quality_certified=false`, `full_factory_e2e_executed=false`, `production_ready=false`, public models `[]` and model state `DISABLED` remain explicit. Dimensions/hash/GPU telemetry alone do not establish semantic quality.

The original [failed v1 observation](../validation/enhancement_observation_v1.json) is retained: ComfyUI needed the isolated base `custom_nodes` directory created before startup. That launch fix is included in the successful v2/v3/v4 runs; v4 follows the final Phase 5 Manifest implementation metadata and production source; v5 repeats the same exact-checkpoint inference after Phase 6 changes; v6 repeats it after Phase 7 with the current source hash. Earlier successful artifacts are not rewritten. The current-source ASR lifecycle rerun v10 also passes, with actual SDK parameters on CUDA/FP16 but no ASR transcription attempt.

### Software checks and retained boundaries

**1,234 default tests + 65 service/workflow tests PASS**, including 21 Vector contracts, 52 Decision contracts, 25 Responses/media contracts and 8 cross-engine extensibility contracts. Manifest validation, compilation, frozen-baseline reproduction, Phase 1–5 gates and diff checks pass. Software results and JUnit hashes are recorded in [phase_status.json](../validation/phase_status.json). Five [evidence-validator tamper checks](../validation/engine_gate_adversarial.json) reject stale source, wrong checkpoint, production promotion, fabricated dimensions and corrupted output even with matching edited receipt digests. Historical mocked tests explicitly use retained Legacy classes/fixtures; positive mock results are software evidence only.

Text/VLM service imports and the control API now share ResponsesEngine. Legacy media workers/templates and proxy sources remain for the ordered migration; production legacy Speech/Enhancement endpoints fail closed. Remote media ingestion, public API dispatch/error normalization and official SDK acceptance were subsequently implemented in Phase 6 below. No OpenAI-hosted inference or official SDK acceptance ran in Phase 5.

[PLAN_FINDINGS.md](PLAN_FINDINGS.md) records the Enhancement checkpoint limit, TTS Base reference requirement, corrected image/video/SDK assumptions, YuE2/T4 block and the newly official OpenAI Decisions API. The Phase 6 Decisions contract must follow the official API under the existing OpenAI-first rule. No architecture or phase-order change was silently applied.

```bash
.venv/bin/python -m pytest tests/contract/test_engine_extensibility.py tests/contract/test_phase5_response_and_media.py -q -o addopts=''
.venv/bin/python benchmarks/backend_lifecycle_gate.py
.venv/bin/python benchmarks/engine_refactor_gate.py
.venv/bin/python -m deployments.kaggle.enhancement_probe --collect
```

The native gate requires the collected PNG bytes matching the receipt. Historical binary outputs stay in local validation storage/Kaggle output rather than being committed in bulk. A source/kernel change requires a new current-source witness; stale evidence fails the gate.

## Phase 6 — OpenAI-Compatible API Layer: PASS

[Exit Gate](../validation/openai_api_gate.json) passes using the **unmodified official Python SDK 3.49.0** against an actual localhost HTTP server mounting the production routers. The gate reruns **54 SDK/native protocol contracts**, checks every Master Plan route, re-evaluates Phase 0–5/current-source evidence and records source/JUnit hashes. Positive model outputs and authorization are explicit `contract_mock` doubles injected only by tests. This is **API/software compatibility acceptance**, not inference certification or Full Factory E2E. No OpenAI-hosted call or OpenAI API key was used.

- Added canonical Engine dispatch for embeddings, transcription/speech, images, videos, Decisions, rerank, music and enhancements. Text/VLM share the Responses engine. Public inference accepts explicit `model`; service/backend/GPU/checkpoint overrides are rejected. Required capabilities remain Manifest-bound and evidence-authorized, with no automatic model/backend fallback.
- Chat and Responses support their native streaming contracts. Max-completion-token translation is explicit; actual native response IDs/timestamps/usage/output are preserved and malformed output fails. Preflight/first upstream event succeed before HTTP 200. Truncated/failed streams do not fabricate completion markers. Undeclared tools/structured/reasoning controls fail closed.
- Official SDK calls cover sync/async Responses, Chat, SSE, default base64 and float embeddings, multipart ASR, binary WAV TTS, image generation/edit/multiple images, video create/retrieve/content/extension, Decisions and model discovery. Request-normalization/native adapter checks cover decision question order/distributions/usage, response SSE identity/sequence/completion, invalid output limits and text-model media rejection.
- Media normalization bounds received bodies to 64 MiB, checks decoded content and uses DNS-pinned SSRF-safe fetches without redirects/environment proxies. Native adapters receive ingested bytes/inline data; they do not fetch arbitrary request URLs or paths. Unknown and unsupported options are errors, rather than silently ignored.
- Video/music/enhancement jobs use a bounded in-process facade over canonical Engines and existing scratch storage, with ownership, real state transitions, content integrity, expiration and shutdown handling. Completion requires actual validated bytes. Completed video dimensions/duration are probed from the MP4, including extension content. This facade does not dispatch through the retained legacy distributed workers.
- Errors now include the OpenAI `message/type/param/code` envelope and request-ID header; internal APIs retain their migration details. SDK tests check authentication, unvalidated sessions returning 503, parameter rejection, failed jobs without content, and cross-owner/object-family access rejection. `/v1/models` remains empty; no public boolean/config switch grants readiness.

The [API guide](OPENAI_API.md) documents the supported subset and examples. Qwen TTS Base requires reference audio/transcript and WAV output. Stored Responses, ASR timestamps/streaming, unsupported image quality/batching/output formats and unresolved file IDs fail explicitly. There is no current Manifest video-edit capability: `/v1/videos/edits` returns an unsupported error instead of changing its operation or model. Enhancement remains DISABLED, YuE2 remains BLOCKED on selected BF16/T4, and unsupported media/checkpoint combinations are not activated.

[PLAN_FINDINGS.md](PLAN_FINDINGS.md) records two upstream constraints before proceeding: the official hosted Videos API shut down on 2026-09-24, while the SDK retains its historical route contract; SDK 3.49.0 also rejects dictionary video references client-side in named edit/extend methods. Acceptance uses the unmodified supported multipart upload form; existing owned video IDs work through Factory's JSON HTTP contract and are tested separately. The Decisions endpoint follows the current official multi-question schema with the supported text-only native subset. These findings do not alter model identities or phase order.

Software regression: **1,288 default tests + 65 service/workflow tests PASS**, with no skips/errors/failures. Manifest validation, compilation, dependency compatibility, frozen baseline, all Phase 1–6 gates and diff checks pass. At the Phase 6 exit, Kaggle [ASR lifecycle v9](../validation/backend_lifecycle_observation_v9.json) and [isolated Enhancement v5](../validation/enhancement_observation_v5.json) passed against its source archive hash. Phase 7 subsequently reran the full chain on v10/v6 below. ASR remains load-only; new inference evidence remains limited to the disabled RealESRGAN backend probe, without semantic/V3/V4 authorization.

```bash
.venv/bin/python -m pytest tests/contract/test_openai_sdk_api.py tests/contract/test_phase6_native_contracts.py -q -o addopts=''
.venv/bin/python benchmarks/openai_api_gate.py
```

[phase_status.json](../validation/phase_status.json) and [openai_sdk_sources.json](../validation/openai_sdk_sources.json) preserve the executed checks, hashes, installed SDK identities and limitations. **Zero model/capability pairs are production-ready.**

## Phase 7 — Simplify Control Plane: PASS

[Exit Gate](../validation/control_plane_gate.json) passes. Core inference composition requires **no Redis/Postgres/S3**. The gate creates a fresh environment, installs only core package dependencies, runs an actual local HTTP/core job path with explicit software doubles, checks cancellation/lifecycle contracts and re-evaluates every prior phase against current-source Kaggle evidence. [CONTROL_PLANE.md](CONTROL_PLANE.md) describes the implementation and migration boundary.

- Default `factory serve`/`factory_control_api:app` now initializes only auth/catalog, explicit backend session, scratch storage and one local async job manager. Broker/scheduler, lease coordinator, worker watchdog, repository/migrations, old lifecycle/runtime managers and legacy worker/storage/Studio gateways are absent from startup and default routing.
- Core dependency declarations no longer require boto3, SQLAlchemy/Alembic/aiosqlite or JWT/crypto. Compatibility dependencies remain in `factory-control-api[legacy]` and `factory-auth[jwt]`; development tests explicitly include the legacy extra. A clean Python 3.13 core-only environment installed **30 packages**, with none of those infrastructure/JWT libraries. The [probe](../validation/core_runtime_probe.json) blocks optional imports and all network connections except its own HTTP listener, with unreachable Redis/Postgres/S3 URLs present. Actual startup/status, unvalidated rejection, canonical job plumbing, scratch integrity and shutdown all pass.
- Shared async device acquisition waits without occupying executor threads, uses stable physical-device ordering and releases partial reservations on cancellation. Sync Engines and Responses streams use the same device locks. A running cancelled job keeps its reservation until the actual native call ends and its output is discarded; repeated task cancellation cannot free the GPU early.
- Local `/v1/jobs` management now supports owner-scoped status/list/cancel/events/content. Waiting jobs cancel without entering native inference; running jobs expose cancelling; progress remains actual 0/100 milestones. Shutdown closes admission, stops owned backends, drains in-flight jobs and then stops scratch sweeping. Late admission cannot create a new job during shutdown. Content requires an owned completed job and matching bytes/hash; expired/cancelled/failed jobs have no public output.
- Former lifecycle/routes remain in `legacy.py` for explicit contract fixtures and Studio migration, and reject production startup. Historical tests import/inject that host explicitly; the production entrypoint never redirects there. No broad deletion or evidence cleanup was performed. The retained Studio route-parity test checks the migration host; the default core awaits Phase 10 Studio consumption changes.

**11 local-control contracts PASS**, covering waiting/running/repeated cancellation, partial dual-GPU locks, shutdown order, admission races, artifact integrity/TTL/ownership, backend failure, HTTP status/events and legacy production rejection. Full regression: **1,299 default tests + 65 service/workflow tests PASS**, with zero skips/failures/errors. Official SDK/native protocol acceptance remains **54 PASS**. Manifest validation, frozen baseline, compilation, dependency compatibility, diff checks and Phase 1–7 gates pass.

The actual current-source [Kaggle ASR lifecycle v10](../validation/backend_lifecycle_observation_v10.json), with its [receipt](../validation/backend_lifecycle_receipt_v10.json), reports a production core process with `external_services_required=[]`, `loaded_optional_infrastructure=[]`, `legacy_components=[]`, in-memory jobs and local scratch. Exact ASR parameters remain on assigned GPU0/FP16 and graceful shutdown passes; ASR inference was not attempted. The [isolated Enhancement v6](../validation/enhancement_observation_v6.json) with [receipt](../validation/enhancement_receipt_v6.json) repeats genuine pinned RealESRGAN inference successfully against the same current source hash. These scope-limited observations do not certify native semantics or Full Factory E2E.

```bash
.venv/bin/python -m pytest tests/contract/test_local_control_plane.py -q -o addopts=''
.venv/bin/python benchmarks/control_plane_gate.py
```

The [machine-readable status](../validation/phase_status.json) records source/JUnit hashes, minimal package inventory and current-source observations. **Zero model/capability pairs are production-ready**; public authorization remains fail-closed.

## Phase 8 — Native Certification V3: IN PROGRESS

Wave A is complete: Decision, Embedding and Rerank ran in private Kaggle vLLM attempt v3; Text ran in private llama.cpp attempt v3. The [native gate](../validation/native_wave_a_gate.json), [collection receipt](../validation/native_vllm_receipt_v3.json) and generated [model/capability matrix](../validation/native_matrix.md) record **ten V3 semantic PASS pairs across four models**. Each case ran one warmup plus three measured calls (Decision/Embedding/Rerank on explicitly assigned GPU0; Text on exclusive fixed GPU0+GPU1), with owned process/VRAM samples, complete checkpoint hash verification, actual output bytes and verified shutdown. The host inventory contains two T4 devices. No public authorization, alternate model or fake output was used.

| Capability | Semantic result | Median adapter roundtrip |
|---|---|---|
| Decision predicate | Paris/France true, Paris/Norway false | 290.25 ms |
| Decision choice | Billing request routed to billing | 262.78 ms |
| Decision score | Shattered device ranked above intact device | 282.82 ms |
| Text embedding | Paris relevance exceeds unrelated bananas by cosine margin 0.4085 | 285.85 ms |
| Reranking | Paris document ranks above Berlin and bananas | 327.66 ms |

These are small golden-case results, not general model accuracy or calibration estimates. Timings include private Adapter health checks and output validation; they are not isolated GPU kernel times. Native inference calls the actual selected Adapter/backend internally: **this is not the Full Factory public API E2E chain**.

Text [attempt v3](../validation/native_text_v3/qwen-text/observation.json) and its [receipt](../validation/native_text_receipt_v3.json) pass all five declared capabilities using the original pinned 16,564,259,744-byte 27B GGUF and llama.cpp CUDA SM75 build. Actual native answers are `42`, `FACTORY_NATIVE_OK`, JSON `{"city":"Paris"}`, a `lookup_weather` call for Paris, and streamed `FACTORY_STREAM_OK` ending with the native `[DONE]`. Measured median roundtrips are respectively 2.771 / 3.057 / 5.223 / 7.836 / 3.686 seconds. The same owned PID used both T4 devices (GPU0 8,472 MiB; GPU1 peak 9,428 MiB), and shutdown passed. Failed vLLM v1/v2 and Text v1/v2 attempts remain intact. [Technical findings](PLAN_FINDINGS.md) record the runtime fixes before rerunning: Qwen2 native Decisions strategy registration, explicit VRAM budget, embedding Matryoshka recognition and safe fixed-port reuse. No checkpoint or precision was substituted.

Software regression after the Wave B integration: **1,374 default tests + 65 service/workflow tests PASS**, with zero failures/errors/skips. Native harness checks and evidence mutation checks are software-only. Current-source ASR lifecycle v16 and isolated Enhancement v12 pass after the explicit FP32/cache policy, topology headroom reservation and precise residency failure reporting. Phase 4–7 gates pass again against that same source SHA. Lifecycle/load and the disabled Enhancement probe do not establish V3 semantic inference.

Wave B [ASR attempt v2](../validation/native_asr_v2/qwen-asr/observation.json), with its [receipt](../validation/native_asr_receipt_v2.json), passes actual transcription on assigned GPU0. The input is an 11-second recorded JFK speech fixture from pinned Whisper revision `31243bad24cc746f07d4c8bfdd2d974872cb1803`, with verified original FLAC and decoded PCM16 WAV hashes. All four native transcripts achieve **WER 0.0**, preserving the required spoken phrases; measured median Adapter roundtrip is **1.681 seconds**. This is a small ASR golden case, not general transcription accuracy or alignment certification. VLM [attempt v2](../validation/native_vlm_v2/qwen-vlm/observation.json), with its [receipt](../validation/native_vlm_receipt_v2.json), passes all five declared capabilities on GPU0: text-only arithmetic, actual orange suit, scanned title, head grounding and real pedestrian video. Grounding uses its predeclared v2 format contract and unchanged IoU threshold; measured IoU is 0.5998. Peak owned GPU0 memory is 11,808 MiB. Original v1 remains incomplete with an immutable failed-format record. TTS/voice cloning [attempt v2](../validation/native_tts_v2/qwen-tts/observation.json), with its [receipt](../validation/native_tts_receipt_v2.json), passes both declared capabilities using the original Base checkpoint and speech tokenizer, FP16 vLLM/Omni 0.32.0 on GPU0. Four actual generated WAVs per case are evaluated after native shutdown by pinned CPU Whisper and WavLM verifiers. TTS WER is 0.0 in all runs; clone WER is 0.0 except one measured 0.1111, within the predeclared 0.2 limit. Clone/reference speaker cosine is 0.8988–0.9673, above the predeclared 0.65 limit. Median native roundtrips are 4.348 and 4.843 seconds. The original failed BF16 preprocessing attempt v1 remains immutable. The [Wave B gate](../validation/native_wave_b_gate.json) passes **8/8**, bringing the total to **18 native PASS pairs in Waves A–B**.

Wave C retains three incomplete native attempts: [initial v1](../validation/native_image_receipt_v1.json), [explicit FP32 v1](../validation/native_image_fp32_receipt_v1.json), and [isolated I2I v1](../validation/native_image_i2i_receipt_v1.json). Standard and Turbo both return four semantically correct T2I PNGs in the diagnostic attempts, but these are not promoted through the current C gate. Explicit FP32 CUDA attestation works; I2I still loses the daemon between requests and its returned warmup does not satisfy the pencil/monochrome oracle. Cgroup OOM counters remain zero. All original outputs and [v1 goldens](../validation/native_golden_wave_c_v1.json) are preserved.

The current candidate keeps the same exact Q4_K_M diffusion, FP8 text encoder and BF16 VAE files, explicit FP32 diffusion, selected physical GPU0 and allowed CPU offload. Its Adapter disables optional prefix KV caching losslessly. The generic launcher now reserves the topology's **0.8 GiB** headroom, while the supervisor retains fail-closed residency enforcement and reports a precise budget violation. [I2I v2](../validation/native_image_i2i_run.json) uses the predeclared new case with existing supported `strength=1.0`; prompt/photo/seeds/semantic thresholds are unchanged. [T2I v1](../validation/native_image_t2i_run.json) reruns both Standard and Turbo under this source. T2I Standard/Turbo and I2I v2 now pass the strict native gate, each with one warmup plus three measured calls and no session failures. Median roundtrips are 177.861 / 49.065 / 350.537 seconds; peak owned GPU0 memory is 14,114 / 14,114 / 14,194 MiB. I2I saturation is 0.02494–0.03646 and reference cosine 0.8583–0.8950 with unchanged oracle thresholds. The initial C trio brought the total to 21 native PASS pairs. Instruction Edit and Reference then pass all four native calls each: medians 341.799 / 341.236 seconds, blue-suit region fraction 0.5725–0.5906, preserved-head MAE 0.02352–0.03312, and cat reference cosine 0.8792–0.8919. Inpaint/Outpaint also pass all four calls each: medians 390.071 / 445.548 seconds; unmasked MAE is 0.01185–0.01258; peak GPU0 memory is 14,194 / 14,338 MiB. Multi-reference also passes all four actual calls: median 427.440 seconds; both source references have CLIP cosine ≥0.7816, and the joint cat/coffee text beats every single-subject negative. Current total is **26 native PASS pairs**, C **8/8 PASS**, with zero production-ready pairs. The complete eight-case C gate aggregates explicit bounded capability batches; submitting a run is not a native PASS. Uncensored stays DISABLED; Pose/Depth/Canny are undeclared and no auxiliary weights are substituted. At Wave C completion, current-source lifecycle regressions and Phase 4–7 gates passed for source `b5ef30ae44eb538db4651da35b2c793006cc5d7b7606e736307ec41e2389859f`. Later video integrations require fresh regressions.

Current video integration software checks: **1,439 default tests + 65 service/workflow tests PASS**, zero failures/errors/skips; **138 native-harness software contracts** are covered across A–D. Model Truth, Capability and Deployment configuration gates pass after the explicit precision and deadline declarations. These checks do not certify GPU inference.

MiniMax Reference [v1](../validation/native_minimax_reference_receipt_v1.json) now passes all four exact native calls, numeric semantic checks and direct review of all 428 decoded frames. Median roundtrip is 494.219 s, peak owned GPU0 14,282 MiB and shutdown is clean. Reference identity/scene and gentle head motion/smiling remain meaningful; illumination flicker/glare, especially in iteration 3, is an explicit quality limitation. This certifies only the small recorded native golden, not artifact-free/general quality or Full Factory E2E. Base Video-with-Audio [v1](../validation/native_minimax_audio_receipt_v1.json) returns four genuine joint-AV outputs, median 502.084 s, peak GPU0 14,186 MiB and clean shutdown. Three iterations fail the unchanged speech oracle because independent pinned Whisper transcripts contain extra/missing words; no Audio V3 PASS is granted. Total native PASS is 28; Wave D is 2/13, production-ready remains zero.

Latest collection (2026-11-14): [Wan v7](../validation/native_wan_t2v_receipt_v7.json) returns four real native-grid videos with fixed VAE tiling, median 1,529.292 s, peak owned GPU0 7,050 MiB, zero OOM/lifecycle failures and clean shutdown. Direct review of all 164 frames confirms coherent balls/tables without v5 color-block corruption. Two iterations still fail the complete original scene/visibility/direction rules, including cropped initial balls and leftward movement. No Wan V3 PASS is granted. [Turbo v2](../validation/native_minimax_turbo_receipt_v2.json) confirms actual owned `--reserve-vram 1.000` but still trips the unchanged 14.2-GiB resident budget at 14,612 MiB before completing sampling; zero videos and zero host OOM. Increasing native reservation did not resolve that measured failure. Reference v1 and Base Video-with-Audio v1 have completed; their collected outcomes are recorded above.

Wave D has all **13 Manifest-declared model/capability goldens** and bounded native batches. MiniMax Base T2V [v3 receipt](../validation/native_minimax_t2v_receipt_v3.json) now passes its native gate: one warmup and three measured actual 384×256, 107-frame, 24-fps videos, all original pinned checkpoint files verified, explicit FP32 diffusion attested on GPU0, and clean shutdown. Median measured roundtrip is **409.010 seconds**, peak owned GPU0 memory **14,218 MiB**. Every decoded frame of all four videos was directly reviewed and hash-bound in the [content review](../validation/native_video_content_review.json): an identifiable red ball moves right across the white table; occasional ghosting/transparency remains a visible limitation. Numeric and direct content acceptance both pass for this small golden. **Wave D is 2/13; total native PASS pairs are 28, production-ready pairs remain zero.**

Wan v1 loses the owned process after loading; its OOM cause remains unproven. Wan v2 produces four actual MP4s but fails the complete original oracle and decoded content review. Wan v3 rejects the original uniform dtype contract before inference. Wan v4 verifies protected mixed precision and completes denoising, then records a host cgroup OOM kill with no returned video. Explicit sequential offload in [v5](../validation/native_wan_t2v_receipt_v5.json) fixes that measured memory failure: four actual MP4s, zero OOM events, clean shutdown, median **177.616 seconds** and peak owned GPU0 **3,554 MiB**. It still fails both the complete semantic oracle and all-frame content review because the outputs contain severe abstract/color-block corruption. None of these attempts is a Wan V3 PASS.

The exact VAE also completes limited [CPU photo](../validation/diagnostics/wan_vae_cpu_roundtrip_v1/report.json) and [CPU recorded-video](../validation/diagnostics/wan_vae_cpu_temporal_roundtrip_v1/report.json) component diagnoses: no missing/mismatched weights, finite output, photo pixel MAE 0.014584 and five recorded-frame MAEs 0.015249–0.017212. Direct review preserves the original scenes without color-block corruption. These results narrow the CPU reconstruction path only; they do not certify Wan T2V, CUDA/offload decoding, or native semantics and add no PASS pair.

[Wan v6](../validation/native_wan_t2v_receipt_v6.json) tests the predeclared native-grid case, 1280×704 / 41 frames / 50 steps / 24 fps. All 50 denoising steps finish in 18m30s, but VAE temporal-cache concatenation fails with CUDA OOM (2.15 GiB requested, 2.07 GiB free). Sampled owned GPU0 peak is 14,554 MiB; host OOM counters are zero and shutdown is clean. There are zero returned videos, so the native-grid quality hypothesis remains unverified. Explicit Manifest-owned VAE tiling (512×512 tiles, 384×384 strides) is being validated before a fresh exact-request rerun. The [tiled CPU photo diagnosis](../validation/diagnostics/wan_vae_cpu_tiled_roundtrip_v1/report.json) completes the original VAE at 1280×704 in 78.582 s with finite output, pixel MAE 0.003847 and coherent direct visual reconstruction. This is component-only CPU evidence; GPU temporal decoding, native video content and memory feasibility remain unverified. Checkpoint, mixed precision, sequential offload, prompt, seeds, numeric thresholds and all-frame content acceptance remain unchanged. Prior failed attempts and goldens remain intact.

MiniMax [I2V/FL2V v1](../validation/native_minimax_conditioned_receipt_v1.json) completes eight exact native MP4s with all numeric checks PASS, median I2V 524.576 s / FL2V 565.010 s, peak GPU0 14,282 MiB and clean shutdown. Direct review of all 856 frames rejects both complete four-output goldens: measured iteration 2 introduces conspicuous colored/luminous discs and flashes not requested by the prompts. Neither capability receives V3 PASS. Turbo v2 and Wan v7 have now completed; their collected results are recorded above. [Turbo v1](../validation/native_minimax_turbo_receipt_v1.json) returns no videos: at sampler initialization, owned GPU0 reaches 14,548 MiB, about 7 MiB over the unchanged 14.2-GiB budget. The supervisor correctly stops the daemon; host OOM counters are zero. All three Turbo capabilities remain NOT_MET. Turbo now explicitly declares 0.2 GiB of additional native allocator headroom, giving Comfy a 1.0-GiB reservation while keeping the telemetry acceptance limit at 14.2 GiB. Unsupported backend policies fail before weight access. Fresh runs additionally capture actual owned Comfy launch argv for policy attestation; a new bounded Turbo batch remains required. Original weights/LoRA, FP32 compute, GPU assignment, requests, seeds, steps and semantics are unchanged.

The exact checkpoint's official example loads its VAE separately in FP32. The generic Manifest schema now owns an explicit `weights.component_precisions` map; Wan declares VAE FP32 while retaining diffusion/text FP16. The launcher passes this metadata to the Wan Adapter, which loads the original VAE separately and constructs shared pipeline wrappers without a global dtype cast. MiniMax v1 is rejected before inference for exceeding the existing 107-frame bound; new golden v2 stays within that bound with unchanged semantic thresholds. MiniMax v2 reaches real weight loading but its guard rejects native manual-cast FP32 under Manifest FP16. Upstream H3 explicitly supports BF16/FP32; the three MiniMax Manifests and private loader now explicitly require FP32 diffusion on T4, preserving all original quantized checkpoint/LoRA files and allowed CPU offload. These are recorded, deliberate Manifest corrections before new runs, not runtime fallback. All failed receipts, source snapshots and original goldens remain immutable.

The Wan I2V/FL2V fixtures derive from the actual astronaut photograph; Extension uses recorded pedestrian footage. Validators decode native videos, verify container/identity/scene, keyframe/reference preservation and motion in the newly generated suffix. Video-with-audio additionally requires actual embedded audio, finite audible samples, matching AV duration and the requested spoken words checked by a pinned independent CPU Whisper verifier. Software controls reject a moving copied prefix with a static suffix, corrupt containers, wrong speech and missing audio tracks. The receipt-verified `benchmarks.native.video_review` exporter produces contact sheets covering every decoded frame and their source hashes, with no automatic PASS. Native image/video bytes still require direct content review; CLIP similarity alone did not establish meaningful v2 video quality. Wan v3 stops at its strict uniform-component dtype guard before inference (exit 3, zero OOM events). The pinned native Transformer/UMT5 APIs deliberately preserve protected FP32 parameters. A no-weights `meta`-device schema check, cross-checked against exact checkpoint configs/tensor indexes, identifies 95 Transformer and 24 UMT5 parameter names. These exact overrides now belong to `weights.component_parameter_precisions`; the launcher passes them to the Adapter, which verifies every actual floating parameter and reports the exact override identities. This is configuration evidence only until the new GPU run succeeds. Extension last-frame extraction is also corrected: the old last-100-ms seek returned success with no image for the real 2-fps input; streaming final-frame extraction now matches the actual decoded source frame exactly. Successful video candidates additionally require direct review of every decoded frame across all four hash-bound native outputs, closing the demonstrated v2 CLIP false positive. The last fully revalidated source `2da7ad7773e50f27bf841590499ae65337c24210e87b9a1d27c5e366aa3766e0` passes ASR lifecycle v20, isolated disabled Enhancement v15 and all Phase 4–7 gates; its [snapshot](../validation/source_gate_snapshots/2da7ad7773e50f27bf841590499ae65337c24210e87b9a1d27c5e366aa3766e0/index.json) preserves the exact reports. Wan v7 was submitted with fixed Manifest VAE tiling and identical native-grid request/seeds/oracles; its collected failure remains NOT_MET as recorded above. The newer launcher correction additionally passes Manifest extra headroom to native `--vram-headroom`, as required by the pinned DynamicVRAM API. Software regression remains 1,428 + 65 PASS; new-source ASR v21/Enhancement v16 and Phase 4–7 gates are queued before Turbo v3. Earlier source snapshots remain historical and intact. ASR is load-only; the disabled Enhancement probe does not establish V3/V4.

**Phase 8 Exit Gate is NOT_MET.** Wave A passes 10/10, Wave B passes 8/8, Wave C passes 8/8 and Wave D passes 2/13; Waves D–F remain unfinished. Full Factory E2E has not run. All public models remain evidence-locked and **zero model/capability pairs are production-ready**. Exact native archives and immutable receipts are referenced from [phase_status.json](../validation/phase_status.json). The [submitted source snapshots](../validation/native_sources) preserve both original core and validation ZIPs for selected Kaggle versions, retrieved from their submitted scripts and matched against kernel-code/core/validation SHA-256 records; source retrieval is provenance only. The recorded Git commit anchors the original baseline; archive SHA-256 values identify the actual local, uncommitted implementation.

```bash
.venv/bin/python -m deployments.kaggle.native_wave_a --kind text --collect
.venv/bin/python -m benchmarks.native.wave_a_gate
.venv/bin/python -m benchmarks.native.matrix
.venv/bin/python -m benchmarks.native.video_review --kind minimax_t2v --version 3 --output /tmp/factory-native-video/review-export
```

## Remaining phases and retained migration work

**Phases 9–11 have not started.**

- Run Native V3 waves in Master Plan order, with exact weights/backend/GPU/I/O and semantic validators. A failed, unsupported or hardware-blocked capability stays truthful; software mocks and container checks cannot certify it.
- Full Factory V4 E2E remains Phase 9 and is still required before production readiness. No public deployment/config flag grants activation.
- Studio migration remains Phase 10; retained migration sources and historical artifacts are only cleaned after their architecture dependencies, with final cleanup in Phase 11.

Current controlled temporal diagnostic: MiniMax max-frame bounds now admit the upstream 124-frame trained minimum; defaults and earlier evidence are unchanged. New Base I2V/FL2V v2 goldens request 124 frames with identical photos/prompts/seeds/steps/semantic thresholds; original 107-frame failures remain immutable. This is an unproven temporal-context hypothesis. Current source 9b036f991f926c0099704b65070a98856ec4b6c31528efa01aa80023f4e7997f requires fresh ASR v21/Enhancement v16 and Phase 4–7 gates before new native submissions. Software regression: 1,428 + 65 PASS. Active native jobs are checked every five minutes.

Current source 9b036f991f926c0099704b65070a98856ec4b6c31528efa01aa80023f4e7997f now passes ASR lifecycle v21, isolated Enhancement v16 and all Phase 4–7 gates; the immutable source snapshot preserves those reports. Turbo v3 is submitted with both native reservation controls. Base I2V/FL2V 124-frame v2 is being submitted as the next controlled temporal experiment. Wan I2V/FL2V native-grid v2 goldens are also predeclared with hashed real-photo JPEG derivatives, unchanged semantic thresholds and source-budget-verified scripts; those GPU batches have not started.


Turbo [v3](../validation/native_minimax_turbo_receipt_v3.json) completes one actual 107-frame warmup with numeric PASS (303.274 s, GPU0 peak 14,420 MiB), and direct inspection of every frame confirms a coherent red ball rolling right. The next request reaches 14,770 MiB, trips the unchanged 14.2-GiB supervisor budget, and stops fail-closed; no complete Turbo case passes. Original evidence remains immutable. A documented Turbo-only Manifest candidate raises explicit extra headroom to 0.75 GiB (native simple reserve 1.55 GiB / per-device extra 0.75 GiB); source `0f6451ebb9d3b260fb602d831b4d33a0ec838e10a770dfd94e99fc90fde99f07` awaits fresh GPU lifecycle regressions before a new Turbo batch. Headroom is a reservation target, not a relaxed acceptance limit or proof of success.

Base conditioned v2 (124 frames) and Reference joint-AV v1 are currently running on the last validated source `9b036f991f926c0099704b65070a98856ec4b6c31528efa01aa80023f4e7997f`; their remote status alone adds no inference evidence. Current-task monitoring polls every 300 seconds and collects actual terminal outputs. Wan I2V/FL2V and Extension native-grid cases are predeclared but unsubmitted; their unchanged semantic checks still require actual GPU outputs and all-frame review. D remains 2/13, total native PASS 28, production-ready zero.


Two further explicit request-only candidates are predeclared, with original failures retained and unchanged semantic/all-frame thresholds: Wan T2V `screen-right-v3` specifies screen coordinates, full-ball visibility and locked framing; Base joint-AV `native-dialogue-v2` uses the official MiniMax image-alignment/multimodal fields and tagged English utterance. Both keep the original four seeds and all checkpoint/runtime settings. Neither has been submitted or accepted. Native D software contracts pass after each change; the 1,430-test complete suite above predates these two request-only edits. No additional core-source change is involved.


A concrete Wan FL2V correctness blocker was found before its first GPU submission: the exact 5B model index sets `expand_timesteps=true`, and pinned native latent preparation ignores `last_image` in that branch. With the original exact FP32 VAE on CPU, changing between two distinct actual photographed last images leaves every returned noise/condition/mask tensor bitwise identical. This is component diagnosis only, not full Wan/GPU inference. The Adapter now rejects that branch before output creation; the FL2V requirement remains unmet rather than being dropped from D. Full software regression after this guard passes **1,431 + 65 tests**, D contracts **56**. Candidate source `b91c61f6ebab9957cc8488f40178ffba818dbb2855893e67bbb7c60cdd17664c` awaits fresh source regressions. The earlier Turbo-only candidate was superseded before any new lifecycle job was launched. A user decision on separately configured native FL2V checkpoint evaluation versus same-5B experimental Adapter research is pending; no implicit fallback is applied.


Reference joint-AV [v1](../validation/native_minimax_reference_audio_receipt_v1.json) completes four actual AV MP4s, median 498.221 s, GPU0 peak 14,314 MiB, no lifecycle failures and clean stop. All four fail unchanged speech checks due additional unrelated words after the requested sentence; no native PASS is added. A six-section native-dialogue-v2 request is predeclared with unchanged words/photo/seeds/settings/oracles.

Source `b91c61f6ebab9957cc8488f40178ffba818dbb2855893e67bbb7c60cdd17664c` now passes actual ASR lifecycle v22, isolated disabled Enhancement v17 and Phase 4–7 gates; the immutable snapshot is retained. Turbo [v4](../validation/native_minimax_turbo_run.json) has been submitted and its exact source archived on that validated revision, with the explicit 0.75-GiB extra headroom and unchanged requests/14.2-GiB guard. Remote running status is not inference proof.

The user explicitly retains Wan's exact 5B checkpoint and authorizes experimental Adapter research. Original-VAE CPU studies reject independent last-anchor representations and a held-first joint context for visible ghosting, even when endpoint MAE passes. A joint linear endpoint context improves the photo-crop component study (final MAE .026836), with mild transient double contours retained as a limitation. This supports only further experimental feasibility work; it is not Transformer/GPU/V3 evidence. The native FL2V guard remains in place pending a source-verified Adapter implementation and actual full native/semantic validation.


The experimental Wan Adapter now uses a source-verified subclass of the pinned native pipeline for explicit FL2V only. Its joint linear endpoint codec context fixes exactly the first/final latent timesteps while every middle latent remains sampled; no returned RGB frame is pasted/replaced. Ordinary native paths that ignore the last image remain rejected. On the original FP32 CPU VAE, changing between two distinct actual photographed last-image inputs changes the final condition while preserving the first condition and seeded noise; exact two-endpoint/free-middle masks and finite tensors are verified. This remains component-only evidence. Source identity, raw input/output MP4 hashes and actual encoded endpoint/mask identities are now required in the FL2V native gate. Full software suite **1,437 + 65 PASS** and **62 D contracts PASS**, Ruff checks pass. Source `08eefd3477ebf7b31900855c881f291a355b74ecfc83d2ca47b21c8fedee574a` is frozen pending fresh ASR/Enhancement source regressions and the complete four-call real Dual-T4 FL2V golden. No new V3 PASS/production readiness is granted.


Latest real collections: Base conditioned124 v2 returns eight MP4s with all numeric PASS, medians 622.179/670.159 s, peak14282MiB and clean stop, but direct review of all992 frames rejects both cases for colored/luminous artifacts. Turbo4 resolves observed residency (12 real videos, peak13844MiB, clean stop); T2V fails one red-visibility iteration and all856 I2V/FL2V frames were reviewed, rejecting extra luminous patches/discs in measured2. No clean-seed selection or oracle relaxation is applied.

Experimental same-checkpoint Wan5B FL2V1 completes four real native-grid videos with all numeric and all164-frame direct content PASS, median1454.215s and peak7086MiB. Static source/weights/runtime identity is present, but its required per-call endpoint/MP4 bindings were suppressed by root WARNING logging; the strict gate refuses certification. Original videos/review remain meaningful unaccepted native evidence. The Adapter now flushes bounded JSON bindings directly to captured stdout. Source `7266e8243bfef0b0161a8e4711619893f4263fd770894f802d13151fdd613f94` is frozen pending fresh ASR24/Enhancement19 source regressions, then the same complete FL2V golden will rerun. The gate is not weakened and historical records are not fabricated.

Base native-dialogue Audio2 and Wan native-grid I2V1 are running on last validated source `08eefd3477ebf7b31900855c881f291a355b74ecfc83d2ca47b21c8fedee574a` (ASR23/Enhancement18/Phase4–7 PASS snapshot retained). Current-task monitors poll every300seconds and collect actual terminal artifacts. A controlled same-latent FP16/FP32 VAE diagnostic is prepared and queued after source gates and an available slot; it changes no canonical Manifest and cannot grant V3/V4/readiness. Current complete software regression is **1,439 +65 PASS**, D contracts63 plus the diagnostic packaging control pass. D remains2/13, native PASS28, production-ready0.


Base Video+Audio v2 completes four actual native AV MP4s with clean stop, median519.573s and GPU0 peak14250MiB. All four independent transcripts exactly match "Factory makes useful tools." and all numeric semantics pass. Direct review of all428 frames nevertheless rejects the complete golden: measured iteration2 repeatedly adds saturated colored luminous discs, including face/neck interference and green face occlusion at frame100. The same visual rule as Base conditioned/Turbo is retained; no clean-seed selection or automatic PASS promotion. Actual outputs, hash-bound negative content review and speech improvement are preserved. The queued same-latent VAE precision diagnosis remains the next controlled investigation; canonical precision/checkpoint unchanged. D remains2/13, native PASS28, READY0.


Source7266 now passes actual ASR lifecycle24, isolated disabled Enhancement19 and Phase4–7 gates; immutable source snapshot saved. Exact same four-call Wan5B FL2V v2 is submitted on this source and archived. Wan I2V1 remains running on source08. A separate CPU-only replay uses receipt-bound actual AV v2 MP4s to diagnose decoder reproducibility, not new model/GPU inference or certification.


Current default software suite after evidence persistence/CPU-audit packaging passes1443 tests with0 failures/errors/skips (JUnit SHA0b2552e066baab4371ce4223107df5eb3fba117b5c70a496b3b589c37c04761f). The unchanged native WaveD gate is being exercised in a CPU-only compatible Linux audit against seven already-collected receipt-bound actual attempts. No semantic criterion is weakened; missing/failed cases and all negative direct visual reviews remain NOT_MET. Core source7266 unchanged.


Compatible CPU audit v3 now executes the unchanged native WaveD gate successfully against four fully receipt-verified mounted actual attempts. All four Base AV2 audio/numeric semantics, including exact decoded PCM identity, independently pass on Linux; its negative all428-frame visual review correctly keeps the case NOT_MET. The audit retains three unavailable exact failed-job mounts (Wan T2V7, Turbo4, Reference AV1), plus pending Wan cases; their original REST-collected actual artifacts remain intact. The partial audit is preserved separately and cannot be adopted as a complete current gate. D remains2/13, native PASS28, READY0. Turbo T2V4 all428-frame direct review also rejects cropping/unrequested people/luminous patches, with only an unsubmitted framing proposal retained.


Wan native-grid I2V v1 now passes the complete native gate: four actual exact-5B Dual-T4 outputs on validated source08, median1466.317s, GPU0 peak7104MiB, clean stop/no lifecycle failures, all numeric semantics PASS and all164 decoded frames directly reviewed as coherent reference-preserving head/expression motion. Minor facial/detail softening and tiny background drift remain scoped quality limitations. Hash-bound positive review retained. D increases to3/13, total native PASS29, READY0/V4 unrun. Source7266 controlled MiniMax same-latent VAE diagnosis v1 is submitted and archived on the freed slot, with no result yet; Wan FL2V2 continues. The default software suite after the final CPU-audit packaging corrections passes1443 with0 failures/errors/skips.


Controlled MiniMax same-latent VAE diagnosis v1 completes both real GPU stages with original weights, fixedGPU0 and clean shutdown. Actual Transformer call+FP16 video takes659.490s (peak14282MiB); isolated FP32 decoding of exactly the same captured latent SHA7bc1305d917e77eec7fe78e0eafa4b35e3d15d94b81ca7844afc0e34665b76c7 takes41.644s (peak5654MiB). Both return124 real frames; per-frame decoded pixel MAE ranges.004857-.009158. Direct review of all248 frames finds the same material colored flares/face occlusion (32-35) and final green helmet glow in both. Therefore the precision-only decoder fix is not supported for this actual case. Preserve this negative result; do not change canonical VAE precision, checkpoint, seeds, thresholds or historical verdicts. No V3 is granted from this one-call/component diagnosis. Wan Extension1 is now submitted/archived on validated source7266 and is running alongside Wan FL2V2. D3/13, native PASS29, READY0.


After the negative decoder-only diagnosis, a bounded original-checkpoint native-canvas one-call experiment is prepared/queued for the next free slot after Wan FL2V2 collection:1152x768, same actual photo bytes/prompt/seed314159/124frames/30steps and oracles; diagnostic Manifest copy only (height768,5400s deadline,static2.0GiB extra headroom), original14.2GiB budget andGPU0. No canonical Manifest/core change. Official pinned README768-default identity and the exact proposal are retained. Software preflight validates the actual request against the diagnostic schema and verifies the canonical schema still rejects the larger height; packaging stays below1MB. Current software suite1445 PASS,0 failures/errors/skips; targeted D/diagnostic/audit checks70 PASS. Source7266 unchanged, native PASS29/READY0, no new native result from preparation.


Before any expensive768p submission, primary-source review identifies a cheaper controlled difference: official workflow commit79b02f4 selects res_multistep while Factory uses Euler; exact backend c575 implements that sampler. The still-unsubmitted larger-canvas queue is deferred. A one-call sampler-only control is queued after Wan FL2V2 collection, preserving original384x256/124frames/30steps/seed314159/photo/prompt/CFG1/simple scheduler/weights/FP16-VAE control/deadline/headroom/oracles/GPU0/budget. This is a hypothesis, not a claimed root cause or fix; canonical/core remain unchanged. The released model is guidance-distilled, so no CFG increase/negative-guidance substitution is introduced. All3 diagnostic modes are explicit and mutually exclusive; default software suite1447 PASS,0 failures/errors/skips. Native PASS29/READY0 unchanged.


### Phase 8 — 2026-11-14: completed Wan runs and negative sampler control

Wan FL2V v2 and Extension v1 collected with real Dual-T4 outputs. All 164 FL2V and 356 extension frames directly reviewed; hash-bound content reviews recorded. Native gate must still verify all runtime/log/endpoint bindings. FL2V median 1495.343s, peak 7016 MiB; extension median 1408.535s, peak 7104 MiB.

The one-call res_multistep diagnostic completed, but all 124 frames retain the face-obscuring luminous artifacts (32–35 and 62–63). This rejects sampler-only repair for this case; canonical Manifest and Adapter remain unchanged. Next execute the already documented isolated 1152×768 spatial diagnosis with original weights, seed, prompt and unchanged semantic oracles; explicit diagnostic-only profile, no V3 credit. Higher resolution may still fail or exceed resource limits. Production readiness remains zero.


Gate re-evaluation after both Wan all-frame reviews confirms D native PASS 5/13 and total native PASS 31, READY 0. Wan FL2V v2 passes strict endpoint and per-call stdout evidence; extension v1 passes. The predeclared screen-right Wan T2V golden is submitted as Kaggle v8 on unchanged source 7266e824, alongside isolated native-canvas diagnosis v1. Both have 300-second status/collection monitors. Remaining D failures and missing E/F gates keep Phase 8 NOT_MET.


Scope clarification from MASTER_PLAN sections 35–36: Phase 8 Wave F is native Enhancement validation; Full Factory E2E V4 is Phase 9. Current Enhancement scope has zero enabled capabilities. Neither completion of D nor diagnosis execution grants Phase 8 aggregate PASS or production readiness. Predeclared Reference+Audio dialogue-v2 is queued after native-canvas collection, retaining four original seeds and exact speech/content oracles.


### Phase 8 — Wan T2V v8 result and queue adjustment

Real original-5B four-call v8 completed; native job NOT_MET (not a backend crash). All four pass fully visible red object and rightward motion, but iteration 3 fails unchanged object/scene semantic oracle. All 164 frames directly reviewed and hash-bound; no V3 credit, seed removal or threshold change. Median 1386.676s, GPU0 peak 7084 MiB, clean lifecycle. MiniMax 1152×768 diagnosis v1 remained Kaggle queued over two hours. Reference+Audio dialogue-v2 is therefore submitted after collected Wan v8, using the free slot; the old canvas-dependent waiter was stopped before submission to avoid duplicate jobs.


### Native-canvas diagnosis v1 failed before output

Kaggle v1 was collected (7 artifacts), real Dual-T4 with all exact weight hashes matched. Backend reached first sampler initialization, then local HTTP history request failed with connection reset. No sampled latent or MP4 exists. Observed GPU peak 10294 MiB; last periodic cgroup observations report no OOM event, which does not prove the final termination cause. Diagnosis cannot test visual quality. Add benchmark-only process exit code and memory capture immediately before intentional shutdown, then repeat identical diagnostic settings to identify termination cause; no canonical source, weights, seed, policy or oracle change. This retry is diagnostic only, not V3.


### Reference+Audio v2: real four-run speech and content success, compatible audit pending

All four actual 107-frame audiovisual outputs pass native numeric and independent ASR exact sentence checks, WER 0, median 505.733s, GPU0 peak 14282 MiB. All 428 frames directly reviewed, hash-bound PASS with minor texture/expression drift; no material face-obscuring color patches. This does not prove phoneme-level AV synchronization. Run compatible Linux unchanged-gate audit to independently reproduce exact PCM identity; local macOS decoder remains incompatible. Keep canonical V3 count unchanged until admissible gate evidence. No partial audit adoption.


Native-canvas v2 reproduced the connection reset before sampling, no output. Before intentional shutdown process.poll() was None, memory.events showed zero OOM counters, observed memory.current 28.19 GB below 32.21 GB. This instant snapshot cannot distinguish a transport interruption from a process death race. Next diagnostic adds a bounded 10-second passive observation only on inference exception, polling actual exit code and memory without resubmitting a prompt or replaying inference. Preserve exact original settings; no canonical change justified yet. Compatible Linux audit v4 independently verifies Reference+Audio v2 as PASS with original PCM decoder; report remains partial because failed Wan v8 and Turbo v4 mounted archives are missing. No partial audit adoption; canonical count stays31.


To complete compatible CPU audit despite Kaggle omitting failed notebook mounts, copy only the exact already-collected Wan v8 and Turbo v4 artifacts into a private evidence dataset under the same Kaggle account. Every file is verified against its original receipt before packaging; copied archives are explicitly not new inference. Audit still verifies complete original receipts and source/review identities. Use opaque archive filenames to preserve bytes through dataset mounting, record container hashes, and bind the private dataset as an audit input. No model weights or credentials uploaded.


### Compatible gate audit v5 fully verified and adopted

All 10/10 original native attempts verified against complete original receipts, with zero unavailable archives. Private byte-identical evidence copies restore the exact failed Wan v8/Turbo v4 archives. Compatible Linux FFmpeg exact executable and PCM identities verified by unchanged Wave D gate. Adoption verified current source/golden/review/run/receipt/code identities; provenance: validation/native_wave_d_gate_provenance.json. Reference+Audio v2 now V3 PASS. D6/13, total native32, READY0, Phase8 NOT_MET. Negative Wan T2V, Base conditioned/Audio and Turbo outcomes remain unchanged. Native-canvas v3 still running, no output evidence yet.


Native-canvas v3 passive observation captured exit SIGTERM(-15) 2.5 seconds after connection reset, with zero cgroup OOM counters. BackendSession monitor calls stop after failed health; prior shutdown reports show generic health/identity/GPU failure. Supervisor termination is a supported hypothesis; exact failing health check remains unknown. Before any core change, next benchmark-only wrapper will observe unchanged _json, listener ownership and GPU ownership checks, recording false returns/exceptions without suppressing or modifying them. No health timeout increase, bypass, new model or retry of a submitted graph.


Wan T2V v8 verifier limitation: exact pinned CLIP preprocessor SHA910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd uses shortest-edge resize224 and center_crop224. For1280×704 this observes approximately original x288..992. Measured iteration3 ends with ball centroid near the right border, outside that crop, although full-frame red-object visibility and direct all-frame review pass. This supports a verifier-field-of-view hypothesis, not proof that the model or oracle is correct. Keep current numeric NOT_MET. Before changing preprocessing, require a predeclared independent calibration with positives/negatives and a versioned full-frame evaluation policy; never retroactively lower the0.03 margin or select clean seeds. No new Wan prompt retry is launched on this finding alone.
