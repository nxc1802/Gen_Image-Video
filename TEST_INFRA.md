# Factory v4 Test Infrastructure & Execution Guide

## 1. Overview & Architecture

Factory v4 is a production-grade, decoupled AI Factory where a dedicated **Control Plane** coordinates asynchronous jobs, service/model registries, asset sanitization, and streaming, while individual AI subsystems (Text, VLM, Decision, Speech, Embedding/Rerank, Image, Video, Music, Enhancement, Studio Web) execute on optimal native serving frameworks (vLLM, ComfyUI, ACE-Step 1.5).

The Factory v4 End-to-End (E2E) Test Suite is an **opaque-box testing framework** designed to validate 100% of the 42 inventoried features across all subsystems without coupling to internal implementation details. The suite treats the system as a distributed black box interacting strictly over standardized network contracts (HTTP REST, SSE, WebSocket, and S3 URLs).

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Pytest Runner & Test Suite                      │
│            (Tier 1: Features • Tier 2: Boundaries • Tier 3 • Tier 4)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Factory Test Harness / Client                    │
│   - Authorization Header Injection (Bearer token)                      │
│   - Dual-Mode Transport Adapter: Live HTTP/WS vs Contract Simulation   │
│   - Response Schema & Error Sanitization Assertion Utilities           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          │ Live Mode (FACTORY_E2E_LIVE=1)                    │ Contract Simulation Mode (Default)
          ▼                                                   ▼
┌───────────────────────────────────┐               ┌───────────────────────────────────┐
│     Live Factory Control Plane    │               │  Factory Contract Mock Server     │
│       & Subsystem Static ngrok    │               │  - Full Schema Validation         │
│               Domains             │               │  - Two-Tier Queue & State Machine │
│   (text.*, image.*, jobs.*, etc.) │               │  - SSRF & Security Defense Engine │
└───────────────────────────────────┘               └───────────────────────────────────┘
```

---

## 2. Testing Methodology

The test suite applies formal software testing methodologies to guarantee robust coverage across all 42 features:

### 2.1 Category-Partition Methodology
Every endpoint parameter and request payload is partitioned into equivalence classes:
1. **Valid Equivalent Inputs**: Standard prompts, valid dimensions within limits (e.g., 1024x1024 for image), supported audio sample rates (16kHz, 24kHz, 44.1kHz), supported video frame counts (49, 81 frames).
2. **Invalid Equivalent Inputs**: Non-existent model IDs, unsupported task names, malformed URLs, empty arrays for choice decisions, invalid MIME types.
3. **Special Character Classes**: UTF-8 multilingual characters, emoji prompts, markdown formatting, SQL/JSON meta-characters, shell command injection patterns.

### 2.2 Boundary Value Analysis (BVA)
Tests probe extreme edges and limit transitions:
- **Zero & Negative Numbers**: `priority: 0`, `priority: -1`, `temperature: 0.0`, `seed: -1`, `scale_factor: 0`.
- **Upper Thresholds**: Image dimensions exceeding 2048x2048 (e.g. 2049, 4096, 8192), video duration > 30s, music duration > 300s, payload body > 50MB.
- **Off-by-One Boundaries**: Exact maximum context tokens vs maximum + 1, audio files at exact size ceiling vs ceiling + 1 byte.
- **Empty / Null Values**: Empty string prompts (`""`), empty message lists (`[]`), empty choice sets, `null` payloads.

### 2.3 Pairwise & Combinatorial Testing
Subsystems do not operate in isolation. Pairwise test matrices exercise interactions across disparate services:
- **Text → Image → Enhancement**: Text prompt generates character concept → Qwen-Image produces base PNG → Real-ESRGAN upscales 4x.
- **Speech ASR → LLM → Speech TTS**: Spoken query transcribed by Qwen3-ASR → Qwen Chat generates response → Qwen3-TTS synthesizes voice cloned response.
- **Asset Service Ingest → SSRF Filter → Visual Pipeline**: User media verified against loopback/cloud IPs → uploaded to S3 → fed to Image-to-Video.
- **Job Submission → State Machine → SSE Progress → S3 Retrieval**: Async job lifecycle tracking across CREATED → QUEUED → RUNNING → STREAMING → SUCCEEDED.

### 2.4 Adversarial & Security Testing
- **Mandatory Bearer Authentication**: Verifying that requests missing tokens, containing empty tokens (`API_KEY=""`), or supplying malformed JWTs receive immediate `401 Unauthorized`.
- **Arbitrary Proxy & Credential Leakage Prevention**: Confirming `x-target-base-url` headers are stripped/ignored, and tokens are never forwarded to fallback domains.
- **SSRF Attack Vectors**: Probing asset validation with IPv4 loopback (`127.0.0.1`), IPv6 loopback (`::1`), private networks (`10.0.0.1`, `172.16.0.1`, `192.168.1.1`), and cloud metadata endpoints (`169.254.169.254`).
- **Error Response Sanitization**: Fuzzing parameters to trigger exceptions while verifying that 0 internal stack traces, Python exception class names, or filesystem paths (e.g., `/Volumes/...`) appear in response JSON.
- **Model Unavailability Fail-Fast**: Ensuring that offline or failing models return `503 MODEL_UNAVAILABLE` rather than generating deterministic fake vectors or mock classifications.

---

## 3. Test Suite Tiers & Gate Thresholds

The test suite is structured into four sequential, progressive tiers:

| Tier | Name | Target Scope | Minimum Threshold | Gate Pass Criteria |
|---|---|---|---|---|
| **Tier 1** | **Feature Coverage** | All 42 inventoried features | $\ge 5$ test cases per feature (210 tests) | 100% Pass. Validates primary happy path and basic contract adherence for every feature. |
| **Tier 2** | **Boundary & Corner Cases** | Security, SSRF, Error Sanitization, BVA | $\ge 5$ test cases per feature (210+ tests) | 100% Pass. Verifies auth enforcement, SSRF blocking, 503 fail-fast, and parameter bounds. |
| **Tier 3** | **Cross-Feature Interactions** | Pairwise multi-service workflows | $\ge 20$ interaction pipeline tests | 100% Pass. Validates inter-service data flow, asset passing, and state synchronization. |
| **Tier 4** | **Real-World Scenarios** | Multi-modal enterprise end-to-end workflows | $\ge 15$ end-to-end workflow tests | 100% Pass. Validates full production user journeys, worker recovery, and multi-step media synthesis. |

**Total Suite Size:** $\ge 450$ automated test cases.

---

## 4. Test Directory Layout

```text
tests/
└── e2e/
    ├── conftest.py                       # Global test fixtures, client factories, auth helpers
    ├── mock_contract_server.py           # Specification-compliant contract simulator
    │
    ├── tier1_feature_coverage/           # Tier 1: 5+ tests per feature across all 42 features
    │   ├── test_t1_control_plane.py      # Features 1-6 (Jobs, Status, Cancel, Retry, SSE, WS)
    │   ├── test_t1_registry.py           # Features 7-8 (Service Manifest, Model Manifest)
    │   ├── test_t1_asset_service.py      # Feature 9 (Upload, Validation)
    │   ├── test_t1_text_vlm.py           # Features 10-11 (Chat Completions, Responses API)
    │   ├── test_t1_decision.py           # Features 12-14 (Predicate, Choice, Score)
    │   ├── test_t1_speech.py             # Features 15-18 (ASR, Streaming ASR, Alignment, TTS)
    │   ├── test_t1_embedding_rerank.py   # Features 19-20 (Embeddings, Rerank)
    │   ├── test_t1_image.py              # Features 21-27 (T2I, I2I, Inpaint, Outpaint, Edit, Ref, ControlNet)
    │   ├── test_t1_video.py              # Features 28-32 (T2V, I2V, FL2V, Extension, Audio VAE)
    │   ├── test_t1_music.py              # Features 33-35 (T2M, Lyrics, Audio2Audio)
    │   ├── test_t1_enhancement.py        # Features 36-39 (Image Upscale, Video Upscale, Interpolation, Audio Enhance)
    │   └── test_t1_studio.py             # Features 40-42 (Schema Form, Streaming Inspector, Codegen)
    │
    ├── tier2_boundaries_corners/         # Tier 2: Boundary, security, and corner cases
    │   ├── test_t2_security_auth.py      # Auth bypass, empty tokens, expired tokens, x-target-base-url stripping
    │   ├── test_t2_ssrf_and_assets.py    # SSRF loops, private ranges (10.x, 172.x, 192.168.x), cloud metadata (169.254.x), oversize
    │   ├── test_t2_error_sanitization.py # 500/400 error structures, zero traceback leakage, zero local path exposure
    │   ├── test_t2_model_unavailability.py # 503 fail-fast, zero fake vectors, zero fake heuristic decisions
    │   ├── test_t2_payload_boundaries.py # Zero/negative numbers, empty strings, max token limits, extreme resolutions
    │   └── test_t2_feature_corners.py    # Corner cases across all 42 features (out-of-range, malformed masks, corrupt audio, etc.)
    │
    ├── tier3_cross_feature/              # Tier 3: Pairwise cross-feature interactions
    │   ├── test_t3_text_image_enhancement.py # Text prompt -> Image generation -> Image Super-Resolution
    │   ├── test_t3_speech_llm_tts.py        # Audio ASR -> LLM reasoning -> TTS voice cloning
    │   ├── test_t3_job_sse_storage.py       # Job submission -> SSE progress events -> S3 signed URL retrieval
    │   ├── test_t3_decision_routing.py      # Input context -> Decision score -> routing to Image or Video worker
    │   ├── test_t3_video_fl2v_audio.py      # Image -> FL2V keyframe interpolation -> Stereo Audio VAE muxing
    │   └── test_t3_asset_ingest_pipeline.py # Asset upload -> SSRF check -> Image-to-Image -> Video animation
    │
    └── tier4_real_world_scenarios/       # Tier 4: End-to-end multi-modal business applications
        ├── test_t4_podcast_production.py    # Real-world podcast generation: Topic -> Script -> TTS -> Music backing -> Audio enhance
        ├── test_t4_video_ad_campaign.py     # End-to-end commercial ad: Product prompt -> Image -> Video -> Upscale -> Audio
        ├── test_t4_multimodal_qa_gating.py  # VLM document analysis -> Decision risk score -> Rerank sources -> Structured response
        └── test_t4_worker_resilience.py     # Worker reboot drill (10x static domain check), dead worker lease reclaim, job cancel/retry
```

---

## 5. Execution Commands & CLI Options

The test suite is driven entirely through standard `pytest`.

### 5.1 Run the Full Test Suite (All Tiers)
```bash
pytest tests/e2e -v
```

### 5.2 Run Specific Tiers
```bash
# Tier 1 only (Feature Coverage)
pytest tests/e2e/tier1_feature_coverage -v

# Tier 2 only (Boundaries & Corner Cases)
pytest tests/e2e/tier2_boundaries_corners -v

# Tier 3 only (Cross-Feature Interactions)
pytest tests/e2e/tier3_cross_feature -v

# Tier 4 only (Real-World Application Scenarios)
pytest tests/e2e/tier4_real_world_scenarios -v
```

### 5.3 Filter by Pytest Markers
```bash
# Run security & auth boundary tests
pytest tests/e2e -m security -v

# Run SSRF validation tests
pytest tests/e2e -m ssrf -v

# Run streaming tests (SSE / WebSocket)
pytest tests/e2e -m streaming -v
```

### 5.4 Live Deployment Mode
To run tests against a live deployed Factory v4 instance (e.g. running locally or on Kaggle behind ngrok static domains):
```bash
export FACTORY_E2E_LIVE=1
export FACTORY_BASE_URL="https://jobs.ai.example.com"
export FACTORY_API_KEY="sk-factory-prod-secret-token"

pytest tests/e2e -v
```

---

## 6. Pass / Fail Criteria & Exit Gates

A test run is considered **PASS** if and only if:
1. **100% Exit Code 0**: Every executed test case passes without unexpected errors or assertion failures.
2. **Zero Facade Tests**: Every test performs concrete assertions against HTTP status codes, headers, response schemas, and payload properties.
3. **Strict Error Conformance**: All negative test cases verify the exact unified error contract:
   ```json
   {
     "error": {
       "code": "ERROR_CODE_STRING",
       "message": "Human readable string",
       "request_id": "req-uuid-v4",
       "details": {}
     }
   }
   ```
4. **Zero Traceback Leakage**: Asserted that no response body contains `Traceback (most recent call last)`, file paths (e.g. `/Volumes/`, `/usr/`, `C:\`), or Python exception classes (`Exception`, `KeyError`, `ZeroDivisionError`).
5. **Fail-Fast Compliance**: Asserted that when an embedding/rerank/decision model fails, the response is HTTP `503 MODEL_UNAVAILABLE` and never a simulated/heuristic mock.
