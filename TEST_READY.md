# TEST_READY: Factory v4 E2E Test Suite Specification & Coverage Report

**Status:** CONTRACT SUITE VERIFIED — LIVE GPU CERTIFICATION PENDING  
**Framework:** `pytest` (v9.1.1)  
**Execution Environment:** macOS / Linux / Python 3.10+  
**Total Automated Tests:** 468  
**Contract Simulation Pass Rate:** 100% (468/468 in the recorded run; not a live inference result)  
**Authoritative Specification:** `/Volumes/WorkSpace/Factory/Gen_Image:Video/Refactor.md`  
**Master Plan & Feature Inventory:** `/Volumes/WorkSpace/Project/Gen_Image:Video/PROJECT.md`  
**Test Infrastructure Guide:** `/Volumes/WorkSpace/Project/Gen_Image:Video/TEST_INFRA.md`  

---

## 1. Executive Summary

The Factory v4 contract test suite is published under `tests/e2e/`. In its default mode it validates API behavior against a deterministic `FactoryContractServer`; it does **not** prove that real GPU models, ngrok, PostgreSQL, Redis or S3 are operational. The suite covers the target contracts across all subsystems (Control Plane, Registry, Asset Service, Text, VLM, Decision, Speech, Embedding/Rerank, Image, Video, Music, Enhancement, Studio Web) over standardized external network contracts (HTTP REST, SSE, WebSocket, and S3 URLs).

The test suite operates in dual mode:
1. **Contract Simulation Mode (Default):** Blazingly fast in-process contract verification running against `FactoryContractServer` via `httpx.MockTransport`, ensuring 100% deterministic CI execution without requiring live GPUs.
2. **Live Cluster Mode (`FACTORY_E2E_LIVE=1`):** Seamless execution targeting live deployed Factory v4 services running locally or on Kaggle behind static ngrok domains.

---

## 2. Test Tier Coverage Summary

| Tier | Focus Area | Target Threshold | Actual Tests | Status |
|---|---|---|---|---|
| **Tier 1** | **Feature Coverage** | $\ge 5$ tests per feature (210 tests) | **210** | **100% PASS** |
| **Tier 2** | **Boundary & Corner Cases** | $\ge 5$ tests per feature (210 tests) | **218** | **100% PASS** |
| **Tier 3** | **Cross-Feature Interactions** | $\ge 20$ pairwise workflow tests | **25** | **100% PASS** |
| **Tier 4** | **Real-World Scenarios** | $\ge 15$ end-to-end multi-modal journeys | **15** | **100% PASS** |
| **Total** | **Contract Simulation Suite** | $\ge 450$ automated tests | **468** | **100% PASS** |

---

## 3. Feature-to-Test Mapping Matrix (All 42 Features)

| # | Feature | Category | Milestone | Tier 1 Test File | Tier 2 Test File | Tier 1 Count | Tier 2 Count |
|---|---------|----------|-----------|------------------|------------------|--------------|--------------|
| 1 | Job Submission | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_security_auth.py`, `test_t2_payload_boundaries.py` | 5 | 5+ |
| 2 | Job Status Query | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_security_auth.py`, `test_t2_error_sanitization.py` | 5 | 5+ |
| 3 | Job Cancellation | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_feature_corners.py` | 5 | 5+ |
| 4 | Job Retry | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_feature_corners.py` | 5 | 5+ |
| 5 | Job SSE Events | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_feature_corners.py` | 5 | 5+ |
| 6 | Job WebSocket | Control Plane | M2 | `test_t1_control_plane.py` | `test_t2_feature_corners.py` | 5 | 5+ |
| 7 | Service Manifest Discovery | Registry | M1 | `test_t1_registry.py` | `test_t2_security_auth.py`, `test_t2_feature_corners.py` | 5 | 5+ |
| 8 | Model Manifest Registry | Registry | M1 | `test_t1_registry.py` | `test_t2_security_auth.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 9 | Asset Service Upload / SSRF Validation | Asset Service | M1 | `test_t1_asset_service.py` | `test_t2_ssrf_and_assets.py` | 5 | 35 |
| 10 | Text Chat Completions | Text / LLM | M3 | `test_t1_text_vlm.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 11 | Text Responses API | Text / LLM | M3 | `test_t1_text_vlm.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 12 | Decision Predicate | Decision Engine | M3 | `test_t1_decision.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 13 | Decision Choice | Decision Engine | M3 | `test_t1_decision.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 14 | Decision Score | Decision Engine | M3 | `test_t1_decision.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 15 | Speech Audio Transcription | Speech Subsystem | M3 | `test_t1_speech.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 16 | Speech Streaming ASR | Speech Subsystem | M3 | `test_t1_speech.py` | `test_t2_security_auth.py` | 5 | 5+ |
| 17 | Speech Forced Alignment | Speech Subsystem | M3 | `test_t1_speech.py` | `test_t2_feature_corners.py` | 5 | 5+ |
| 18 | Speech Synthesis (TTS) | Speech Subsystem | M3 | `test_t1_speech.py` | `test_t2_feature_corners.py`, `test_t2_payload_boundaries.py` | 5 | 5+ |
| 19 | Vector Embeddings | Embedding & Rerank | M3 | `test_t1_embedding_rerank.py` | `test_t2_model_unavailability.py`, `test_t2_payload_boundaries.py` | 5 | 5+ |
| 20 | Document Rerank | Embedding & Rerank | M3 | `test_t1_embedding_rerank.py` | `test_t2_model_unavailability.py`, `test_t2_payload_boundaries.py` | 5 | 5+ |
| 21 | Image Text-to-Image | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 22 | Image Image-to-Image | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 23 | Image Inpainting | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 24 | Image Outpainting | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 25 | Image Instruction Edit | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 26 | Image Reference Conditioning | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 27 | Image ControlNet | Image Subsystem | M3 | `test_t1_image.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 28 | Video Text-to-Video | Video Subsystem | M3 | `test_t1_video.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 29 | Video Image-to-Video | Video Subsystem | M3 | `test_t1_video.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 30 | Video First & Last Frame (FL2V) | Video Subsystem | M3 | `test_t1_video.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 31 | Video Extension & Chaining | Video Subsystem | M3 | `test_t1_video.py` | `test_t2_feature_corners.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 32 | Video with Audio | Video Subsystem | M3 | `test_t1_video.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 33 | Music Text-to-Music | Music Subsystem | M3 | `test_t1_music.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 34 | Music Lyrics-to-Music | Music Subsystem | M3 | `test_t1_music.py` | `test_t2_payload_boundaries.py`, `test_t2_model_unavailability.py` | 5 | 5+ |
| 35 | Music Audio-to-Audio | Music Subsystem | M3 | `test_t1_music.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 36 | Enhancement Image Super-Resolution | Enhancement Subsystem | M3 | `test_t1_enhancement.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 37 | Enhancement Video Super-Resolution | Enhancement Subsystem | M3 | `test_t1_enhancement.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 38 | Enhancement Frame Interpolation | Enhancement Subsystem | M3 | `test_t1_enhancement.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 39 | Enhancement Audio Enhancement | Enhancement Subsystem | M3 | `test_t1_enhancement.py` | `test_t2_model_unavailability.py` | 5 | 5+ |
| 40 | Studio Dynamic Schema Form | Studio Web | M4 | `test_t1_studio.py` | `test_t2_security_auth.py` | 5 | 5+ |
| 41 | Studio Streaming Inspector | Studio Web | M4 | `test_t1_studio.py` | `test_t2_security_auth.py` | 5 | 5+ |
| 42 | Studio Code Generator | Studio Web | M4 | `test_t1_studio.py` | `test_t2_security_auth.py` | 5 | 5+ |

---

## 4. Test Execution Commands

### Run Full Test Suite (468 tests)
```bash
pytest tests/e2e -v
```

### Run by Tier
```bash
pytest tests/e2e/tier1_feature_coverage -v
pytest tests/e2e/tier2_boundaries_corners -v
pytest tests/e2e/tier3_cross_feature -v
pytest tests/e2e/tier4_real_world_scenarios -v
```

### Run by Security & Capability Markers
```bash
pytest tests/e2e -m security -v
pytest tests/e2e -m ssrf -v
pytest tests/e2e -m streaming -v
```

### Run Against Live Cluster Deployment
```bash
export FACTORY_E2E_LIVE=1
export FACTORY_BASE_URL="https://jobs.ai.example.com"
export FACTORY_API_KEY="sk-factory-prod-secret-token"

pytest tests/e2e -v
```

---

## 5. Verification Sign-Off

- **Contract Simulation:** recorded 468 passed. This is not production certification.
- **Execution Speed:** 0.29s for 468 tests
- **Sanitization Checks:** verified against the contract simulation; repeat against the live deployment before release.
- **SSRF Defense:** contract vectors exist; live DNS-rebinding and redirect tests remain a release gate.
- **Fail-Fast Compliance:** source and CI gates prohibit known fake-inference fallbacks; live unavailable-model behavior remains a release gate.
- **Handoff Artifact:** Ready for orchestrator and Milestone validation gates.


## Live-certification warning

Do not use the 468 contract tests or the historical 1,065-test commit message as evidence that Factory v4 is production-ready. Release certification requires the real-infrastructure gates in `LIVE_VALIDATION.md`, including native model inference, GPU OOM/admission tests, fixed-domain reconnect, durable queue failover, artifact integrity, and real streaming reconnect.
