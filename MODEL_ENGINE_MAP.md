# Factory v4 Lite — Model Manifest & Native Engine Mapping Matrix

**Milestone:** Phase 0 (Freeze, Inventory, and Baseline Recovery)  
**Date:** 2026-11-08  
**Hardware Environment:** Kaggle Dual NVIDIA Tesla T4 (Discrete PCIe, 2x 15.0 GB total / 14.2 GB allocatable)  
**Authoritative Manifests Directory:** `manifests/models/*.yaml`  
**Hardware Manifest:** `manifests/hardware/kaggle-dual-t4.yaml`  

---

## 1. Hardware Topology & Allocation Policy

Factory v4 Lite is designed to execute directly on Kaggle Dual Tesla T4 hardware. Memory is discrete and **never pooled** across PCIe cards:

```text
+-------------------------------------------------------------------------------------------------+
|                                    Kaggle Dual T4 Host Topology                                  |
|   4 vCPUs | 32.0 GB System RAM | 70.0 GB Temp Storage (/tmp & /kaggle/working) | CUDA 12.1+    |
+---------------------------------------------------------------+---------------------------------+
|                           GPU 0                               |              GPU 1              |
|  NVIDIA Tesla T4 (15.0 GB driver VRAM / 14.2 GB allocatable)   |  NVIDIA Tesla T4 (15.0 / 14.2)  |
|  Assigned Services:                                           |  Assigned Services:             |
|   - text-service (qwen-text)                                  |   - embed-service (qwen-embed)  |
|   - vision-service (qwen-vlm)                                 |   - rerank-service (qwen-rerank)|
|   - decision-service (qwen-decision)                          |   - video-service (wan, minimax)|
|   - speech-service: ASR component (qwen-speech)               |   - music-service (ace, yue2)   |
|   - image-service (qwen-image-2.1)                            |   - enhancement-service (suite) |
|                                                               |   - speech-service: TTS comp.   |
+---------------------------------------------------------------+---------------------------------+
```

### Discrete Memory Placement Rules
1. **Single-Card VRAM Ceiling**: A workload assigned to `single_gpu` MUST fit within 14.2 GB VRAM on its assigned device. Any single model checkpoint requiring > 14.2 GB without supported quantization is classified as `HARDWARE_BLOCKED`.
2. **Dual-GPU Partitioning**: GPU 0 hosts interactive visual and conversational LLM engines (vLLM, ComfyUI Image). GPU 1 hosts generative video, music, and embedding engines.
3. **Dual-GPU Split Workload**: `qwen-speech` uses `dual_gpu_split`, assigning ASR (Qwen3-ASR 1.7B) to GPU 0 and TTS (Qwen3-TTS 12Hz 1.7B) to GPU 1.

---

## 2. Consolidated 11-Model Engine Matrix

| Model ID | Service ID | Checkpoint & Format | Native Backend | Target GPU | Min VRAM | Status | Coverage | Supported Tasks | Unsupported Capabilities (Fail Closed) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`qwen-text`** | `text-service` | `Qwen/Qwen2.5-7B-Instruct-AWQ`<br>*(safetensors, AWQ, fp16)* | vLLM (`vllm>=0.6.0`) | GPU 0 | 12.0 GB | `full` | 100.0% | `chat_completion`, `responses_api`, `structured_output`, `tool_calling`, `streaming` | None (100% declared supported) |
| **`qwen-vlm`** | `vision-service` | `Qwen/Qwen2-VL-7B-Instruct-AWQ`<br>*(safetensors, AWQ, fp16)* | vLLM Vision (`vllm>=0.6.0`) | GPU 0 | 14.0 GB | `full` | 100.0% | `image_chat`, `responses_api`, `video_chat`, `visual_grounding`, `document_qa` | None (100% declared supported) |
| **`qwen-decision`** | `decision-service` | `Qwen/Qwen2.5-3B-Instruct`<br>*(safetensors, bf16)* | vLLM Router (`vllm>=0.6.0`) | GPU 0 | 6.0 GB | `full` | 100.0% | `predicate_evaluation`, `choice_routing`, `score_ranking` | None (100% declared supported) |
| **`qwen-embedding`** | `embed-service` | `Qwen/Qwen3-Embedding-0.6B`<br>+ `Qwen/Qwen3-Reranker-0.6B`<br>*(safetensors, fp16)* | PyTorch / Transformers (`transformers>=4.40.0`) | GPU 1 | 4.0 GB | `partial` | 66.67% | `text_embedding`, `reranking` | `multimodal_embedding` |
| **`qwen-speech`** | `speech-service` | `Qwen/Qwen3-ASR-1.7B`<br>+ `Qwen/Qwen3-TTS-12Hz-1.7B-Base`<br>*(safetensors, bf16)* | vLLM STT + vLLM-Omni TTS (`vllm>=0.6.0`) | GPU 0 & 1 (`dual_gpu_split`) | 8.0 GB | `partial` | 75.0% | `speech_to_text`, `text_to_speech`, `voice_cloning` | `word_timestamps` |
| **`qwen-image-2.1`** | `image-service` | `abenzerps/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf`<br>*(GGUF, Q4_K_M, bf16)* | ComfyUI Headless (`diffusers`, `gguf`, `comfyui`) | GPU 0 | 14.0 GB | `full` | 100.0% | `text_to_image`, `image_to_image`, `inpaint`, `outpaint`, `instruction_edit`, `reference_image`, `multi_reference`, `controlnet`, `control_pose`, `control_depth` | None (100% declared supported) |
| **`wan-2.2`** | `video-service` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`<br>*(diffusers, 4-bit, fp16)* | ComfyUI Headless (`diffusers>=0.30.0`, `imageio`) | GPU 1 | 14.0 GB | `partial` | 66.67% | `text_to_video`, `image_to_video`, `first_last_frame_to_video`, `video_extension` | `video_with_audio`, `reference_conditioning` |
| **`minimax-h3`** | `video-service` | `MiniMax/Hailuo-H3-Diffusers`<br>*(diffusers, 4-bit, fp16)*<br>+ `audio_vae` + `video_vae` | ComfyUI Headless (`diffusers`, `imageio`) | GPU 1 | 14.0 GB | `partial` | 83.33% | `text_to_video`, `image_to_video`, `first_last_frame_to_video`, `video_with_audio`, `reference_conditioning` | `video_extension` |
| **`ace-step-1.5`** | `music-service` | `ACE-Studio/ACE-Step-1.5`<br>*(safetensors, bf16, 2B/XL)* | Native ACE-Step API (`run_acestep_api.sh`, PyTorch) | GPU 1 | 6.0–12.0 GB | `partial` | 81.82% | `text_to_music`, `lyrics_to_music`, `reference_audio`, `cover`, `repaint` (all 5 variants);<br>`extract`, `lego`, `complete`, `music_extend` (`base` & `xl_base` only) | `audio_to_audio`, `audio_to_music` |
| **`yue2`** | `music-service` | `m-a-p/YuE-s1-7B-anneal-en-cot`<br>*(safetensors, bf16)* | Native YuE2 API (`xformers`, `transformers`) | GPU 1 | 12.0 GB (T4) / 24 GB (native) | `partial` | 85.71% | `text_to_music`, `lyrics_to_music`, `symbolic_score`, `custom_score`, `score_edit`, `song_variation` | `cover_edit`<br>*(Gate Y0 pending live verification)* |
| **`enhancement-suite-v1`** | `enhancement-service` | `xinntao/Real-ESRGAN`<br>+ `RIFE` + `BSRGAN` + `DeepFilterNet` | Real enhancement HTTP worker (unconfigured) | GPU 1 | 6.0 GB | `partial` | 0.0% | None currently runnable. | `image_super_resolution`, `video_super_resolution`, `frame_interpolation`, `audio_enhancement` *(all fail closed with `BACKEND_UNAVAILABLE`)* |

---

## 3. Deep-Dive Specification by Model

### 3.1 Model 1: `qwen-text`
- **Manifest:** `manifests/models/qwen-text.yaml`
- **Model ID:** `qwen-text` | **Service ID:** `text-service` | **Service Port:** `8080`
- **Primary Checkpoint:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Weights Format:** `safetensors` | **Quantization:** `awq` | **Precision:** `fp16`
- **Native Backend:** vLLM (`vllm>=0.6.0`, OpenAI-compatible completions API)
- **Target GPU:** GPU 0 | **Min VRAM:** 12.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `chat_completion`: `standard`
  - `responses_api`: `standard`
  - `structured_output`: `standard` (JSON schema enforcement)
  - `tool_calling`: `standard` (function execution schemas)
  - `streaming`: `standard` (SSE token chunks)
- **Capability Truth:** 100% declared supported. All 5 capabilities implemented.
- **Fail-Closed Safeguard:** Unsupported parameters (e.g. invalid JSON schema or missing prompt) return HTTP 400. Backend offline returns HTTP 503 (`BACKEND_UNAVAILABLE`).

---

### 3.2 Model 2: `qwen-vlm`
- **Manifest:** `manifests/models/qwen-vlm.yaml`
- **Model ID:** `qwen-vlm` | **Service ID:** `vision-service` | **Service Port:** `8081`
- **Primary Checkpoint:** `Qwen/Qwen2-VL-7B-Instruct-AWQ`
- **Weights Format:** `safetensors` | **Quantization:** `awq` | **Precision:** `fp16`
- **Native Backend:** vLLM Vision (`vllm>=0.6.0`, multimodal chat API)
- **Target GPU:** GPU 0 | **Min VRAM:** 14.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `image_chat`: `standard`
  - `responses_api`: `standard`
  - `video_chat`: `standard` (frame-sampled visual comprehension)
  - `visual_grounding`: `standard` (bounding box coordinate prediction `[ymin, xmin, ymax, xmax]`)
  - `document_qa`: `standard` (OCR and document structure extraction)
- **Capability Truth:** 100% declared supported.
- **Fail-Closed Safeguard:** Remote image URLs targeting loopback or private IPs trigger SSRF 403. Corrupt media containers return HTTP 400.

---

### 3.3 Model 3: `qwen-decision`
- **Manifest:** `manifests/models/qwen-decision.yaml`
- **Model ID:** `qwen-decision` | **Service ID:** `decision-service` | **Service Port:** `8082`
- **Primary Checkpoint:** `Qwen/Qwen2.5-3B-Instruct`
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `bf16`
- **Native Backend:** vLLM Structured Router (`vllm>=0.6.0`)
- **Target GPU:** GPU 0 | **Min VRAM:** 6.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `predicate_evaluation`: `standard` (boolean deterministic evaluation)
  - `choice_routing`: `standard` (optimal candidate route selection)
  - `score_ranking`: `standard` (item ranking and score normalization)
- **Capability Truth:** 100% declared supported.
- **Fail-Closed Safeguard:** Empty candidate lists or unparseable context objects fail with HTTP 400.

---

### 3.4 Model 4: `qwen-embedding`
- **Manifest:** `manifests/models/qwen-embedding.yaml`
- **Model ID:** `qwen-embedding` | **Service ID:** `embed-service` | **Service Port:** `8083` (rerank: `8084`)
- **Primary Checkpoint:** `Qwen/Qwen3-Embedding-0.6B`
- **Components:** `reranker: Qwen/Qwen3-Reranker-0.6B`
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `fp16`
- **Native Backend:** PyTorch / Transformers (`transformers>=4.40.0`, TEI-compatible)
- **Target GPU:** GPU 1 | **Min VRAM:** 4.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `text_embedding`: `standard` (1024-dim dense vector embedding)
  - `reranking`: `standard` (cross-encoder query-document score ranking)
- **Capability Truth (Coverage: 66.67%)**:
  - `text_embedding`: `true`
  - `reranking`: `true`
  - `multimodal_embedding`: `false`
- **Fail-Closed Safeguard:** Requesting `multimodal_embedding` raises `UnsupportedCapabilityError` with HTTP 400. Zero fake vectors are ever returned.

---

### 3.5 Model 5: `qwen-speech`
- **Manifest:** `manifests/models/qwen-speech.yaml`
- **Model ID:** `qwen-speech` | **Service ID:** `speech-service` | **Service Port:** `8288`
- **Primary Checkpoint:** `Qwen/Qwen3-ASR-1.7B` (ASR)
- **Components:** `tts_base: Qwen/Qwen3-TTS-12Hz-1.7B-Base` (TTS)
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `bf16`
- **Native Backend:** vLLM STT + vLLM-Omni TTS (`vllm>=0.6.0`, torchaudio)
- **Target GPU:** GPU 0 & 1 | **Min VRAM:** 8.0 GB | **Alloc Policy:** `dual_gpu_split`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `speech_to_text`: `standard` (ASR transcription)
  - `text_to_speech`: `standard` (TTS audio synthesis)
  - `voice_cloning`: `standard` (few-shot voice replication)
- **Capability Truth (Coverage: 75.0%)**:
  - `speech_to_text`: `true`
  - `text_to_speech`: `true`
  - `voice_cloning`: `true`
  - `word_timestamps`: `false`
- **Fail-Closed Safeguard:** Requesting `word_timestamps` raises `UnsupportedCapabilityError`. Invalid audio formats return HTTP 400.

---

### 3.6 Model 6: `qwen-image-2.1`
- **Manifest:** `manifests/models/qwen-image-2.1.yaml`
- **Model ID:** `qwen-image-2.1` | **Service ID:** `image-service` | **Service Port:** `8188`
- **Primary Checkpoint:** `abenzerps/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf`
- **Components:**
  - `clip`: `qwen3vl_8b_fp8_scaled.safetensors`
  - `vae`: `qwen_image_2.1_vae_bf16.safetensors`
  - `control_pose`: `qwen_image_control_pose.safetensors`
  - `control_depth`: `qwen_image_control_depth.safetensors`
- **Weights Format:** `gguf` | **Quantization:** `q4_k_m` | **Precision:** `bf16`
- **Native Backend:** ComfyUI Headless (`diffusers`, `gguf`, `comfyui`)
- **Target GPU:** GPU 0 | **Min VRAM:** 14.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard` (50 steps), `turbo` (`qwen-image-2.1-turbo-Q4_K_M.gguf`, 8 Euler steps)
- **Task Matrix:**
  - `text_to_image`: `standard`, `turbo`
  - `image_to_image`: `standard`, `turbo`
  - `inpaint`: `standard`
  - `outpaint`: `standard`
  - `instruction_edit`: `standard`
  - `reference_image`: `standard`
  - `multi_reference`: `standard`
  - `controlnet`: `standard`
  - `control_pose`: `standard`
  - `control_depth`: `standard`
- **Capability Truth:** 100% declared supported.
- **Fail-Closed Safeguard:** Dimension mismatch, mask incompatibility, or backend offline returns explicit error; zero placeholder images.

---

### 3.7 Model 7: `wan-2.2`
- **Manifest:** `manifests/models/wan-2.2.yaml`
- **Model ID:** `wan-2.2` | **Service ID:** `video-service` | **Service Port:** `8189`
- **Primary Checkpoint:** `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- **Weights Format:** `diffusers` | **Quantization:** `4bit` | **Precision:** `fp16`
- **Native Backend:** ComfyUI Headless (`diffusers>=0.30.0`, `imageio`, `imageio-ffmpeg`)
- **Target GPU:** GPU 1 | **Min VRAM:** 14.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `text_to_video`: `standard`
  - `image_to_video`: `standard`
  - `first_last_frame_to_video`: `standard`
  - `video_extension`: `standard`
- **Capability Truth (Coverage: 66.67%)**:
  - `text_to_video`: `true`
  - `image_to_video`: `true`
  - `first_last_frame_to_video`: `true`
  - `video_extension`: `true`
  - `video_with_audio`: `false`
  - `reference_conditioning`: `false`
- **Fail-Closed Safeguard:** Requesting `video_with_audio` or `reference_conditioning` raises `UnsupportedCapabilityError`. Requests exceeding 14.2 GB VRAM fail with `HARDWARE_BLOCKED`.

---

### 3.8 Model 8: `minimax-h3`
- **Manifest:** `manifests/models/minimax-h3.yaml`
- **Model ID:** `minimax-h3` | **Service ID:** `video-service` | **Service Port:** `8189`
- **Primary Checkpoint:** `MiniMax/Hailuo-H3-Diffusers`
- **Components:**
  - `audio_vae`: `minimax_h3_audio_vae_fp32.safetensors`
  - `video_vae`: `minimax_h3_video_vae_fp16.safetensors`
- **Weights Format:** `diffusers` | **Quantization:** `4bit` | **Precision:** `fp16`
- **Native Backend:** ComfyUI Headless (`diffusers`, `imageio`, `imageio-ffmpeg`)
- **Target GPU:** GPU 1 | **Min VRAM:** 14.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `text_to_video`: `standard`
  - `image_to_video`: `standard`
  - `first_last_frame_to_video`: `standard`
  - `video_with_audio`: `standard`
  - `reference_conditioning`: `standard`
- **Capability Truth (Coverage: 83.33%)**:
  - `text_to_video`: `true`
  - `image_to_video`: `true`
  - `first_last_frame_to_video`: `true`
  - `video_with_audio`: `true`
  - `reference_conditioning`: `true`
  - `video_extension`: `false`
- **Fail-Closed Safeguard:** Requesting `video_extension` raises `UnsupportedCapabilityError`.

---

### 3.9 Model 9: `ace-step-1.5`
- **Manifest:** `manifests/models/ace-step-1.5.yaml`
- **Model ID:** `ace-step-1.5` | **Service ID:** `music-service` | **Service Port:** `8008`
- **Primary Checkpoint:** `ACE-Studio/ACE-Step-1.5`
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `bf16`
- **Native Backend:** Native ACE-Step HTTP API (`run_acestep_api.sh`, PyTorch, torchaudio)
- **Target GPU:** GPU 1 | **Min VRAM:** 6.0 GB (base/turbo) to 12.0 GB (xl_base) | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`, `base` (2B Base, 50 steps), `turbo` (2B Turbo, 8 steps), `sft` (2B SFT, 25 steps), `xl_base` (XL Base, 50 steps)
- **Task Matrix & Strict Variant Restrictions**:
  - `text_to_music`: `['standard', 'base', 'turbo', 'sft', 'xl_base']`
  - `lyrics_to_music`: `['standard', 'base', 'turbo', 'sft', 'xl_base']`
  - `reference_audio`: `['standard', 'base', 'turbo', 'sft', 'xl_base']`
  - `cover`: `['standard', 'base', 'turbo', 'sft', 'xl_base']`
  - `repaint`: `['standard', 'base', 'turbo', 'sft', 'xl_base']`
  - `extract`: `['base', 'xl_base']` **(STRICTLY REJECT on `turbo` / `sft`)**
  - `lego`: `['base', 'xl_base']` **(STRICTLY REJECT on `turbo` / `sft`)**
  - `complete`: `['base', 'xl_base']` **(STRICTLY REJECT on `turbo` / `sft`)**
  - `music_extend`: `['base', 'xl_base']` **(STRICTLY REJECT on `turbo` / `sft`)**
- **Capability Truth (Coverage: 81.82%)**:
  - 9 specialized generative tasks: `true`
  - `audio_to_audio`: `false` (unbacked legacy endpoint)
  - `audio_to_music`: `false` (unbacked legacy endpoint)
- **Fail-Closed Safeguard:** Submitting `extract`, `lego`, `complete`, or `music_extend` with variant `turbo` or `sft` immediately raises `UnsupportedCapabilityError` with HTTP 400.

---

### 3.10 Model 10: `yue2`
- **Manifest:** `manifests/models/yue2.yaml`
- **Model ID:** `yue2` | **Service ID:** `music-service` | **Service Port:** `8009`
- **Primary Checkpoint:** `m-a-p/YuE-s1-7B-anneal-en-cot`
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `bf16`
- **Native Backend:** Native YuE2 HTTP Client / PyTorch (`xformers`, `transformers`)
- **Target GPU:** GPU 1 | **Min VRAM:** 12.0 GB (declared for T4) / 24.0 GB (native) | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `text_to_music`: `standard`
  - `lyrics_to_music`: `standard`
  - `symbolic_score`: `standard` (ABC notation score parsing & generation)
  - `custom_score`: `standard`
  - `score_edit`: `standard` (measure/chord diffing)
  - `song_variation`: `standard`
- **Capability Truth (Coverage: 85.71%)**:
  - 6 music & score tasks: `true`
  - `cover_edit`: `false`
- **Hardware Feasibility Gate Y0**: Unquantized 7B BF16 requires ~14 GB weights + 2–4 GB activations, pushing past the 14.2 GB allocatable ceiling on T4. If memory exceeds 14.2 GB during Phase 3 Kaggle benchmark, it must be reported honestly as `HARDWARE_BLOCKED` rather than simulated.

---

### 3.11 Model 11: `enhancement-suite-v1`
- **Manifest:** `manifests/models/enhancement-suite-v1.yaml`
- **Model ID:** `enhancement-suite-v1` | **Service ID:** `enhancement-service` | **Service Port:** `8002`
- **Primary Checkpoint:** `xinntao/Real-ESRGAN`
- **Components:** `rife: hzwer/RIFE`, `bsrgan: cszn/BSRGAN`, `audio_denoise: Rikorose/DeepFilterNet`
- **Weights Format:** `safetensors` | **Quantization:** `none` | **Precision:** `fp16`
- **Native Backend:** Custom enhancement HTTP worker (unconfigured)
- **Target GPU:** GPU 1 | **Min VRAM:** 6.0 GB | **Alloc Policy:** `single_gpu`
- **Registered Variants:** `standard`
- **Task Matrix:**
  - `image_super_resolution`: `standard`
  - `video_super_resolution`: `standard`
  - `frame_interpolation`: `standard`
  - `audio_enhancement`: `standard`
- **Capability Truth (Coverage: 0.0%)**:
  - `installed`: `false`
  - `checkpoint_valid`: `false`
  - `backend_ready`: `false`
  - All 4 tasks: `declared: true`, `implemented: false`, `live_verified: false`
- **Fail-Closed Safeguard:** All enhancement endpoints MUST return explicit `BACKEND_UNAVAILABLE` (503) or `NOT_IMPLEMENTED` (501). Zero mock upscaled frames or synthetic waveforms permitted.

---

## 4. Capability Truth & Fail-Closed Enforcement Rules

To satisfy the strict Integrity Mandate across all 11 foundation models:

1. **Negative Capability Enforcement**:
   - Router and worker adapters must verify requested tasks against `capabilities_truth`.
   - If `capabilities_truth[task].implemented == False`, the request MUST fail fast with `UnsupportedCapabilityError` (HTTP 400).
2. **Variant Restriction Enforcement**:
   - For models with task-restricted variants (notably `ace-step-1.5`), requesting an unmapped variant (e.g. `turbo` for `extract`) MUST fail closed with HTTP 400.
3. **Backend Availability Probe**:
   - `ModelRuntimeManager` must probe the live backend health endpoint before marking `READY`.
   - If the engine process is not responding, return HTTP 503 `BACKEND_UNAVAILABLE` without falling back to synthetic tokens or dummy images.
4. **Hardware Feasibility Enforcement**:
   - If required VRAM exceeds 14.2 GB on the assigned GPU, the job must fail closed with `HARDWARE_BLOCKED` rather than attempting an unsafe allocation or fabricating a PASS.
