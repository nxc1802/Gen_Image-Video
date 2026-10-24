
# Factory v4 — Complete Refactor Plan

## 1. Mục tiêu của v4

Factory v4 sẽ có 8 nguyên tắc bắt buộc:

1. **Không còn một monolithic FastAPI tự implement inference cho mọi model.**
2. **Mỗi AI subsystem dùng native/standard inference framework phù hợp nhất.**
3. **Mỗi subsystem có API domain cố định**, không đổi sau mỗi lần restart Kaggle.
4. **Security bắt buộc từ Edge đến Worker**, không còn chế độ public unauthenticated.
5. **Queue và concurrency là thành phần kiến trúc chính**, không phải workaround.
6. **Streaming là first-class feature.**
7. **Model được khai báo bằng capability manifest**, không giả lập tính năng model không hỗ trợ.
8. **Frontend là API Studio + Operations Console**, không chỉ là demo UI.

---

# 2. Kiến trúc mục tiêu

```text
                         ┌────────────────────────┐
                         │      Factory Studio     │
                         │ Next.js / React         │
                         └───────────┬────────────┘
                                     │
                         ┌───────────▼────────────┐
                         │   Factory Control API   │
                         │ Jobs / Registry / Auth  │
                         │ Assets / Observability  │
                         └──────┬───────────┬─────┘
                                │           │
                         PostgreSQL      Redis
                         Jobs/Models      Events
                                │
       ┌────────────────────────┼───────────────────────────┐
       │                        │                           │
       ▼                        ▼                           ▼
 text.factory.*          image.factory.*           music.factory.*
 ngrok static            ngrok static              ngrok static
       │                        │                           │
     vLLM                    ComfyUI                  ACE-Step
       │                        │                           │
 LLM / Responses          T2I / I2I / Edit           Music Gen
 VLM / Decisions          Inpaint / Ref              Audio2Audio
                          etc.
```

Quan trọng: **Control Plane không inference**.

Nó chỉ quản lý:

- authentication;
- model/service registry;
- job lifecycle;
- queue/admission;
- artifact storage;
- worker health;
- metrics;
- logs;
- frontend API.

Inference vẫn chạy hoàn toàn trong framework chuẩn của từng model.

---

# 3. Phân chia subsystem mới

| Subsystem | Framework chính | API chuẩn |
|---|---|---|
| Text / LLM | **vLLM** | OpenAI Chat + Responses |
| VLM | **vLLM** | OpenAI multimodal Chat/Responses |
| Decision | **vLLM Decisions** + provider adapters | `/v1/decisions` |
| Embedding | **vLLM hoặc TEI** | `/v1/embeddings`, `/v2/embed` |
| Rerank | **vLLM/TEI** | `/rerank`, `/v1/rerank`, `/v2/rerank` |
| ASR | **Qwen3-ASR + vLLM** | `/v1/audio/transcriptions` |
| TTS | **vLLM-Omni** | `/v1/audio/speech` |
| Image | **ComfyUI** | Native Queue / Prompt / WS |
| Video | **ComfyUI** | Native Queue / Prompt / WS |
| Music | **ACE-Step 1.5** | Native async REST |
| Enhancement | ComfyUI / dedicated engines | Async Jobs |
| Factory Jobs | Factory Control API | `/v1/jobs/*` |

vLLM hiện đã support cả Decisions API dạng predicate/choice/score bên cạnh Chat, Responses, Embeddings, ASR và reranking APIs.

---

# 4. Decision subsystem

Đây nên là subsystem mới hoàn toàn:

```text
services/decision/
```

API chuẩn:

```text
POST /v1/decisions
```

Task:

```text
predicate
choice
score
```

Ví dụ use case:

```text
request routing
model routing
risk classification
priority scoring
quality gate
agent selection
tool selection
```

vLLM đã implement `/v1/decisions` cho các generation architecture được hỗ trợ, ví dụ Qwen.

### Model strategy

Không hardcode một Decision model.

Registry:

```yaml
decision_models:

  qwen-small:
    backend: vllm
    capabilities:
      predicate: true
      choice: true
      score: true
      vision: false

  specialized-decision:
    backend: vllm
    capabilities:
      predicate: true
      choice: true
      score: true

  hosted-multimodal:
    backend: provider
    capabilities:
      vision: true
```

Có thể benchmark các model chuyên dụng sau.

Không chọn model chỉ dựa trên tên hoặc vendor claim.

Decision model phải qua:

```text
Decision Accuracy
Calibration
Latency
Throughput
VRAM
License
```

---

# 5. Music subsystem

Đề xuất chính:

# ACE-Step 1.5

Lý do rất phù hợp với Factory:

- có native REST API server;
- có async task API;
- text → music;
- lyrics;
- audio conditioning;
- Audio2Audio;
- model khá nhẹ;
- có thể chạy từ khoảng 4–6 GB VRAM tùy chế độ.

ACE-Step 1.5 chính thức có standalone REST server `acestep-api`; API còn sử dụng cơ chế submit task → `task_id` → query result, tức đã có pattern phù hợp cho long-running generation.

Do đó:

```text
Không viết lại ACE-Step bằng FastAPI của Factory.
```

Factory chỉ register nó như một native worker.

---

# 6. Speech subsystem

Tách Speech thành:

```text
speech/
├── asr
├── tts
└── alignment
```

### ASR

```text
Qwen3-ASR
   ↓
vLLM
```

Qwen3-ASR hiện có official vLLM serving và hỗ trợ streaming inference.

Capability:

```text
Speech → Text
Language Detection
Streaming ASR
Long Audio
Timestamp
Forced Alignment
Translation
```

Forced Alignment:

```text
Qwen3-ForcedAligner
```

---

### TTS

Không duy trì `audio_manager.py` kiểu cũ nữa.

Dùng:

```text
Qwen3-TTS
   ↓
vLLM-Omni
```

vLLM-Omni hiện expose:

```text
POST /v1/audio/speech
```

và hỗ trợ Qwen3-TTS, voice cloning, voice preset/upload và streaming.

Capability contract:

```text
TTS
Streaming TTS
Voice Clone
Custom Voice
Voice Design
Voice Preset
Reference Audio
```

---

# 7. Image subsystem

Image sẽ không còn code dạng:

```text
qwen_image.engine.py
→ manually create workflow
→ manually call Comfy API
→ custom OpenAI response
```

Thay bằng:

```text
ComfyUI
    +
Workflow Registry
    +
Capability Manifest
```

ComfyUI bản thân đã có execution queue và WebSocket event/progress model.

## Required Image capabilities

Một **logical model family** chỉ được đưa vào production nếu có đầy đủ capability yêu cầu.

Ví dụ:

```text
T2I
I2I
Text+Image → Image
Inpainting
Outpainting
Instruction Edit
Reference Image
Multi Reference
Style Reference
ControlNet
Pose Conditioning
Depth Conditioning
Canny Conditioning
Batch Generation
Seed Reproducibility
```

---

## Quan trọng: “1 model” nên được định nghĩa là Model Family

Không nên ép **một checkpoint vật lý** làm những task kiến trúc của nó không support.

Ví dụ:

```text
qwen-image-2.1
```

có thể là logical family:

```text
Qwen Image Base
Qwen Image Edit
Control adapter
VAE
Text encoder
Workflow definitions
```

Nhưng bên ngoài vẫn xuất hiện là:

```text
qwen-image-2.1
```

Capability router tự chọn workflow/checkpoint thích hợp.

Điều này đảm bảo:

> Model family cover đầy đủ task, nhưng hệ thống không giả vờ rằng một weight duy nhất làm được tất cả.

---

# 8. Video subsystem

Tương tự Image:

```text
ComfyUI native server
+
Workflow Registry
```

Required capabilities:

```text
T2V
I2V
First/Last Frame → Video
Reference → Video
Multi-reference → Video
Video Extension
Video Chaining
Video + Audio
Motion Guidance
Video-to-Video
Video Edit
Seed Reproducibility
```

Candidate families:

```text
Wan
MiniMax H3
future video models
```

Model nào thiếu capability:

```text
capabilities:
  video_to_video: false
```

Factory không được route request đó vào model này.

---

# 9. Enhancement subsystem

Tôi khuyên bổ sung subsystem riêng:

```text
enhancement.factory.*
```

Task:

```text
Image Upscale
Video Upscale
Frame Interpolation
Video Denoise
Image Restoration
Face Restoration
Audio Enhancement
```

Như vậy Image/Video Generation không bị trộn với post-processing.

Pipeline sau này:

```text
Generate
   ↓
Upscale
   ↓
Interpolate
   ↓
Audio
   ↓
Mux
   ↓
Final Output
```

---

# 10. Fixed API Domain

Cloudflare Quick Tunnel bị loại bỏ hoàn toàn.

Dùng:

# ngrok static/reserved domains

Ví dụ:

```text
text.ai.example.com
vision.ai.example.com
decision.ai.example.com
embed.ai.example.com
speech.ai.example.com
image.ai.example.com
video.ai.example.com
music.ai.example.com
enhance.ai.example.com

jobs.ai.example.com
studio.ai.example.com
```

Kaggle restart:

```text
Worker offline
→ Worker restart
→ ngrok agent reconnect
→ SAME URL
```

ngrok hỗ trợ domain đã đăng ký/reserved; Traffic Policy cũng có thể xử lý auth, IP restrictions và rate limiting tại edge.

Lưu ý về giá: free plan hiện chỉ có **một assigned dev domain**; nhiều custom/reserved static domains cần paid plan.

Nếu muốn đúng yêu cầu **mỗi subsystem một URL cố định**, tôi khuyến nghị paid ngrok + domain riêng.

ngrok cũng phù hợp hơn Quick Tunnel trong trường hợp này vì stable dev domain hỗ trợ SSE và các HTTP/WebSocket workloads; tài liệu ngrok cũng ghi rõ WebSocket được hỗ trợ.

---

# 11. Security Architecture

P0 phải được giải quyết trước khi migrate model.

## Edge

```text
ngrok
```

enforce:

```text
JWT validation
Rate limiting
IP restriction cho admin
Request size
WAF
TLS
```

---

## Application layer

Tất cả API:

```text
Authorization: Bearer ...
```

Không có:

```text
API_KEY=""
→ allow everything
```

---

## Bỏ hoàn toàn x-target-base-url

Frontend hiện tại cho client quyết định upstream.

Factory v4:

```text
Service Registry
```

mới là nơi giữ URL.

Client chỉ gửi:

```text
service = image
```

Control Plane resolve:

```text
image → https://image.ai.example.com
```

Không có arbitrary proxy.

---

## Không fallback credential sang domain khác

Pattern hiện tại:

```text
target A fail
→ retry target B
→ reuse Authorization
```

bị cấm hoàn toàn.

---

## Remote asset security

Model server không trực tiếp fetch URL arbitrary.

Thay vào đó:

```text
Client
 ↓
Asset Service
 ↓
validate MIME / size / URL
 ↓
Object Storage
 ↓
Model
```

Chống:

```text
SSRF
oversized file
malformed media
internal network access
```

---

## Error handling

Production API không trả:

```text
traceback
absolute path
stack trace
exception internals
```

Chỉ:

```json
{
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "request_id": "..."
  }
}
```

Trace thật lưu trong logs.

---

# 12. Queue architecture

Không có một queue duy nhất cố override native engine.

Factory dùng **hai tầng queue**.

## Tier 1 — Native Engine Queue

### vLLM

Dùng:

```text
continuous batching
scheduler
request concurrency
```

Không tự viết queue.

### ComfyUI

Dùng:

```text
/prompt
/queue
/history
/ws
```

Không tự viết queue.

### ACE-Step

Dùng async task system của ACE-Step.

---

# Tier 2 — Factory Job Layer

Cho orchestration dài hạn.

Stack:

```text
PostgreSQL
+
Redis
```

PostgreSQL:

```text
job state
metadata
artifact
model
worker
retry
timestamps
```

Redis:

```text
job events
progress
worker heartbeat
short-lived locks
stream events
```

Job lifecycle:

```text
CREATED
   ↓
QUEUED
   ↓
ADMITTED
   ↓
RUNNING
   ↓
STREAMING
   ↓
SUCCEEDED

or

FAILED
CANCELLED
RETRYING
EXPIRED
```

---

# 13. Job contract

Mọi long-running request có:

```text
job_id
request_id
idempotency_key
service
model
task
priority
created_at
started_at
progress
worker
retry_count
artifacts
```

API:

```text
POST   /v1/jobs
GET    /v1/jobs/{id}
DELETE /v1/jobs/{id}
POST   /v1/jobs/{id}/retry

GET    /v1/jobs/{id}/events
WS     /v1/jobs/{id}/ws
```

---

# 14. Concurrency policy

Điều cần phân biệt:

> “Handle nhiều request đồng thời” không có nghĩa GPU phải inference tất cả cùng lúc.

Factory phải có thể:

```text
nhận 100 requests
```

mà không crash.

Nhưng GPU có thể xử lý:

```text
1–N request thực tế
```

tuỳ engine.

Ví dụ:

### Video GPU

```text
100 submitted
1 running
99 queued
```

### LLM

```text
100 submitted
vLLM continuous batching
```

### Embedding

```text
dynamic batching
```

### Music

```text
ACE-Step native async queue
```

---

# 15. Streaming

Streaming phải được phân loại.

## Token Streaming

```text
LLM / VLM / Decision where applicable
```

SSE.

---

## Audio Streaming

```text
ASR
TTS
```

WebSocket / chunked PCM.

---

## Progress Streaming

```text
Image
Video
Music
Enhancement
```

SSE hoặc WS:

```json
{
  "job_id": "...",
  "event": "progress",
  "stage": "sampling",
  "step": 11,
  "total": 20,
  "percent": 55
}
```

---

# 16. Artifact handling

Không trả video lớn dưới dạng base64 JSON.

Dùng:

```text
S3-compatible Object Storage
```

Kết quả:

```json
{
  "job_id": "...",
  "artifact": {
    "type": "video",
    "url": "...",
    "mime": "video/mp4",
    "size": 1234567
  }
}
```

Base64 chỉ giữ cho:

```text
small image
small test payload
```

---

# 17. Model Registry

Mọi model có manifest.

Ví dụ:

```yaml
id: qwen-image-2.1
service: image

backend:
  type: comfyui

hardware:
  min_vram_gb: 14
  preferred_gpu: t4

capabilities:
  text_to_image: true
  image_to_image: true
  inpaint: true
  outpaint: true
  instruction_edit: true
  multi_reference: true
  control_pose: true
  control_depth: true

streaming:
  progress: true

limits:
  max_width: 2048
  max_height: 2048
```

Frontend đọc manifest này.

Không hardcode model options trong React nữa.

---

# 18. Folder Structure v4

```text
factory/
│
├── apps/
│   ├── control-api/
│   └── studio-web/
│
├── services/
│   ├── text/
│   │   ├── models/
│   │   ├── configs/
│   │   └── launcher/
│   │
│   ├── vlm/
│   ├── decision/
│   ├── embedding/
│   ├── speech/
│   │   ├── asr/
│   │   ├── tts/
│   │   └── alignment/
│   │
│   ├── image/
│   ├── video/
│   ├── music/
│   └── enhancement/
│
├── workflows/
│   ├── image/
│   │   ├── t2i/
│   │   ├── i2i/
│   │   ├── inpaint/
│   │   ├── outpaint/
│   │   └── control/
│   │
│   └── video/
│       ├── t2v/
│       ├── i2v/
│       ├── fl2v/
│       └── ref2v/
│
├── packages/
│   ├── contracts/
│   ├── model-registry/
│   ├── service-registry/
│   ├── job-client/
│   ├── auth/
│   ├── observability/
│   ├── asset-client/
│   └── sdk/
│
├── infrastructure/
│   ├── ngrok/
│   ├── redis/
│   ├── postgres/
│   ├── storage/
│   ├── docker/
│   └── kaggle/
│
├── deployments/
│   ├── kaggle/
│   ├── colab/
│   └── local/
│
├── manifests/
│   ├── models/
│   ├── services/
│   └── hardware/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   ├── load/
│   ├── security/
│   └── gpu/
│
├── benchmarks/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── models/
│   └── operations/
│
└── legacy/
```

---

# 19. Sub-Agent Organization

Đề xuất sử dụng 11 Sub Agent cố định.

| Agent | Responsibility |
|---|---|
| `SA-ARCH` | Architecture / contracts |
| `SA-SEC` | Security |
| `SA-PLATFORM` | Kaggle / ngrok / deployment |
| `SA-JOBS` | Queue / jobs / storage |
| `SA-TEXT` | LLM / VLM |
| `SA-DECISION` | Decision models |
| `SA-SPEECH` | ASR / TTS |
| `SA-MEDIA` | Image / Video |
| `SA-MUSIC` | Music / Enhancement |
| `SA-FRONTEND` | Studio Web |
| `SA-QA` | Test / benchmark / validation |

`SA-ARCH` đóng vai reviewer chính.

Không agent nào tự merge thay đổi kiến trúc chung nếu chưa pass contract test.

---

# WAVE 0 — Freeze, Audit & Specification

## SA-ARCH

Task:

```text
W0-A1 Freeze v3.5
W0-A2 Tag legacy release
W0-A3 Inventory toàn bộ modules
W0-A4 Inventory endpoint
W0-A5 Inventory model
W0-A6 Dependency graph
W0-A7 Define v4 contracts
```

Deliverables:

```text
docs/architecture/v3-audit.md
docs/architecture/v4-target.md
manifests/service-capability-matrix.yaml
```

---

## SA-SEC

Task:

```text
W0-S1 Threat model
W0-S2 Map toàn bộ P0
W0-S3 Map toàn bộ P1
W0-S4 Secret inventory
W0-S5 Endpoint exposure inventory
```

---

## SA-QA

Task:

```text
W0-Q1 Capture baseline API behavior
W0-Q2 Capture current benchmark
W0-Q3 Save golden requests
W0-Q4 Save expected response schemas
```

### Wave 0 Exit Gate

Phải biết chính xác:

```text
what exists
what works
what is broken
what is removed
what is migrated
```

---

# WAVE 1 — Foundation & Clean Architecture

## SA-ARCH

```text
W1-A1 Create new folder structure
W1-A2 Define ServiceManifest
W1-A3 Define ModelManifest
W1-A4 Define CapabilityManifest
W1-A5 Define common error contract
W1-A6 Define job contract
```

---

## SA-PLATFORM

```text
W1-P1 uv / lock strategy
W1-P2 isolated environment per service
W1-P3 remove global requirements.txt
W1-P4 remove global STUDIO_MODE
W1-P5 create independent launchers
```

---

## SA-SEC

```text
W1-S1 mandatory auth
W1-S2 CORS allowlist
W1-S3 request validation
W1-S4 upload limits
W1-S5 secret abstraction
W1-S6 error sanitization
```

---

## SA-QA

```text
W1-Q1 schema tests
W1-Q2 manifest validation
W1-Q3 auth tests
W1-Q4 configuration tests
```

### Wave 1 Exit Gate

Không còn:

```text
STUDIO_MODE
duplicated models.yaml
unauthenticated admin
global MemoryManager
fake inference fallback
```

---

# WAVE 2 — Networking, Fixed APIs & Security Edge

## SA-PLATFORM

```text
W2-P1 ngrok agent installation
W2-P2 static domain per subsystem
W2-P3 worker health registration
W2-P4 automatic reconnect
W2-P5 restart recovery
W2-P6 domain configuration manifests
```

---

## SA-SEC

```text
W2-S1 ngrok Traffic Policy
W2-S2 JWT/API-key edge validation
W2-S3 rate limiting
W2-S4 admin IP restriction
W2-S5 disable request capture where sensitive
```

---

## SA-QA

```text
W2-Q1 reboot worker
W2-Q2 verify URL unchanged
W2-Q3 test SSE
W2-Q4 test WebSocket
W2-Q5 auth bypass tests
W2-Q6 SSRF tests
```

### Exit Gate

Restart Kaggle 10 lần:

```text
API URL vẫn không thay đổi
```

---

# WAVE 3 — Factory Jobs, Queue, Storage & Observability

## SA-JOBS

```text
W3-J1 PostgreSQL job schema
W3-J2 Redis event architecture
W3-J3 state machine
W3-J4 retries
W3-J5 cancellation
W3-J6 priorities
W3-J7 idempotency
W3-J8 worker heartbeat
W3-J9 worker leases
W3-J10 dead-job recovery
```

---

## SA-JOBS — Streaming

```text
W3-J11 SSE progress
W3-J12 WebSocket progress
W3-J13 reconnect/resume
W3-J14 event history
```

---

## SA-PLATFORM

```text
W3-P1 S3-compatible storage
W3-P2 signed artifact URLs
W3-P3 cleanup policy
W3-P4 upload service
```

---

## SA-ARCH

```text
W3-A1 OpenTelemetry schema
W3-A2 standard metrics
W3-A3 request correlation ID
```

Metrics:

```text
request latency
queue wait
inference time
GPU memory
GPU utilization
tokens/sec
frames/sec
job failure rate
worker uptime
```

---

## SA-QA

Stress test:

```text
100 simultaneous submissions
worker restart
Redis restart
failed model
timeout
cancel
retry
duplicate request
```

### Exit Gate

Không request nào bị mất.

---

# WAVE 4 — Text, VLM, Decision, Embedding & Speech Migration

## SA-TEXT

```text
W4-T1 vLLM text service
W4-T2 Chat Completions
W4-T3 Responses API
W4-T4 streaming
W4-T5 tool calling
W4-T6 structured output
W4-T7 VLM input
W4-T8 model registry
```

---

## SA-DECISION

```text
W4-D1 vLLM Decisions server
W4-D2 predicate tests
W4-D3 choice tests
W4-D4 score tests
W4-D5 calibration benchmark
W4-D6 Decision model comparison
W4-D7 fallback policy
```

Quan trọng:

```text
fallback model
```

được phép.

```text
fake heuristic result
```

không được phép.

---

## SA-SPEECH

```text
W4-S1 Qwen3-ASR on vLLM
W4-S2 streaming ASR
W4-S3 timestamps
W4-S4 forced alignment
W4-S5 Qwen3-TTS vLLM-Omni
W4-S6 streaming TTS
W4-S7 voice clone
W4-S8 voice design
```

---

## SA-TEXT / Embedding Worker

```text
W4-E1 Qwen embedding
W4-E2 rerank
W4-E3 multimodal embedding
W4-E4 proper model failure
W4-E5 remove deterministic fake vectors
W4-E6 batch benchmark
```

### Exit Gate

Nếu model không load:

```text
503 MODEL_UNAVAILABLE
```

Tuyệt đối không trả embedding giả.

---

# WAVE 5 — Image, Video, Music & Enhancement

## SA-MEDIA — Image

```text
W5-I1 clean ComfyUI installation
W5-I2 Qwen Image workflow registry
W5-I3 T2I
W5-I4 I2I
W5-I5 inpaint
W5-I6 outpaint
W5-I7 instruction edit
W5-I8 reference
W5-I9 multi-reference
W5-I10 control workflows
```

---

## SA-MEDIA — Video

```text
W5-V1 clean ComfyUI video worker
W5-V2 Wan workflows
W5-V3 MiniMax workflows
W5-V4 T2V
W5-V5 I2V
W5-V6 FL2V
W5-V7 reference
W5-V8 multi-reference
W5-V9 chaining
W5-V10 extension
W5-V11 audio
W5-V12 native progress WS
```

---

## SA-MUSIC

```text
W5-M1 ACE-Step installation
W5-M2 native REST API
W5-M3 music queue
W5-M4 text-to-music
W5-M5 lyric generation
W5-M6 audio-to-audio
W5-M7 continuation/edit
W5-M8 artifact integration
```

---

## SA-MUSIC / Enhancement

```text
W5-X1 image upscale
W5-X2 video upscale
W5-X3 frame interpolation
W5-X4 restoration
W5-X5 audio enhancement
```

---

# WAVE 6 — Capability Completion & Model Expansion

Đây là Wave quan trọng để giải quyết yêu cầu:

> “Model phải được cover đầy đủ task.”

## SA-ARCH

Tạo Required Capability Matrix.

Ví dụ Image:

| Capability | Required |
|---|---:|
| T2I | ✅ |
| I2I | ✅ |
| Inpaint | ✅ |
| Outpaint | ✅ |
| Instruction Edit | ✅ |
| Reference | ✅ |
| Multi-reference | ✅ |
| Control | ✅ |

---

Mỗi model family phải đạt:

```text
100% REQUIRED capabilities
```

hoặc:

```text
không được label FULL
```

Ví dụ:

```text
qwen-image-2.1
coverage: 100%
status: full
```

hoặc:

```text
model-x
coverage: 62%
status: partial
```

Frontend phải hiển thị rõ.

---

## SA-QA

Mỗi capability có automated contract test.

Ví dụ:

```text
test_qwen_t2i
test_qwen_i2i
test_qwen_inpaint
test_qwen_outpaint
...
```

Không còn report dạng:

```text
"15/15 pass"
```

chỉ do người viết tự đánh giá.

Kết quả phải lưu:

```text
request
model version
workflow hash
seed
runtime
GPU
output hash
metrics
artifact
```

---

# WAVE 7 — Factory Studio Frontend

Frontend không còn hardcode model.

Model và capability load từ Registry.

## SA-FRONTEND

### Main dashboard

```text
Workers
Services
Models
GPU
Queue
Jobs
Errors
```

---

### API Explorer

Chọn:

```text
Subsystem
→ Model
→ Task
```

UI tự render parameters từ schema.

Ví dụ:

```text
Image
  Qwen Image
    Text to Image
    Image to Image
    Inpaint
    Outpaint
```

---

### Streaming Inspector

Hiển thị:

```text
SSE
WebSocket events
tokens
audio chunks
ComfyUI progress
queue state
```

---

### Job Dashboard

```text
Queued
Running
Completed
Failed
Cancelled
```

Có:

```text
retry
cancel
download
copy request
re-run
```

---

### API Code Generator

Tự tạo:

```text
curl
Python
JavaScript
OpenAI SDK
```

---

### Asset Browser

```text
Images
Videos
Audio
Music
```

Có metadata:

```text
model
prompt
seed
workflow
duration
generation time
```

---

### Model Inspector

Hiển thị:

```text
Backend
Quantization
VRAM
Capabilities
Loaded status
Health
API URL
```

---

# WAVE 8 — Hardening, Benchmark & Release

## SA-QA

### Security

```text
Auth bypass
SSRF
CORS
Oversized upload
Malicious URL
Admin exposure
Rate limit
Token leakage
```

---

### Load

```text
100 concurrent jobs
500 queued jobs
long-lived SSE
WebSocket reconnect
worker crash
worker restart
```

---

### GPU

```text
OOM
VRAM fragmentation
model load
unload
multi-GPU
```

---

### Failure Injection

```text
kill ComfyUI
kill vLLM
kill Redis
kill ngrok
disconnect internet
stop Kaggle
```

Factory phải recover đúng state.

---

## SA-PLATFORM

```text
W8-P1 deployment manifests
W8-P2 one-command service startup
W8-P3 Kaggle setup
W8-P4 local setup
W8-P5 Colab setup
```

---

## SA-ARCH

Remove old architecture:

```text
MemoryManager
ModelRegistry v3
Cloudflare quick tunnel
custom embedding fake fallback
global server/app.py
STUDIO_MODE
old notebook payload packer
```

Legacy được giữ:

```text
legacy/v3/
```

không sử dụng runtime.

---

# 20. Dependency graph giữa các Wave

```text
Wave 0
  │
  ▼
Wave 1
  │
  ├──────────────┐
  ▼              ▼
Wave 2          Wave 3
  │              │
  └──────┬───────┘
         ▼
       Wave 4
         │
         ▼
       Wave 5
         │
         ▼
       Wave 6
         │
         ▼
       Wave 7
         │
         ▼
       Wave 8
```

Wave 2 và Wave 3 có thể chạy song song.

Trong Wave 4:

```text
SA-TEXT
SA-DECISION
SA-SPEECH
```

có thể chạy song song.

Trong Wave 5:

```text
SA-MEDIA Image
SA-MEDIA Video
SA-MUSIC
```

cũng chạy song song.

---

# 21. Pull Request strategy

Mỗi Task = một branch.

Ví dụ:

```text
wave4/decision-vllm
wave4/qwen-asr-vllm
wave5/image-qwen-workflows
wave5/music-acestep
```

Không tạo branch:

```text
refactor-everything
```

Mỗi PR phải pass:

```text
lint
unit
contract
security
smoke
```

trước merge.

---

# 22. Acceptance criteria toàn dự án

Factory v4 chỉ coi là hoàn tất khi:

### Architecture

```text
0 monolithic inference server
0 global mode switch
0 duplicated config
```

### Security

```text
0 unauthenticated public inference
0 arbitrary proxy
0 credential forwarding fallback
0 fake inference
```

### Network

```text
100% fixed API URLs
```

### Queue

```text
100+ requests accepted safely
cancel/retry/resume supported
```

### Streaming

```text
LLM streaming
ASR streaming
TTS streaming
Image progress
Video progress
Music progress
```

### Capability

```text
Every FULL model family:
100% required capability tests passed
```

### Frontend

```text
100% APIs testable from Studio
```

### Observability

```text
request → job → worker → model → artifact
```

phải trace được end-to-end.

---

# 23. Kiến trúc cuối cùng tôi đề xuất

```text
                        FACTORY CONTROL PLANE
                  ┌───────────────────────────┐
                  │ Auth / Jobs / Assets      │
                  │ Registry / Metrics        │
                  └─────────────┬─────────────┘
                                │
        ┌───────────────┬───────┼────────┬───────────────┐
        │               │       │        │               │
        ▼               ▼       ▼        ▼               ▼
      TEXT            SPEECH   IMAGE    VIDEO           MUSIC
      vLLM          vLLM/Omni ComfyUI  ComfyUI        ACE-Step
        │               │       │        │               │
       ngrok           ngrok   ngrok    ngrok           ngrok
     fixed URL       fixed URL fixed URL fixed URL     fixed URL
```

Bên trên:

```text
Factory Studio
```

Bên dưới:

```text
Kaggle GPU workers
```

Ở giữa:

```text
PostgreSQL
Redis
S3 Storage
```

Đây mới là kiến trúc tôi cho rằng phù hợp với mục tiêu **“AI Factory”** của dự án.

---

# 24. Thứ tự ưu tiên tuyệt đối

Nếu bắt đầu thực hiện ngay, thứ tự phải là:

**P0**

```text
Wave 0
→ Wave 1 Security/Foundation
→ Wave 2 Fixed API
```

**P1**

```text
Wave 3 Queue/Jobs
→ Wave 4 Core AI migration
```

**P2**

```text
Wave 5 New AI subsystems
→ Wave 6 capability completeness
```

**P3**

```text
Wave 7 UI
→ Wave 8 hardening
```

Không nên bắt đầu bằng việc thêm model mới.

Nếu thêm Decision/Music trước khi làm Wave 1–3 thì kiến trúc cũ chỉ tiếp tục phình to và lần refactor tiếp theo sẽ còn khó hơn.