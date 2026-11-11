# Factory — Master Upgrade & Refactor Plan

## 0. Mục tiêu cuối cùng

Factory được đưa về mô hình:

> **OpenAI-compatible, single-host / multi-GPU AI inference platform**, tối ưu trước tiên cho **Kaggle Dual-T4**, phục vụ cá nhân và khoảng 2–3 ứng dụng, nhưng kiến trúc đủ generic để sau này mở rộng sang phần cứng khác mà không phải viết lại core.

Factory không phải enterprise cloud platform, không phải multi-tenant SaaS và cũng không phải wrapper riêng cho từng model.

Mục tiêu kiến trúc:

```text
Application
    │
    │ OpenAI SDK / HTTP
    ▼
OpenAI-Compatible API
    │
    ▼
Request Normalizer
    │
    ▼
Capability Resolver
    │
    ▼
Active Deployment Registry
    │
    ├── model available?
    ├── capability supported?
    └── deployment healthy?
    │
    ▼
Functional Engine
    │
    ▼
Model Adapter
    │
    ▼
Preconfigured Backend
    │
    ▼
Preassigned GPU(s)
    │
    ▼
Real Inference
```

Deployment được quyết định trước:

```text
Model Catalog
    │
    ▼
Factory CLI
    │
    ├── select models
    ├── assign GPU(s)
    ├── validate weights
    ├── validate VRAM
    ├── launch backend
    └── health check
    │
    ▼
Active Deployment Registry
```

Không có dynamic model routing trong request path.

---

## 1. Nguyên tắc không được phá vỡ

### 1.1 Public API ưu tiên OpenAI compatibility

Nếu OpenAI đã có API cho capability tương ứng thì Factory phải bám theo contract đó càng sát càng tốt.

Ứng dụng đang dùng OpenAI SDK phải có thể chuyển sang Factory chủ yếu bằng:

```python
client = OpenAI(
    api_key=FACTORY_API_KEY,
    base_url=FACTORY_BASE_URL
)
```

sau đó thay `model`.

Không yêu cầu ứng dụng hiểu:

```text
FactoryJob
service
GPU
backend
ComfyUI
vLLM
llama.cpp
worker
queue
```

Các khái niệm này là implementation detail.

### 1.2 Engine thuộc capability, không thuộc model

Sai:

```text
WanEngine
MiniMaxEngine
QwenEngine
ACEEngine
```

Đúng:

```text
ResponsesEngine
DecisionEngine
VectorEngine
SpeechEngine
ImageEngine
VideoEngine
MusicEngine
EnhancementEngine
```

Model-specific behavior nằm trong Adapter.

### 1.3 Thêm model mới không được sửa Engine

Target:

```text
new model
    ↓
manifest
+ adapter
+ optional workflows
+ tests
```

Không:

```text
edit VideoEngine
edit scheduler
edit API
edit worker
edit model maps
edit Studio manually
```

### 1.4 Một Source of Truth

`ModelManifest` là authoritative source cho:

```text
model identity
engine
checkpoint
backend compatibility
capabilities
parameters
runtime requirements
hardware profiles
validation state
aliases
```

Không tiếp tục duy trì cùng một thông tin tại:

```text
tasks.py
worker.py
service config
manifest
workflow registry
README
MODEL_ENGINE_MAP
benchmark scripts
Studio
```

### 1.5 Khác weights → khác model ID

Ví dụ:

```text
qwen-image-2.1
qwen-image-2.1-turbo
```

Không còn:

```text
qwen-image-2.1 + variant=turbo
```

nếu Turbo thực sự sử dụng checkpoint khác.

Inference configuration trên cùng weights dùng parameter:

```text
steps
quality
reasoning_effort
speed
seed
cfg
resolution
```

### 1.6 Không automatic model fallback

Request:

```text
model=A
```

chỉ có hai kết quả:

```text
A chạy thành công
```

hoặc:

```text
A trả lỗi
```

Không được tự đổi sang model B.

Không có cả `fallback=true` trong production API.

### 1.7 Production fail-closed

Tuyệt đối cấm:

```text
synthetic image
fake WAV
sine-wave TTS
fabricated embedding
fake telemetry
mock transcript
fallback pretrained model khác
```

trong production inference path.

Backend unavailable:

```text
503
```

Model không active:

```text
model_not_found
```

Capability không hỗ trợ:

```text
unsupported_capability
```

---

## 2. Kiến trúc Engine đích

### 2.1 Responses Engine

Bao gồm:

```text
Text
VLM
multimodal text/image/video understanding
structured output
tool calling
streaming
reasoning parameters
```

Public API chính:

```text
POST /v1/responses
```

Compatibility:

```text
POST /v1/chat/completions
```

Engine không quan tâm model chạy:

```text
vLLM
llama.cpp
khác
```

Adapter quyết định.

### 2.2 Decision Engine

Capabilities:

```text
predicate_evaluation
choice_routing
score_ranking
```

Custom endpoint:

```text
POST /v1/decisions
```

Request/response/error conventions vẫn phải giống phong cách OpenAI.

### 2.3 Vector Engine

Bao gồm:

```text
Embedding
Rerank
```

Public:

```text
POST /v1/embeddings
POST /v1/rerank
```

Không cần tách thành hai service runtime nếu cùng một deployment có thể quản lý cả hai.

Model manifest quyết định model nào hỗ trợ embedding/rerank.

### 2.4 Speech Engine

Bao gồm:

```text
ASR
TTS
Voice Cloning
Alignment
Streaming ASR
```

Public OpenAI-compatible:

```text
POST /v1/audio/transcriptions
POST /v1/audio/speech
```

Factory extensions:

```text
POST /v1/audio/alignments
```

Voice cloning không trở thành endpoint/model riêng.

Ví dụ:

```text
/v1/audio/speech
+ reference_audio
+ reference_text
→ voice cloning capability
```

Nếu model không support cloning:

```text
unsupported_capability
```

### 2.5 Image Engine

Capability vocabulary:

```text
text_to_image
image_to_image
inpaint
outpaint
instruction_edit
reference_image
multi_reference
control_pose
control_depth
control_canny
```

Public:

```text
POST /v1/images/generations
POST /v1/images/edits
```

OpenAI-compatible fields được ưu tiên.

Factory-specific capability được suy luận từ inputs.

Ví dụ:

```text
prompt only
→ text_to_image

image + prompt
→ image_to_image / instruction_edit

image + mask
→ inpaint

reference_images
→ reference conditioning

control_image + control_type=pose
→ control_pose
```

Internal request luôn được normalize thành canonical capability rõ ràng.

### 2.6 Video Engine

Capability vocabulary:

```text
text_to_video
image_to_video
first_last_frame_to_video
reference_to_video
video_extension
video_edit
video_with_audio
```

Public ưu tiên:

```text
POST /v1/videos
POST /v1/videos/edits
POST /v1/videos/extensions
GET  /v1/videos/{id}
GET  /v1/videos/{id}/content
```

Factory mở rộng input schema cho:

```text
first_frame
last_frame
reference_images
enable_audio
```

Internal capability resolver chuyển sang:

```text
T2V / I2V / FL2V / Ref2V / Extension / ...
```

### 2.7 Music Engine

OpenAI chưa cung cấp một music generation API tương đương đầy đủ, do đó dùng Factory extension.

Public:

```text
POST /v1/music
GET  /v1/music/{id}
GET  /v1/music/{id}/content
```

Capability vocabulary:

```text
create
lyrics_to_music
cover
remake
repaint
reference_audio
extend
stem_extract
stem_add
score_generate
score_edit
song_variation
```

ACE-Step/YuE2 không bị ép phải support toàn bộ.

Engine hiểu vocabulary toàn bộ.

Adapter khai báo subset model support.

### 2.8 Enhancement Engine

Capabilities:

```text
image_super_resolution
video_super_resolution
frame_interpolation
audio_enhancement
```

Public Factory extension:

```text
POST /v1/enhancements
GET  /v1/enhancements/{id}
```

Enhancement tiếp tục được phát triển thành production Engine, nhưng chỉ trở thành READY sau khi backend inference thật hoàn chỉnh.

---

## 3. Model Catalog đích

Factory tiếp tục support các family hiện tại:

| Engine | Model target |
|---|---|
| Responses/Text | Qwen Text |
| Responses/VLM | Qwen VLM |
| Decision | Qwen Decision |
| Vector | Qwen Embedding + Reranker |
| Speech | Qwen3 ASR + Qwen3 TTS |
| Image | Qwen Image 2.1 |
| Image | Qwen Image 2.1 Turbo |
| Video | Wan 2.2 |
| Video | MiniMax H3 |
| Music | ACE-Step 1.5 |
| Music | YuE2 |
| Enhancement | Enhancement Suite |

Exact checkpoint/revision/backend phải được freeze lại trong Phase 1.

Không được giữ trạng thái hiện tại nơi:

```text
README nói model A
service config nói model B
manifest nói model C
benchmark chạy model D
```

---

## 4. Model Catalog vs Active Deployment

Hai khái niệm phải hoàn toàn tách biệt.

### 4.1 Model Catalog

Là toàn bộ model Factory biết cách chạy.

Ví dụ:

```text
qwen-text
qwen-vlm
qwen-decision
qwen-embedding
qwen-speech
qwen-image-2.1
qwen-image-2.1-turbo
wan-2.2
minimax-h3
ace-step-1.5
yue2
enhancement-suite-v1
```

### 4.2 Active Deployment Registry

Chỉ chứa model thực sự được cấu hình và load trong Kaggle session hiện tại.

Ví dụ:

```yaml
hardware: kaggle-dual-t4

deployments:
  - model: qwen-image-2.1-turbo
    devices: [0]

  - model: minimax-h3
    devices: [1]
```

Hoặc:

```yaml
deployments:
  - model: qwen-vlm
    devices: [0, 1]
    exclusive: true
```

### 4.3 `/v1/models`

Chỉ trả Active Deployment.

Không trả toàn Model Catalog.

Điều kiện model xuất hiện:

```text
configured
+ checkpoint present
+ backend launched
+ health check PASS
+ production validation allowed
```

Model Catalog được xem qua CLI/admin API riêng.

---

## 5. Kaggle Dual-T4 deployment model

### 5.1 Không dynamic GPU routing

Loại bỏ:

```text
Image → GPU0
Video → GPU1
Text → GPU0
Music → GPU1
```

khỏi core scheduler.

Không còn `SUBSYSTEM_GPU_MAP` làm source of truth.

### 5.2 GPU assignment do CLI quyết định trước

Ví dụ:

```bash
factory deploy add qwen-text --gpu 0
factory deploy add qwen-embedding --gpu 0
factory deploy add ace-step-1.5 --gpu 1
```

Dual:

```bash
factory deploy add qwen-vlm --gpu 0,1 --exclusive
```

### 5.3 Một GPU có thể chứa nhiều model

Cho phép:

```text
GPU0:
  model A
  model B
```

nếu deployment preflight xác nhận khả thi.

Không làm memory packing scheduler phức tạp.

Không tự swap model giữa các request trong Phase hiện tại.

### 5.4 Dual-GPU model độc quyền

Nếu deployment khai báo:

```yaml
devices: [0, 1]
exclusive: true
```

Factory không chạy job GPU khác đồng thời.

Qwen VLM lớn / Qwen Text lớn nếu cần Dual-T4 sẽ sử dụng cơ chế này.

---

## 6. Factory CLI

CLI trở thành control surface chính của deployment.

Target commands:

```text
factory models list
factory models inspect <model>

factory deployment init
factory deployment show
factory deployment add <model> --gpu ...
factory deployment remove <model>
factory deployment validate

factory weights verify
factory backend check
factory serve
factory status

factory validate model <model>
factory validate deployment
factory validate all
```

Không cần CLI quá nhiều framework abstraction.

### 6.1 `factory models list`

Hiển thị:

```text
model
engine
checkpoint
capabilities
backend
required VRAM
validation status
```

### 6.2 `factory deployment validate`

Kiểm tra trước khi serve:

```text
model exists
checkpoint exists
checkpoint integrity
required backend installed
GPU IDs valid
VRAM requirements plausible
port conflicts
exclusive GPU conflicts
adapter available
capability definitions valid
```

Không cần bảo đảm tuyệt đối không OOM chỉ từ metadata.

Sau backend load thật phải có second-stage runtime validation.

### 6.3 `factory serve`

Flow:

```text
load deployment config
→ verify environment
→ start required backends
→ wait health
→ validate models
→ build Active Deployment Registry
→ start API
```

Model backend fail:

```text
không đưa model vào active registry
```

Không được pretend READY.

---

## 7. Manifest redesign

### 7.1 ModelManifest mới

Target conceptual schema:

```yaml
id: qwen-image-2.1-turbo

aliases:
  - qwen-image-turbo

engine: image

checkpoint:
  repository: ...
  revision: ...
  files:
    - ...
  checksum: ...

backend:
  adapter: qwen_image
  supported:
    - comfyui

capabilities:
  text_to_image:
    supported: true
  image_to_image:
    supported: true
  inpaint:
    supported: false

parameters:
  steps:
    min: 1
    max: 25
  width:
    ...
  height:
    ...

runtime_profiles:
  kaggle-dual-t4:
    gpu_count: 1
    min_vram_gb: ...
    preferred_backend: comfyui

validation:
  state: experimental
```

### 7.2 Loại bỏ `task_variant_matrix`

Nếu variant = checkpoint khác:

```text
model khác
```

Nếu variant = inference parameter:

```text
parameter
```

Do đó `task_variant_matrix` hiện tại không còn cần thiết ở dạng hiện nay.

### 7.3 Capability registry

Capability definition chỉ có một canonical registry.

Ví dụ:

```text
image.text_to_image
image.image_to_image
video.text_to_video
speech.tts
music.cover
```

Aliases chỉ dùng tại Request Normalizer.

Internal code luôn dùng canonical ID.

---

## 8. Adapter architecture

Interface chung khái niệm:

```python
class ModelAdapter:
    def validate_request(...)
    def resolve_capability(...)
    def prepare(...)
    def infer(...)
    def normalize_output(...)
    def health(...)
```

Không ép tất cả backend phải implementation giống hệt.

### 8.1 Responses adapters

Ví dụ:

```text
VLLMAdapter
LlamaCppAdapter
```

Một model manifest chọn backend-compatible adapters.

### 8.2 Image adapter

Ví dụ:

```text
QwenImageAdapter
```

Adapter quản lý:

```text
capability → workflow
parameter mapping
asset staging
artifact extraction
```

Image Engine không biết node IDs ComfyUI.

### 8.3 Video adapters

```text
WanAdapter
MiniMaxAdapter
```

VideoEngine không được chứa:

```python
if "wan"...
if "minimax"...
```

### 8.4 Music adapters

```text
ACEStepAdapter
YuEAdapter
```

Giữ hướng adapter hiện tại nhưng chuẩn hóa interface.

### 8.5 Speech adapters

Tách:

```text
QwenASRAdapter
QwenTTSAdapter
```

cùng nằm trong SpeechEngine.

Voice clone là capability của QwenTTSAdapter.

---

## 9. Request normalization

Public request không bắt buộc gửi `task`.

Ví dụ Image:

```text
prompt only
→ text_to_image

image
→ image_to_image

image + mask
→ inpaint

instruction + image
→ instruction_edit

reference_images
→ reference_image / multi_reference
```

Video:

```text
prompt
→ T2V

image + prompt
→ I2V

first_frame + last_frame
→ FL2V

source_video
→ edit/extension
```

Speech:

```text
input text
→ TTS

input text + reference_audio
→ voice cloning
```

Nếu ambiguity:

```text
400 invalid_request
```

Không đoán nguy hiểm.

---

## 10. Error model

Chuẩn hóa error envelope theo style OpenAI.

Canonical errors:

```text
invalid_request_error
model_not_found
unsupported_capability
model_unavailable
backend_unavailable
hardware_unavailable
checkpoint_missing
checkpoint_invalid
resource_exhausted
timeout
internal_error
```

Các trường hợp:

### Model Factory không biết

```text
404 model_not_found
```

### Model có trong Catalog nhưng không active session

```text
404 model_not_found
```

Message:

```text
The requested model is not available in the current deployment.
```

### Capability không support

```text
400 unsupported_capability
```

### Backend chết

```text
503 model_unavailable
```

### OOM

```text
503 resource_exhausted
```

Không retry OOM tự động.

---

## 11. Sync vs Async

### Sync

```text
Responses
Chat Completions
Decision
Embedding
Rerank
ASR
short TTS
```

Support streaming nếu backend có.

### Async

```text
Image
Video
Music
long Enhancement
```

Không expose raw FactoryJob nếu API tương ứng có object lifecycle riêng.

Internal vẫn có Job abstraction.

---

## 12. Queue redesign

Queue chỉ phục vụ workload cần async/serialization.

Core:

```text
simple local queue
```

Không cần Redis abstraction trong target core.

Một GPU:

```text
max concurrent inference
```

được quy định bởi deployment/backend config.

Không cần distributed lease architecture.

Worker chỉ chạy trên host hiện tại.

---

## 13. Storage redesign

Default:

```text
local scratch
```

Output flow:

```text
backend output
→ verify
→ artifact metadata
→ client download
→ cleanup policy
```

S3/MinIO:

```text
optional extension
```

không thuộc core path.

---

## 14. Persistence

Mục tiêu đơn giản:

```text
in-memory
```

cho ephemeral session state.

Nếu cần lưu metadata/history:

```text
SQLite
```

Không cần PostgreSQL core.

Alembic/Postgres-specific infrastructure được loại khỏi core.

---

## 15. Authentication

Core:

```text
Bearer API Key
```

Một hoặc vài API key là đủ cho 2–3 ứng dụng.

Không cần:

```text
RBAC
organization
project
multi-tenant authorization
JWT audience matrix
```

ở core.

---

## 16. Telemetry

Giữ:

```text
request ID
model
capability
backend
GPU
queue latency
load latency
inference latency
total latency
peak VRAM
GPU utilization
output size
error code
```

Không cần full distributed OpenTelemetry architecture trong core.

Structured JSON logging là đủ.

---

## 17. Retry

Retry chỉ khi:

```text
temporary connection reset
temporary backend unavailable
transient filesystem issue
```

Không retry:

```text
OOM
unsupported capability
invalid input
missing weights
invalid config
checkpoint error
deterministic inference failure
```

---

## 18. Validation hierarchy mới

Dùng 4 cấp rõ ràng.

### V1 — Contract Test

Không GPU.

Kiểm tra:

```text
schemas
request normalization
error contracts
manifest validation
capability resolution
deployment validation
API compatibility
```

PASS không có nghĩa model chạy.

### V2 — Engine Integration

Có engine/backend boundary nhưng có thể dùng test fixture.

Kiểm tra:

```text
API → Engine
Engine → Adapter
Adapter parameter mapping
asset staging
streaming
job lifecycle
cancellation
artifact handling
error propagation
```

Không dùng V2 để chứng nhận model inference.

### V3 — Native Model Inference

Bắt buộc:

```text
exact target checkpoint
real GPU
real backend
real input
real inference
real output
real telemetry
fallback=false
```

Mỗi:

```text
model × capability
```

phải có ít nhất **1 golden case**.

Performance:

```text
1 warm-up
+ 3 measured runs
```

### V4 — Full Factory E2E

Flow đầy đủ:

```text
official SDK / HTTP
→ public API
→ normalization
→ deployment registry
→ Engine
→ Adapter
→ native backend
→ GPU
→ output
→ semantic validator
→ response/artifact
```

Đây là certification cuối cùng.

---

## 19. Semantic validation

Artifact tồn tại không đủ.

### Text

Kiểm tra:

```text
non-empty
prompt relevance
structured output validity nếu yêu cầu
tool call schema nếu dùng tools
```

### Decision

Golden cases phải có expected answer.

Ví dụ:

```text
predicate → expected boolean
choice → expected option
ranking → expected ordering
```

### Embedding

Dataset nhỏ:

```text
query
positive
negative
```

Requirement:

```text
similarity(query, positive)
>
similarity(query, negative)
```

Kiểm tra:

```text
dimension
finite values
normalization nếu expected
```

### Reranker

Expected relevant document phải rank cao hơn distractors.

### VLM

Golden input:

```text
real image
real question
expected facts
```

Grounding nếu support:

```text
bounding box IoU
```

Document:

```text
known answer
```

### ASR

Real audio + transcript.

Metric:

```text
WER
CER
```

### TTS

Kiểm tra:

```text
valid waveform
duration
non-silence
ASR generated speech
transcribed text similarity
```

### Voice Cloning

Ngoài intelligibility:

```text
speaker embedding similarity
```

giữa reference và output.

### Image T2I

Kiểm tra:

```text
real valid image
dimensions
prompt adherence
```

Có thể dùng CLIP/VLM judge.

### I2I/Edit

Kiểm tra đồng thời:

```text
input preservation
requested modification
```

### Inpaint

Difference phải tập trung ở mask region.

### Reference

Output phải thể hiện reference adherence.

### Control

Pose/depth/canny condition phải có measurable adherence.

### Video

Kiểm tra:

```text
container
frame count
fps
duration
non-static motion
conditioning fidelity
```

### FL2V

Kiểm tra:

```text
first-frame similarity
last-frame similarity
motion between them
```

### Video with Audio

Bắt buộc xác nhận audio stream thật:

```text
audio track exists
non-silent
duration compatible
```

Không suy luận chỉ từ MP4 container.

### Music

Kiểm tra:

```text
valid audio
non-silent
duration
prompt/style adherence
lyrics adherence nếu có
```

Cover:

```text
source/reference similarity
+
meaningful transformation
```

### Enhancement

Image:

```text
PSNR
SSIM
LPIPS
```

nếu có ground truth.

Interpolation:

```text
frame count
fps
temporal consistency
```

Audio:

```text
STOI / PESQ / SI-SDR
```

khi dataset phù hợp.

---

## 20. Production capability rule

Một model chỉ expose capability nếu:

```text
implemented = true
V3 = PASS
V4 = PASS
semantic_validation = PASS
```

Nếu chưa:

```text
không expose
```

Không có kiểu:

```text
implemented=true nhưng chưa test
→ vẫn trả trong /v1/models
```

---

## 21. `/v1/models` capability metadata

OpenAI compatibility được giữ cho fields chuẩn.

Factory-specific metadata không nên phá client.

Có thể cung cấp thêm:

```text
GET /v1/models/{id}/capabilities
```

hoặc:

```text
GET /factory/models/{id}
```

cho Studio/admin.

Ứng dụng thông thường chỉ cần `/v1/models`.

---

## 22. Model validation state

Internal:

```text
READY
EXPERIMENTAL
BLOCKED
DISABLED
DEPRECATED
```

Ý nghĩa:

- **READY**: Production V4 PASS.
- **EXPERIMENTAL**: Implemented nhưng chưa full certification.
- **BLOCKED**: Không thể chạy vì hardware/dependency/checkpoint.
- **DISABLED**: Không được deploy.
- **DEPRECATED**: Đang chuẩn bị removal.

Public `/v1/models` chỉ hiển thị:

```text
active + READY
```

---

## 23. Studio

Không refactor Studio cùng lúc với backend.

Thứ tự:

```text
Backend/API stable
→ OpenAPI/schema stable
→ Studio update
```

Studio controls được generate từ capability metadata.

Ví dụ chọn:

```text
qwen-image-2.1-turbo
```

Studio chỉ hiển thị những input thực sự support.

Không hard-code:

```javascript
if model === ...
```

---

## 24. Repo target structure

Target gần như:

```text
factory/
  api/
    responses/
    images/
    audio/
    videos/
    models/
    custom/

  core/
    manifests/
    catalog/
    deployment/
    capabilities/
    errors/
    telemetry/

  engines/
    responses/
    decision/
    vector/
    speech/
    image/
    video/
    music/
    enhancement/

  adapters/
    qwen_text/
    qwen_vlm/
    qwen_decision/
    qwen_embedding/
    qwen_asr/
    qwen_tts/
    qwen_image/
    wan/
    minimax/
    acestep/
    yue2/
    enhancement/

  runtime/
    backends/
    queue/
    artifacts/

  cli/

manifests/
  models/
  hardware/

deployments/
  kaggle/

tests/
  contract/
  integration/
  native/
  e2e/
  fixtures/

studio/

docs/
```

Không bắt buộc đổi toàn bộ filesystem trong một commit.

Đích kiến trúc quan trọng hơn tên folder cụ thể.

---

## 25. Docs target

Chỉ giữ khoảng 5–7 canonical docs.

```text
README.md
ARCHITECTURE.md
API.md
DEPLOYMENT.md
VALIDATION.md
CONTRIBUTING.md
```

Optional:

```text
MODELS.md
```

nhưng nếu model table generate được thì không maintain thủ công.

Xóa khỏi main tree:

```text
old plans
phase reports
fix reports
duplicate audit docs
old benchmark summaries
temporary migration docs
obsolete engine maps
```

Git history đã lưu lịch sử.

---

## 26. Benchmark output cleanup

Source repo chỉ giữ:

```text
benchmarks/
  cases/
  validators/
  runner.py
```

Generated output:

```text
validation/latest.json
validation/matrix.md
```

Historical outputs không commit hàng loạt.

Có thể lưu ngoài repo hoặc GitHub artifacts/releases.

---

# 27. PHASE 0 — Freeze & Baseline

## Mục tiêu

Ngăn repo tiếp tục drift trong lúc refactor.

## Công việc

Freeze:

```text
current main commit
current model catalog
current capability matrix
current API endpoints
current tests
```

Tạo baseline report:

```text
current model
claimed capabilities
actual implementation
actual validation
backend
checkpoint
GPU requirement
```

Đặc biệt downgrade mọi certification có evidence fallback/synthetic.

Qwen Speech phải được đánh dấu lại nếu evidence hiện tại không chứng minh inference thật.

## Output

```text
baseline.json
baseline_matrix.md
```

## Exit Gate

Mọi model/capability có một trạng thái truthful.

Không chỉnh architecture trước khi truth baseline sạch.

---

# 28. PHASE 1 — Canonical Model Truth

## Mục tiêu

Giải quyết source-of-truth drift.

## Công việc

Refactor ModelManifest.

Freeze chính xác cho từng model:

```text
canonical ID
checkpoint repository
revision
weight file
backend
engine
capabilities
VRAM requirement
aliases
```

Loại:

```text
task_variant_matrix
```

nơi nó đang dùng để biểu diễn checkpoint khác.

Resolve các mismatch hiện tại như:

```text
Qwen2.5 vs Qwen3.8
vLLM vs llama-server
standard vs turbo
duplicate Qwen Image Turbo manifest/variant
```

## Exit Gate

Một command:

```bash
factory models inspect MODEL
```

phải cung cấp toàn bộ truth không cần đọc source code.

---

# 29. PHASE 2 — Capability Canonicalization

## Mục tiêu

Một vocabulary duy nhất.

## Công việc

Định nghĩa canonical capability registry.

Xóa duplicate aliases/maps dần khỏi:

```text
leasing.py
workers
workflow registry
service configs
```

Request aliases chỉ resolve tại boundary.

Example:

```text
t2i
image.t2i
text2image
```

đều normalize một lần:

```text
image.text_to_image
```

sau đó internal không còn alias.

## Exit Gate

Search repo không còn nhiều module maintain capability mapping riêng.

---

# 30. PHASE 3 — Deployment Registry + CLI

## Mục tiêu

Thay dynamic routing bằng explicit deployment configuration.

## Công việc

Implement:

```text
ModelCatalog
DeploymentConfig
DeploymentRegistry
DeploymentValidator
```

CLI:

```text
models list
models inspect
deployment init
deployment add
deployment remove
deployment validate
deployment show
serve
status
```

Loại `SUBSYSTEM_GPU_MAP` khỏi production scheduling logic.

## Exit Gate

Có thể cấu hình:

```text
GPU0 → model A + B
GPU1 → model C
```

và:

```text
GPU0+GPU1 → model D exclusive
```

mà không sửa Python.

---

# 31. PHASE 4 — Backend Lifecycle

## Mục tiêu

Deployment config có thể launch backend thật.

## Công việc

Chuẩn hóa Backend interface:

```text
prepare
launch
health
stop
runtime_info
```

Adapters/backends dự kiến:

```text
vLLM
llama.cpp
ComfyUI
vLLM-Omni
ACE-Step native
YuE runtime
custom enhancement runtime
```

Không auto fallback backend nếu backend được chọn fail.

Một model có thể khai báo nhiều backend-compatible implementations, nhưng deployment chọn một implementation cụ thể trước khi serve.

## Exit Gate

`factory serve` launch đúng backends từ deployment config và chỉ active model healthy.

---

# 32. PHASE 5 — Engine Refactor

Thực hiện theo thứ tự rủi ro thấp → cao.

## 5.1 Vector

Refactor Embedding/Rerank thành VectorEngine + adapters.

## 5.2 Decision

Remove backend-specific logic khỏi DecisionEngine.

## 5.3 Responses

Hợp nhất public semantics Text/VLM.

Không nhất thiết merge process runtime; chỉ merge Engine/API contract.

## 5.4 Speech

ASR/TTS adapters.

Voice clone capability.

## 5.5 Image

Chuyển ComfyUI model-specific workflow logic sang QwenImageAdapter.

## 5.6 Video

Chuyển:

```python
if wan
if minimax
```

sang adapters.

## 5.7 Music

Chuẩn hóa ACE/YuE adapters cùng interface.

## 5.8 Enhancement

Hoàn chỉnh actual backend.

## Exit Gate

Thêm model mới vào một Engine không cần sửa Engine core.

---

# 33. PHASE 6 — OpenAI-Compatible API Layer

## Mục tiêu

Application integration trở thành first-class.

## Implement

```text
/v1/responses
/v1/chat/completions

/v1/embeddings

/v1/audio/transcriptions
/v1/audio/speech

/v1/images/generations
/v1/images/edits

/v1/videos
/v1/videos/{id}
/v1/videos/{id}/content
/v1/videos/edits
/v1/videos/extensions

/v1/models
```

Custom:

```text
/v1/decisions
/v1/rerank
/v1/music
/v1/enhancements
```

Normalize errors/output objects.

Không expose `service`.

Public user chỉ chọn `model`.

## Exit Gate

Official OpenAI Python SDK chạy được trên các endpoint compatible chỉ bằng đổi `base_url`.

---

# 34. PHASE 7 — Simplify Control Plane

Sau khi API/Engine ổn định mới cleanup.

## Remove/Reduce

```text
distributed worker assumptions
Redis requirement
Postgres requirement
complex lease ownership
enterprise RBAC
duplicated gateways
multiple queue repositories
unnecessary database abstractions
```

Giữ:

```text
simple async jobs
GPU mutual exclusion
heartbeat nếu thực sự cần long-running workers
cancellation
progress
artifact lifecycle
```

## Exit Gate

Core inference không yêu cầu Redis/Postgres/S3.

---

# 35. PHASE 8 — Native Certification V3

Test theo chi phí tăng dần.

## Wave A — Cheap

```text
Decision
Embedding
Rerank
Text
```

Mục tiêu: phát hiện sai architecture/backend nhanh.

## Wave B — VLM + Speech

```text
VLM text-only
VLM image
ASR
TTS
Voice Clone
Alignment
```

## Wave C — Image

```text
Standard T2I
Standard I2I
Inpaint
Outpaint
Edit
Reference
Multi-ref
Pose
Depth
Canny

Turbo supported capabilities
```

Turbo chỉ test những capability manifest cho phép.

## Wave D — Video

Wan:

```text
T2V
I2V
FL2V
Extension
```

MiniMax:

```text
T2V
I2V
FL2V
Reference
Video with Audio
```

Exact subset được manifest quyết định.

## Wave E — Music

ACE-Step:

```text
Create
Lyrics
Cover
Repaint
Reference
Extract
Lego
Complete
Extend
```

YuE:

```text
Create
Lyrics
Symbolic Score
Custom Score
Score Edit
Song Variation
```

## Wave F — Enhancement

```text
Image SR
Video SR
Interpolation
Audio Enhancement
```

---

# 36. PHASE 9 — Full Factory E2E V4

Mỗi production `model × capability` phải đi qua:

```text
client
→ API
→ resolver
→ deployment registry
→ engine
→ adapter
→ native backend
→ GPU
→ semantic validator
→ output
```

Không trực tiếp gọi backend để claim V4.

---

# 37. Official SDK acceptance suite

Python SDK bắt buộc.

Test:

```text
OpenAI client
base_url=Factory
```

Responses:

```python
client.responses.create(...)
```

Embeddings:

```python
client.embeddings.create(...)
```

Audio/Image tương ứng nếu SDK surface hỗ trợ.

Node SDK thêm compatibility coverage sau Python PASS.

Nếu một endpoint custom:

```text
Decision
Rerank
Music
Enhancement
```

dùng Factory SDK hoặc raw HTTP tests.

---

# 38. Benchmark evidence schema

Mỗi V3/V4 run tạo một evidence record:

```json
{
  "run_id": "...",
  "commit": "...",
  "hardware": "kaggle-dual-t4",
  "model": "...",
  "checkpoint": "...",
  "revision": "...",
  "backend": "...",
  "capability": "...",
  "input_sha256": "...",
  "output_sha256": "...",
  "fallback": false,
  "weights_verified": true,
  "latency_ms": {},
  "peak_vram_mb": 0,
  "semantic_metrics": {},
  "status": "PASS"
}
```

Không report Markdown thủ công làm source of truth.

Markdown matrix generate từ JSON.

---

# 39. Performance measurement

Mỗi capability:

```text
warm-up: 1
measured: 3
```

Ghi:

```text
median
min
max
```

Không lấy một run tốt nhất.

Metrics tùy modality:

```text
tokens/s
TTFT
images/min
seconds/image
fps
seconds/video
real-time factor
```

---

# 40. PHASE 10 — Studio Migration

Sau khi API stable.

Studio model selector đọc Active Deployment Registry.

Studio capability UI đọc Model Metadata.

Không hard-code model behaviors.

Ví dụ:

```text
Model selected
→ capabilities fetched
→ controls generated
```

Model không active không xuất hiện.

---

# 41. PHASE 11 — Repository Cleanup

Sau khi tests PASS.

Không cleanup lớn trước refactor vì sẽ khó phân biệt obsolete và required code.

## Remove

```text
old benchmark runners
synthetic certification runners
old phase scripts
duplicate live runners
obsolete compatibility adapters
deprecated service configs
duplicate manifest mappings
stale docs
historical reports
```

## Consolidate benchmark

Chỉ giữ một:

```text
CertificationRunner
```

với modes:

```text
contract
integration
native
e2e
```

---

# 42. Test suite cleanup

Hiện tests quá tập trung lifecycle.

Target phân bổ:

```text
contract tests
engine integration tests
adapter tests
native certification
full E2E
```

Không cần hàng trăm tests kiểm tra cùng queue state bằng nhiều cách.

Ưu tiên:

```text
meaningful coverage > raw test count
```

Không quảng bá:

```text
900 tests passed
```

như production evidence.

Quan trọng hơn:

```text
all production model-capability pairs certified
```

---

# 43. Những thành phần nên loại khỏi Core

```text
PostgreSQL
Redis
MinIO/S3
Alembic
JWT/RBAC
distributed worker orchestration
complex multi-tenant logic
automatic model routing
automatic fallback
multiple benchmark systems
historical report machinery
```

Có thể giữ extension nếu implementation nhỏ và không làm core phức tạp.

---

# 44. Những thành phần bắt buộc giữ

```text
OpenAI-compatible API
Model Catalog
Active Deployment Registry
CLI
Engine abstraction
Adapter abstraction
GPU assignment
simple queue
streaming
cancellation
artifact management
Bearer auth
structured errors
basic telemetry
Studio
real inference certification
```

---

# 45. Priority

Priority toàn Master Plan:

```text
1. Correctness
2. Real inference truth
3. Maintainability
4. Model extensibility
5. OpenAI compatibility
6. Simplicity
7. Performance optimization
```

Nguyên tắc:

> Correct → Generic → Compatible → Simple → Fast.

---

# 46. Definition of READY

Factory chỉ được gọi:

```text
READY
```

khi:

### Architecture

```text
single source of truth
deployment CLI working
no request-time GPU routing
model adapters isolated
```

### Public API

```text
OpenAI-compatible endpoints PASS
official SDK PASS
error schemas stable
```

### Models

Mọi model được expose:

```text
exact checkpoint verified
backend real
hardware real
```

### Capabilities

Mọi capability được expose:

```text
V3 PASS
V4 PASS
semantic validation PASS
```

### Safety against fake success

```text
fallback=false
synthetic=false
mock=false
```

### Runtime

```text
timeout works
cancellation works
OOM fails closed
backend crash returns 503
model unavailable returns correct error
```

### Artifacts

```text
valid output
correct MIME
real SHA256
download works
cleanup works
```

---

# 47. Definition of DONE cho một Model

Ví dụ `wan-2.2`.

DONE chỉ khi:

```text
manifest canonical
checkpoint pinned
adapter implemented
deployment profile defined
backend launches
health PASS

T2V V3 PASS
T2V V4 PASS

I2V V3 PASS
I2V V4 PASS

FL2V V3 PASS
FL2V V4 PASS

Extension V3 PASS
Extension V4 PASS

unsupported capabilities return explicit errors

OpenAI-compatible Video API PASS
```

Không cần support capability model thực tế không có.

---

# 48. Definition of DONE cho một Engine

Ví dụ VideoEngine.

```text
no model-specific branches
canonical capability resolver
adapter registry
input validation
output normalization
errors standardized
sync/async behavior stable
deployment registry integration
native adapters certified
```

Sau đó model video mới chỉ cần:

```text
manifest
adapter
workflow/assets
tests
```

---

# 49. Release Gate cuối cùng

Trước release tạo duy nhất:

```text
validation/latest.json
validation/matrix.md
```

Matrix phải có từng pair:

| Model | Capability | V3 | V4 | Semantic | Ready |
|---|---|---|---|---|---|

Không được có production capability ở trạng thái:

```text
UNKNOWN
UNTESTED
claimed
assumed
synthetic-pass
```

Chỉ:

```text
PASS
FAIL
BLOCKED
UNSUPPORTED
DISABLED
```

---

# 50. Thứ tự triển khai bắt buộc

Không đảo phase tùy tiện.

```text
0 Truth Baseline
      ↓
1 Model Truth
      ↓
2 Capabilities
      ↓
3 Deployment CLI
      ↓
4 Backend Lifecycle
      ↓
5 Engine/Adapters
      ↓
6 OpenAI API
      ↓
7 Control Plane Simplification
      ↓
8 Native V3
      ↓
9 Factory V4
      ↓
10 Studio
      ↓
11 Cleanup
      ↓
Release
```

Lý do:

Không nên cleanup trước khi biết phần nào còn dùng.

Không nên benchmark trước khi model identity sạch.

Không nên Studio refactor trước khi API stable.

Không nên E2E tất cả capability trước khi Engine generic.

Không nên tối ưu performance trước khi inference correctness PASS.

---

# 51. Kết quả kiến trúc mong đợi sau Master Plan

Trước:

```text
many models
many configs
many maps
many runners
many reports
many abstractions
few genuine verified inference paths
```

Sau:

```text
OpenAI-compatible API
        ↓
8 functional engines
        ↓
small set of clean adapters
        ↓
explicit Kaggle deployment
        ↓
real model inference
        ↓
semantic validation
```

Thêm model mới:

```text
manifest
+
adapter
+
tests
```

Thêm hardware mới sau này:

```text
new hardware profile
+
deployment config
```

không rewrite Engine.

---

# 52. Target cuối về độ phức tạp

Hiện trạng ước tính:

```text
Over Engineering ≈ 8/10
```

Target:

```text
≈ 4–5/10
```

Factory vẫn là production-quality inference system, nhưng complexity phải nằm ở nơi tạo giá trị:

```text
model integration
capability handling
real inference
GPU deployment
API compatibility
validation
```

thay vì:

```text
distributed infrastructure
enterprise persistence
duplicate documentation
duplicate routing
synthetic verification
framework abstraction không dùng
```

---

# MASTER SUCCESS CRITERION

Master Plan được coi hoàn tất khi một ứng dụng bất kỳ có thể:

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://factory.example.com/v1"
)
```

sau đó gọi model đang active trong Kaggle session bằng API tương thích OpenAI mà không cần biết Factory chạy bằng:

```text
Dual T4
single T4
ComfyUI
vLLM
llama.cpp
ACE-Step
worker
queue
```

và mọi model/capability mà Factory tuyên bố support đều đã được chứng minh bằng:

> **exact model + exact checkpoint + real input + real GPU inference + real meaningful output + semantic validation + full API E2E.**
