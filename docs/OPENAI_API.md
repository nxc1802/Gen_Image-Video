# Factory application API

This document describes the Phase 6 supported wire contract. [MASTER_PLAN.md](MASTER_PLAN.md) remains the architecture authority. The production registry currently authorizes **zero model/capability pairs**: configured/healthy backends still return `503 MODEL_NOT_LOADED` until exact-checkpoint Native V3, semantic validation and Full Factory E2E evidence authorize the selected capability. SDK software acceptance does not unlock a model.

Use the unmodified official Python SDK, tested at **openai 3.49.0**, with a Factory credential and local `base_url`:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FACTORY_API_KEY"],
    base_url="http://127.0.0.1:8000/v1",
)

models = client.models.list()
response = client.responses.create(
    model="qwen-text", input="Explain this function.", store=False
)
print(response.output_text)
```

The example requires a configured, evidence-authorized deployment; it is not a current readiness claim. Model IDs select an explicit deployment. Requests cannot choose a backend, GPU, checkpoint or variant, and there is no automatic fallback. Public inference requests do not have a `service` field. Catalog administration remains separate under `/v1/registry/*`.

## Supported contracts

| Route | Supported subset and limits |
|---|---|
| `POST /v1/responses` | Text or supported message/function-call input; nonstreaming and native Responses SSE; text/tool/structured features remain gated by the selected Manifest. `store=true`, previous-response/conversation persistence and undeclared features are rejected. Native response IDs, timestamps, output and usage are preserved. |
| `POST /v1/chat/completions` | Messages, supported tools/response format, sampling and output limits, SSE. `max_completion_tokens` maps explicitly to the selected backend's output-limit field; conflicting limits fail. Undeclared reasoning controls fail rather than manufacture reasoning. |
| `POST /v1/embeddings` | Text/batched text, Manifest-bounded dimensions, `float` or little-endian float32 `base64`. The SDK's default base64 request is decoded by the SDK normally. Token-array and undeclared multimodal inputs are rejected. |
| `POST /v1/audio/transcriptions` | Multipart `file`, explicit model, language/prompt, `json` or `text`; current ASR profile requires PCM16 WAV. Timestamps, streaming and nonzero temperature are unsupported. |
| `POST /v1/audio/speech` | Binary WAV; current Qwen TTS **Base** requires reference audio plus its transcript and `voice="default"`. No preset voice/VoiceDesign checkpoint substitution. Unsupported formats and speech SSE fail explicitly. |
| `POST /v1/images/generations` | Explicit model/prompt, supported dimensions, one PNG with `b64_json`; optional native steps/seed/cfg extensions. Undeclared quality modes, formats, batching or URL output are rejected. |
| `POST /v1/images/edits` | Multipart image or image array, prompt, optional mask; JSON inline/remote images are also normalized. Mask selects inpaint, multiple images select multi-reference, one image selects instruction edit. Explicit canonical edit capability can select I2I/outpaint/reference where the Manifest supports it. Outpaint requires an expanded source canvas plus mask. |
| `POST /v1/videos` | Historical SDK create contract, multipart/JSON, optional reference upload, explicit model, prompt/size/seconds. Factory extensions: first/last frame, references, enable_audio, num_frames, steps, seed, fps. Canonical input resolution does not choose another checkpoint. |
| `GET /v1/videos/{id}` | Actual Factory job states `queued`, `in_progress`, `completed`, `failed`. Progress is 0 until actual completion, then 100; no invented inference progress. Completed size/duration are probed from the actual MP4. |
| `GET /v1/videos/{id}/content` | Owned completed original MP4 only. Missing/expired content is 404; incomplete/failed jobs are 409; corrupted bytes fail integrity checks. Thumbnail/spritesheet variants are unsupported. |
| `POST /v1/videos/extensions` | Source upload with explicit model, or an owned completed Factory video ID; source-ID selection reuses that job's exact model. The current Wan adapter concatenates actual output with the source and fails if concatenation fails. |
| `POST /v1/videos/edits` | Route and SDK request are accepted for validation, but the current catalog declares **no video-edit capability**. Returns an explicit unsupported-capability error; never reinterpreted as I2V/extension. |
| `GET /v1/models` | Only evidence-authorized active-session models; currently an empty SDK-compatible list. |
| `POST /v1/decisions` | Current official OpenAI shape: model, text input, ordered questions and native answers/usage. Supported native subset: predicate, 2–26 unique string choices, 2–10 ordered score labels. Native score is an expected zero-based level index. Unsupported Boolean choices, multimedia/question parts and caching controls fail explicitly. |
| `POST /v1/rerank` | Factory extension: model/query/documents/optional top_n, original documents and native relevance scores. |
| `POST /v1/music` | Factory extension: model/canonical capability/inputs; asynchronous job. Status/content under `/v1/music/{id}` and `/content`. Only capabilities supported by the selected ACE/YuE adapter are accepted. YuE2 remains BLOCKED on the selected BF16/T4 runtime. |
| `POST /v1/enhancements` | Same explicit custom request form and asynchronous status/content routes. Enhancement remains DISABLED; the isolated RealESRGAN backend witness does not authorize this public route. Unpinned RIFE/audio/video-SR capabilities remain unsupported. |

Media is ingested before adapter dispatch. HTTP(S) inputs use the existing DNS-pinned SSRF validator, disallow redirects/environment proxies and enforce byte limits. Local filesystem paths and unresolved file IDs are not forwarded to a backend. Multipart/JSON public bodies are bounded to 64 MiB using actual received bytes. Inline bytes are container-validated; container validity is not semantic acceptance.

For video requests, `seconds` maps to native frame span using `num_frames = seconds * fps + 1`. The selected Adapter enforces its native frame grid and Manifest limits; Factory never silently rounds an invalid request. Native container duration may include the final sampled frame; completed job metadata reports the actual probed duration, including concatenated extension content.

## SDK examples for supported media

```python
import base64
from pathlib import Path

vectors = client.embeddings.create(
    model="qwen-embedding", input=["first document", "second document"]
)

with Path("speech.wav").open("rb") as audio:
    transcript = client.audio.transcriptions.create(model="qwen-asr", file=audio)

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
speech = client.audio.speech.create(
    model="qwen-tts", voice="default", input="Hello.", response_format="wav",
    extra_body={
        "reference_audio": "data:audio/wav;base64," + reference,
        "reference_text": "The actual words spoken in reference.wav.",
    },
)
speech.write_to_file("output.wav")

image = client.images.generate(
    model="qwen-image-2.1", prompt="A red bicycle beside a stone wall.",
    size="1024x1024", response_format="b64_json",
)
Path("output.png").write_bytes(base64.b64decode(image.data[0].b64_json))

with Path("source.png").open("rb") as source:
    edited = client.images.edit(
        model="qwen-image-2.1", image=source,
        prompt="Change the bicycle to blue.",
    )

video = client.videos.create(
    model="wan-2.2", prompt="A small boat moving across calm water.",
    seconds="4", size="512x512",
)
# Retrieve until status is completed/failed before downloading content.
status = client.videos.retrieve(video.id)
# client.videos.download_content(video.id).write_to_file("output.mp4")
```

AsyncOpenAI uses the same `base_url` and supported schema. Chat and Responses streaming forward validated native records; preflight errors arrive before HTTP 200, later failures emit an error event without a fabricated success/completion marker. Native usage is never invented to satisfy an SDK object.

## Historical video SDK limitation

OpenAI's [deprecation notice](https://developers.openai.com/api/docs/deprecations) and [retained Videos reference](https://developers.openai.com/api/reference/resources/videos/methods/create) say the hosted Videos API shut down on 2026-09-24. Factory preserves the Master Plan's historical route family and tests the methods still present in SDK 3.49.0; this is not a claim that OpenAI still hosts video generation.

That SDK version's `videos.edit/extend` file extractor rejects a dictionary reference such as `video={"id": ...}` before sending HTTP. Upload a source clip through the unmodified named method instead:

```python
with Path("source.mp4").open("rb") as source:
    extension = client.videos.extend(
        video=source, prompt="Continue the camera movement.", seconds="4",
        extra_body={"model": "wan-2.2", "size": "512x512"},
    )
```

Factory also accepts `{"video":{"id":"video_..."},"prompt":"...","seconds":"4"}` over its JSON HTTP contract, resolving only an owned completed job. The client-side extraction limitation is recorded in [PLAN_FINDINGS.md](PLAN_FINDINGS.md); neither client patching nor another model is used to hide it.

## Jobs, errors and evidence

Public media jobs use a bounded in-process facade (32 outstanding, 256 retained metadata entries) over the canonical Engines and existing scratch storage. They are ephemeral: restart loses metadata, TTL expires bytes/status, and retrieval checks the authenticated owner and object family. Metadata and IDs describe Factory's actual lifecycle. Phase 7 removed the older distributed worker/job composition from default startup; it remains only in the explicit non-production migration host. Core job management and cancellation are documented in [CONTROL_PLANE.md](CONTROL_PLANE.md).

Errors include `error.message`, `error.type`, `error.param`, `error.code` and `X-Request-ID`; retained internal APIs also preserve legacy request/details fields. No host paths, backend tokens or substitute model outputs are returned. Unknown parameters are rejected rather than silently ignored. SDK auth, request-validation, rate-limit and server-error classes follow HTTP status.

[openai_api_gate.json](../validation/openai_api_gate.json) records official SDK version, source/test hashes, route inventory and executed software acceptance. Test authorization, adapter responses, transcripts and media fixtures are explicitly mocked under `contract_mock`. The localhost server and SDK HTTP serialization/decoding are real. These checks are **software compatibility only**, with no GPU/semantic/Full Factory E2E inference claim. [openai_sdk_sources.json](../validation/openai_sdk_sources.json) records the installed official SDK source identities.
