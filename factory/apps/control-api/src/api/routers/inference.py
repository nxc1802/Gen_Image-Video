"""OpenAI wire contracts and explicit Factory-only inference extensions."""

import asyncio
import base64
import json
import struct
import time
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from factory_auth import verify_bearer_token
from factory_contracts.errors import (
    InvalidRequestError,
    UnsupportedCapabilityError,
    ModelUnavailableError,
)
from factory.core.media import MediaRequest, check_media
from factory.engines.vector import VectorEngine
from factory.engines.vector.contracts import EmbeddingRequest, RerankRequest
from factory.engines.speech import SpeechEngine
from factory.engines.image import ImageEngine
from factory.engines.video import VideoEngine
from factory.engines.music import MusicEngine
from factory.engines.enhancement import EnhancementEngine
from factory.engines.decision import DecisionEngine
from factory.engines.decision.contracts import PublicDecisionRequest
from factory.apps.control_api.src.api.public import (
    payload,
    decode,
    selected,
    engine,
    admit,
    media,
    dimensions,
    only,
    number,
)
from factory.apps.control_api.src.api.public.jobs import owner

router = APIRouter(prefix="/v1", tags=["Inference"], dependencies=[Depends(verify_bearer_token)])


def dump(result):
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result


@router.post("/embeddings")
async def embeddings(request: Request):
    body = await payload(request, {"model", "input", "dimensions", "encoding_format", "user"})
    only(body, "encoding_format", {"float", "base64"}, "float")
    selected(request, body.get("model"))
    call = decode(
        EmbeddingRequest, {k: v for k, v in body.items() if k not in {"encoding_format", "user"}}
    )
    result = dump(await asyncio.to_thread(engine(request, VectorEngine).generate_embeddings, call))
    if body.get("encoding_format", "float") == "base64":
        for item in result["data"]:
            item["embedding"] = base64.b64encode(
                struct.pack("<" + "f" * len(item["embedding"]), *item["embedding"])
            ).decode()
    return result


@router.post("/rerank")
async def rerank(request: Request):
    body = await payload(request, set(RerankRequest.model_fields))
    selected(request, body.get("model"))
    return dump(
        await asyncio.to_thread(
            engine(request, VectorEngine).rerank_documents, decode(RerankRequest, body)
        )
    )


@router.post("/decisions")
async def decisions(request: Request):
    body = await payload(request, set(PublicDecisionRequest.model_fields))
    selected(request, body.get("model"))
    return dump(
        await asyncio.to_thread(
            engine(request, DecisionEngine).create, decode(PublicDecisionRequest, body)
        )
    )


@router.post("/audio/transcriptions")
async def transcriptions(request: Request):
    body = await payload(
        request,
        {
            "model",
            "file",
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
            "stream",
        },
    )
    selected(request, body.get("model"))
    format = only(body, "response_format", {"json", "text"}, "json")
    only(body, "stream", {False, "false", None}, False)
    only(body, "temperature", {0, 0.0, "0", "0.0", None}, 0)
    if body.get("timestamp_granularities"):
        raise UnsupportedCapabilityError(
            "Selected ASR profile does not implement timestamp output."
        )
    if "file" not in body:
        raise InvalidRequestError("Audio file required.", details={"param": "file"})
    instance = engine(request, SpeechEngine)
    # Authorize before downloading or decoding externally supplied media.
    instance.registry.require_active(body["model"], "speech.asr")
    raw = await media(body["file"], "audio", request)
    import io, wave

    with wave.open(io.BytesIO(raw), "rb") as audio:
        if audio.getsampwidth() != 2:
            raise UnsupportedCapabilityError("Selected ASR profile requires PCM16 WAV input.")
    inputs = {"audio": raw, **{k: body[k] for k in ("language", "prompt") if k in body}}
    result = await asyncio.to_thread(
        instance.transcribe,
        MediaRequest(model=body["model"], capability="speech.asr", inputs=inputs),
    )
    return (
        Response(result["text"], media_type="text/plain")
        if format == "text"
        else {"text": result["text"]}
    )


@router.post("/audio/speech")
async def speech(request: Request):
    body = await payload(
        request,
        {
            "model",
            "input",
            "voice",
            "speed",
            "response_format",
            "reference_audio",
            "reference_text",
            "stream_format",
        },
    )
    selected(request, body.get("model"))
    only(body, "stream_format", {"audio", None}, "audio")
    format = only(body, "response_format", {"wav"}, "wav")
    if not isinstance(body.get("input"), str) or not body["input"].strip():
        raise InvalidRequestError("Nonempty input text required.", details={"param": "input"})
    instance = engine(request, SpeechEngine)
    instance.registry.require_active(body["model"], "speech.tts")
    inputs = {
        "text": body["input"],
        "voice": body.get("voice", "default"),
        "response_format": format,
    }
    for key in ("speed", "reference_text"):
        if key in body:
            inputs[key] = body[key]
    if "reference_audio" in body:
        inputs["reference_audio"] = await media(body["reference_audio"], "audio", request)
    result = await asyncio.to_thread(
        instance.synthesize,
        MediaRequest(model=body["model"], capability="speech.tts", inputs=inputs),
    )
    if (
        result.get("model") != instance.registry.assignment(body["model"]).model
        or len(result.get("artifacts", [])) != 1
    ):
        raise ModelUnavailableError("TTS returned no matching waveform.")
    item = result["artifacts"][0]
    await asyncio.to_thread(check_media, item["data"], "audio")
    return Response(item["data"], media_type="audio/wav")


IMAGE_FIELDS = {
    "model",
    "prompt",
    "size",
    "n",
    "response_format",
    "output_format",
    "quality",
    "steps",
    "seed",
    "cfg",
    "strength",
    "capability",
    "reference_images",
}


async def image_call(request, edit):
    body = await payload(
        request,
        IMAGE_FIELDS | ({"image", "mask"} if edit else set()),
        repeated={"image", "reference_images"},
    )
    _, binding, model = selected(request, body.get("model"))
    only(body, "n", {1, "1"}, 1)
    only(body, "response_format", {"b64_json"}, "b64_json")
    only(body, "output_format", {"png"}, "png")
    only(body, "quality", {"auto", None}, "auto")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidRequestError("Nonempty prompt required.", details={"param": "prompt"})
    if not edit and any(k in body for k in ("reference_images", "instruction", "capability")):
        raise InvalidRequestError("Reference/edit inputs belong on images/edits.")
    instance = engine(request, ImageEngine)
    inputs = {"prompt": prompt, **dimensions(body.get("size"), model)}
    for key in ("steps", "seed", "cfg", "strength"):
        if key in body:
            inputs[key] = number(body[key], key, integer=key not in {"cfg", "strength"})
    capability = "image.text_to_image"
    if not edit and "strength" in body:
        raise UnsupportedCapabilityError("strength applies only to image edits.")
    if edit:
        sources = body.get("image", [])
        sources = sources if isinstance(sources, list) else [sources]
        refs = body.get("reference_images", [])
        refs = refs if isinstance(refs, list) else [refs]
        if not sources and not refs:
            raise InvalidRequestError("At least one image required.", details={"param": "image"})
        if body.get("mask"):
            if len(sources) != 1 or refs:
                raise InvalidRequestError("Masked edits require exactly one source image.")
            capability = "image.inpaint"
        elif refs:
            capability = (
                "image.multi_reference" if len(refs) + len(sources) > 1 else "image.reference_image"
            )
        elif len(sources) > 1:
            capability = "image.multi_reference"
        else:
            capability = "image.instruction_edit"
        if body.get("capability"):
            if body["capability"] not in {
                "image.inpaint",
                "image.outpaint",
                "image.image_to_image",
                "image.instruction_edit",
                "image.reference_image",
                "image.multi_reference",
            }:
                raise UnsupportedCapabilityError("Unsupported explicit image edit capability.")
            capability = body["capability"]
        instance.registry.require_active(binding.model, capability)
        if refs or len(sources) > 1:
            inputs["reference_images"] = [
                await media(value, "image", request) for value in (sources + refs)
            ]
        else:
            inputs["image"] = await media(sources[0], "image", request)
        if body.get("mask"):
            inputs["mask"] = await media(body["mask"], "image", request)
    result = await asyncio.to_thread(
        instance.run, MediaRequest(model=binding.model, capability=capability, inputs=inputs)
    )
    if result.get("model") != binding.model or len(result.get("artifacts", [])) != 1:
        raise ModelUnavailableError("Image backend returned no matching output.")
    data = []
    for item in result["artifacts"]:
        if item["type"] != "image":
            raise ModelUnavailableError("Image backend returned another media type.")
        await asyncio.to_thread(check_media, item["data"], "image")
        data.append({"b64_json": base64.b64encode(item["data"]).decode()})
    # Factory generated the envelope at this time; native usage/prompt revisions
    # are omitted unless the backend actually supplies them.
    return {"created": int(time.time()), "data": data}


@router.post("/images/generations")
async def image_generations(request: Request):
    return await image_call(request, False)


@router.post("/images/edits")
async def image_edits(request: Request):
    return await image_call(request, True)


VIDEO_FIELDS = {
    "model",
    "prompt",
    "seconds",
    "size",
    "input_reference",
    "first_frame",
    "last_frame",
    "reference_images",
    "enable_audio",
    "num_frames",
    "steps",
    "seed",
    "fps",
}


async def create_video(request, operation):
    body = await payload(request, VIDEO_FIELDS | {"video"}, repeated={"reference_images"})
    jobs = request.app.state.public_media_jobs
    source = None
    if operation != "create":
        reference = body.get("video")
        if isinstance(reference, str) and reference.startswith("video_"):
            source = jobs.get(reference, owner(request), "video")
            body.setdefault("model", source["model"])
            body.setdefault("size", source["size"])
        elif isinstance(reference, dict):
            if set(reference) == {"id"}:
                source = jobs.get(reference["id"], owner(request), "video")
                body.setdefault("model", source["model"])
                body.setdefault("size", source["size"])
            else:
                raise UnsupportedCapabilityError(
                    "Video references accept an existing Factory video ID or uploaded bytes."
                )
    _, binding, model = selected(request, body.get("model"))
    instance = engine(request, VideoEngine)
    if operation == "edits":
        # No current Manifest declares video editing. Do not reinterpret it as
        # I2V/extension or select another model.
        raise UnsupportedCapabilityError("Selected catalog has no video edit capability.")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidRequestError("Nonempty prompt required.", details={"param": "prompt"})
    inputs = {"prompt": prompt, **dimensions(body.get("size"), model)}
    for field in ("steps", "seed", "fps", "num_frames"):
        if field in body:
            inputs[field] = number(body[field], field, integer=True)
    fps = inputs.get("fps", model.parameters["fps"]["default"])
    if fps <= 0:
        raise InvalidRequestError("fps must be positive.")
    if "seconds" in body:
        seconds = number(body["seconds"], "seconds")
        frames = seconds * fps + 1
        if not __import__("math").isfinite(frames) or seconds <= 0 or not frames.is_integer():
            raise InvalidRequestError(
                "Duration cannot be represented exactly by the selected frame rate."
            )
        if "num_frames" in inputs and inputs["num_frames"] != int(frames):
            raise InvalidRequestError("Conflicting seconds and num_frames.")
        inputs["num_frames"] = int(frames)
    else:
        seconds = (inputs.get("num_frames", model.parameters["num_frames"]["default"]) - 1) / fps
    first = body.get("first_frame", body.get("input_reference"))
    if body.get("first_frame") is not None and body.get("input_reference") is not None:
        raise InvalidRequestError("Duplicate initial frame.")
    audio = only(body, "enable_audio", {True, False, "true", "false"}, False) in (True, "true")
    refs = body.get("reference_images", [])
    capability = (
        "video.video_extension"
        if operation == "extensions"
        else "video.video_with_audio"
        if audio
        else "video.reference_conditioning"
        if refs
        else "video.first_last_frame_to_video"
        if body.get("last_frame")
        else "video.image_to_video"
        if first
        else "video.text_to_video"
    )
    if (refs and (first or body.get("last_frame"))) or operation == "create" and "video" in body:
        raise InvalidRequestError("Conflicting conditioning inputs.")
    if body.get("last_frame") and not first:
        raise InvalidRequestError("last_frame requires first_frame.")
    if operation == "extensions" and (first or body.get("last_frame") or refs or audio):
        raise InvalidRequestError("Extension does not accept image/audio conditioning.")
    instance.registry.require_active(binding.model, capability)
    if operation == "extensions":
        if source:
            if source["model"] != binding.model:
                raise InvalidRequestError(
                    "Extension must explicitly select the same configured model as its source."
                )
            raw, _ = await jobs.content(source["id"], owner(request), "video")
        else:
            raw = await media(body.get("video"), "video", request)
        inputs["video"] = raw
    else:
        if first:
            inputs["image"] = await media(first, "image", request)
        if body.get("last_frame"):
            inputs["last_image"] = await media(body["last_frame"], "image", request)
        if refs:
            inputs["reference_images"] = [await media(value, "image", request) for value in refs]
    call = MediaRequest(model=binding.model, capability=capability, inputs=inputs)
    return await jobs.submit(
        instance,
        call,
        owner(request),
        "video",
        {
            "size": str(inputs["width"]) + "x" + str(inputs["height"]),
            "seconds": str(seconds).removesuffix(".0"),
            "prompt": prompt,
            "remixed_from_video_id": source["id"] if source else None,
        },
    )


@router.post("/videos")
async def videos(request: Request):
    return await create_video(request, "create")


@router.post("/videos/edits")
async def video_edits(request: Request):
    return await create_video(request, "edits")


@router.post("/videos/extensions")
async def video_extensions(request: Request):
    return await create_video(request, "extensions")


@router.get("/videos/{identifier}")
async def video_status(identifier: str, request: Request):
    return request.app.state.public_media_jobs.view(
        request.app.state.public_media_jobs.get(identifier, owner(request), "video")
    )


@router.get("/videos/{identifier}/content")
async def video_content(identifier: str, request: Request, variant: str = "video"):
    if variant != "video":
        raise UnsupportedCapabilityError("Only original video content is implemented.")
    raw, mime = await request.app.state.public_media_jobs.content(
        identifier, owner(request), "video"
    )
    return Response(raw, media_type=mime)


async def custom_media(request, cls, kind):
    body = await payload(request, {"model", "capability", "inputs"})
    selected(request, body.get("model"))
    call = decode(MediaRequest, body)
    instance = engine(request, cls)
    instance.registry.require_active(call.model, call.capability)
    for field, value in list(call.inputs.items()):
        media_kind = {
            "image": "image",
            "audio": "audio",
            "source_audio": "audio",
            "reference_audio": "audio",
            "video": "video",
        }.get(field)
        if media_kind:
            call.inputs[field] = await media(value, media_kind, request)
    return await request.app.state.public_media_jobs.submit(
        instance, call, owner(request), kind, {}
    )


@router.post("/music")
async def music(request: Request):
    return await custom_media(request, MusicEngine, "music")


@router.get("/music/{identifier}")
async def music_status(identifier: str, request: Request):
    return request.app.state.public_media_jobs.view(
        request.app.state.public_media_jobs.get(identifier, owner(request), "music")
    )


@router.get("/music/{identifier}/content")
async def music_content(identifier: str, request: Request):
    raw, mime = await request.app.state.public_media_jobs.content(
        identifier, owner(request), "music"
    )
    return Response(raw, media_type=mime)


@router.post("/enhancements")
async def enhancements(request: Request):
    return await custom_media(request, EnhancementEngine, "enhancement")


@router.get("/enhancements/{identifier}")
async def enhancement_status(identifier: str, request: Request):
    return request.app.state.public_media_jobs.view(
        request.app.state.public_media_jobs.get(identifier, owner(request), "enhancement")
    )


@router.get("/enhancements/{identifier}/content")
async def enhancement_content(identifier: str, request: Request):
    raw, mime = await request.app.state.public_media_jobs.content(
        identifier, owner(request), "enhancement"
    )
    return Response(raw, media_type=mime)
