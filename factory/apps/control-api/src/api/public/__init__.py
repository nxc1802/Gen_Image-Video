"""Public API normalization; model/backend logic remains in selected Adapters."""

import base64
import json
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from factory_contracts.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedCapabilityError,
    ModelNotLoadedError,
)
from factory.core.media import check_media, data_bytes
from factory.core.model_adapter import ExecutionContext

MAX_BODY = 64 * 1024 * 1024


async def payload(request, allowed, *, repeated=()):
    # Bound the actual body, including multipart uploads, before parsing. The
    # cached body is consumed by Starlette's normal JSON/form parser afterwards.
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise PayloadTooLargeError("Request body exceeds 64 MiB.")
        chunks.append(chunk)
    request._body = b"".join(chunks)
    content = request.headers.get("content-type", "").split(";", 1)[0]
    if content == "application/json":
        try:
            body = await request.json()
        except (ValueError, UnicodeError) as exc:
            raise InvalidRequestError("Invalid JSON body.") from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("Expected a JSON object.")
    elif content in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        form = await request.form(max_files=20, max_fields=80, max_part_size=MAX_BODY)
        body = {}
        for key, value in form.multi_items():
            key = key[:-2] if key.endswith("[]") else key
            if "[" in key and key.endswith("]") and not key.endswith("[]"):
                parent, child = key[:-1].split("[", 1)
                if parent not in {"video", "input_reference"} or child not in {
                    "id",
                    "image_url",
                    "file_id",
                }:
                    raise InvalidRequestError("Unsupported nested parameter.")
                obj = body.setdefault(parent, {})
                if not isinstance(obj, dict) or child in obj:
                    raise InvalidRequestError("Conflicting nested parameter.")
                obj[child] = value
                continue
            if key in repeated:
                body.setdefault(key, []).append(value)
            elif key in body:
                raise InvalidRequestError("Duplicate parameter.", details={"param": key})
            else:
                body[key] = value
    else:
        raise InvalidRequestError("Expected application/json or multipart/form-data.")
    unknown = set(body) - set(allowed)
    if unknown:
        raise InvalidRequestError("Unsupported parameter.", details={"param": sorted(unknown)[0]})
    return body


def decode(cls, body):
    try:
        return cls.model_validate(body)
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        raise InvalidRequestError(
            "Invalid request parameter: " + ".".join(map(str, first["loc"])),
            details={"param": ".".join(map(str, first["loc"]))},
        ) from exc


def selected(request, model):
    if not isinstance(model, str) or not model.strip():
        raise InvalidRequestError("Explicit model is required.", details={"param": "model"})
    registry = getattr(request.app.state, "deployment_registry", None)
    if registry is None:
        raise ModelNotLoadedError("No configured deployment in this session.")
    binding = registry.assignment(model)
    return registry, binding, registry.catalog.get(binding.model)


def engine(request, cls):
    return cls(
        registry=request.app.state.deployment_registry,
        session=getattr(request.app.state, "backend_session", None),
    )


def admit(instance, request, operation):
    """Validate before uploads/job creation/stream headers; execution checks again."""
    binding, model, adapter, capability, required = instance.resolve(request, operation)
    for cid in required | {capability}:
        instance.registry.require_active(binding.model, cid)
    context = ExecutionContext(instance.registry, instance.session, binding, model)
    context.backend
    if not adapter.health(context):
        raise ModelNotLoadedError("Selected model backend is not healthy.")
    return model


async def media(value, kind, request):
    if isinstance(value, UploadFile):
        raw = await value.read(MAX_BODY + 1)
    elif isinstance(value, dict):
        if set(value) != {"image_url"}:
            raise UnsupportedCapabilityError(
                "Only image_url references are supported; file IDs require explicit upload resolution."
            )
        return await media(value["image_url"], kind, request)
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        from factory.apps.control_api.src.assets.service import AssetService

        raw = await AssetService(max_size_bytes=MAX_BODY).fetch_bytes(value)
    else:
        raw = data_bytes(value, kind)
    if len(raw) > MAX_BODY:
        raise PayloadTooLargeError("Media exceeds 64 MiB.")
    try:
        check_media(raw, kind)
    except Exception as exc:
        raise InvalidRequestError("Input " + kind + " cannot be decoded.") from exc
    return raw


def inline(raw, kind):
    mime = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[kind]
    if kind == "image":
        import io
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            mime = image.get_format_mimetype()
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode()


async def normalize_messages(messages, request):
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            kind = part.get("type")
            if kind in {"image_url", "video_url"}:
                value = part.get(kind)
                url = value.get("url") if isinstance(value, dict) else value
                raw = await media(url, "image" if kind == "image_url" else "video", request)
                part[kind] = {"url": inline(raw, "image" if kind == "image_url" else "video")}
    return messages


def dimensions(value, model):
    if value is None or value == "auto":
        return {k: model.parameters[k]["default"] for k in ("width", "height")}
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{1,4}x[1-9][0-9]{1,4}", value):
        raise InvalidRequestError("size must be WIDTHxHEIGHT.", details={"param": "size"})
    width, height = map(int, value.split("x"))
    return {"width": width, "height": height}


def only(body, field, allowed, default=None):
    value = body.get(field, default)
    if value not in allowed:
        raise UnsupportedCapabilityError("Unsupported " + field + ".", details={"param": field})
    return value


def number(value, name, integer=False):
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidRequestError("Invalid " + name + ".", details={"param": name})
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidRequestError("Invalid " + name + ".", details={"param": name}) from exc
    if not math.isfinite(parsed) or integer and not parsed.is_integer():
        raise InvalidRequestError("Invalid " + name + ".", details={"param": name})
    return int(parsed) if integer else parsed


async def normalize_response_input(value, request):
    if isinstance(value, str):
        return value
    for item in value:
        if not isinstance(item, dict):
            raise InvalidRequestError("Invalid response input item.")
        content = item.get("content")
        for part in content if isinstance(content, list) else []:
            if part.get("type") == "input_image":
                if part.get("file_id"):
                    raise UnsupportedCapabilityError(
                        "File IDs are not implemented for this profile."
                    )
                raw = await media(part.get("image_url"), "image", request)
                part["image_url"] = inline(raw, "image")
            elif part.get("type") != "input_text":
                raise UnsupportedCapabilityError("Unsupported response input part.")
    return value
