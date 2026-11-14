"""Canonical capability vocabulary and the sole request-alias registry.

ModelManifest owns model support. This registry owns names, never readiness.
Aliases are resolved by request/config/storage boundaries, not by dispatch code.
"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Capability:
    id: str
    aliases: tuple[str, ...] = ()
    legacy_key: str | None = None

    @property
    def engine(self):
        return self.id.split(".", 1)[0]


_DEFINITIONS = (
    Capability("image.text_to_image", ("image.t2i", "t2i", "text_to_image", "text2image")),
    Capability("image.image_to_image", ("image.i2i", "i2i", "image_to_image", "image2image")),
    Capability("image.inpaint", ("inpaint", "inpainting")),
    Capability("image.outpaint", ("outpaint", "outpainting")),
    Capability("image.instruction_edit", ("instruction_edit", "edit", "image_edit")),
    Capability("image.reference_image", ("image.reference", "image.reference_conditioning", "reference_image", "reference_conditioning", "reference", "ref2img")),
    Capability("image.multi_reference", ("multi_reference",)),
    Capability("image.controlnet", ("controlnet",)),
    Capability("image.control_pose", ("control_pose",)),
    Capability("image.control_depth", ("control_depth",)),
    Capability("video.text_to_video", ("video.t2v", "t2v", "text_to_video", "text2video")),
    Capability("video.image_to_video", ("video.i2v", "i2v", "image_to_video", "image2video")),
    Capability("video.first_last_frame_to_video", ("video.fl2v", "fl2v", "first_last_frame_to_video", "first_last_frame")),
    Capability("video.video_extension", ("video.extend", "video.extension", "extension", "video_extension")),
    Capability("video.video_with_audio", ("video.audio", "video.with_audio", "with_audio", "video_with_audio")),
    Capability("video.reference_conditioning", ("video.reference", "video.ref2va", "ref2va", "reference_conditioning", "reference_image", "reference")),
    Capability("music.text_to_music", ("music.t2m", "audio.music", "t2m", "text_to_music", "text2music")),
    Capability("music.lyrics_to_music", ("music.lyrics", "music.l2m", "l2m", "lyrics_to_music", "lyrics2music")),
    Capability("music.audio_to_audio", ("music.a2a", "a2a", "audio_to_audio")),
    Capability("music.audio_to_music", ("audio_to_music",)),
    Capability("music.cover", ("cover", "remix")),
    Capability("music.repaint", ("repaint",)),
    Capability("music.reference_audio", ("music.reference", "reference_audio", "reference_conditioning", "reference")),
    Capability("music.extract", ("extract",)),
    Capability("music.lego", ("lego",)),
    Capability("music.complete", ("complete",)),
    Capability("music.extend", ("music_extend", "extend"), "music_extend"),
    Capability("music.symbolic_score", ("symbolic_score", "symbolic", "score", "music.score", "music.symbolic")),
    Capability("music.custom_score", ("custom_score", "custom", "music.custom")),
    Capability("music.score_edit", ("score_edit", "edit_score", "edit", "music.edit")),
    Capability("music.song_variation", ("song_variation", "variation", "song_var", "music.variation")),
    Capability("music.cover_edit", ("cover_edit",)),
    Capability("speech.tts", ("speech.synthesis", "tts", "text_to_speech", "text2speech", "speech_synthesis"), "text_to_speech"),
    Capability("speech.asr", ("speech.transcription", "asr", "speech_to_text", "speech2text", "speech_transcription", "transcription"), "speech_to_text"),
    Capability("speech.word_timestamps", ("word_timestamps",)),
    Capability("speech.voice_cloning", ("voice_cloning", "voice_clone")),
    Capability("enhancement.image_super_resolution", ("enhance.sr_image", "enhancement.sr", "super_resolution", "image_super_resolution")),
    Capability("enhancement.video_super_resolution", ("enhance.sr_video", "enhancement.vsr", "video_super_resolution")),
    Capability("enhancement.frame_interpolation", ("enhance.interpolation", "enhancement.interp", "frame_interpolation")),
    Capability("enhancement.audio_enhancement", ("enhance.audio", "enhancement.audio", "audio_enhancement")),
    Capability("responses.chat_completion", ("chat", "chat_completion", "text.chat")),
    Capability("responses.create", ("responses_api",), "responses_api"),
    Capability("responses.structured_output", ("structured_output",)),
    Capability("responses.tool_calling", ("tool_calling",)),
    Capability("responses.streaming", ("streaming",)),
    Capability("responses.image_chat", ("image_chat", "visual_language", "vlm", "vision.vlm")),
    Capability("responses.video_chat", ("video_chat",)),
    Capability("responses.visual_grounding", ("visual_grounding",)),
    Capability("responses.document_qa", ("document_qa",)),
    Capability("decision.predicate_evaluation", ("predicate_evaluation", "predicate", "decision.classify")),
    Capability("decision.choice_routing", ("choice_routing", "choice")),
    Capability("decision.score_ranking", ("score_ranking", "score")),
    Capability("vector.text_embedding", ("text_embedding", "embedding", "embeddings", "vector_embeddings", "embedding.text")),
    Capability("vector.multimodal_embedding", ("multimodal_embedding",)),
    Capability("vector.reranking", ("reranking", "rerank", "rerank.text")),
)

CAPABILITIES = MappingProxyType({item.id: item for item in _DEFINITIONS})
if len(CAPABILITIES) != len(_DEFINITIONS):
    raise ValueError("Duplicate canonical capability ID")

_ENGINE_CONTEXTS = MappingProxyType({"text": "responses", "vlm": "responses", "vision": "responses",
    "embedding": "vector", "embed": "vector", "rerank": "vector"})


def engine_context(value):
    key = str(getattr(value, "value", value) or "").strip().lower().removesuffix("-service")
    return _ENGINE_CONTEXTS.get(key, key)


def _make_index(definitions):
    index = {}
    for item in definitions:
        for alias in (item.id, *item.aliases):
            targets = index.setdefault(alias, set())
            if any(CAPABILITIES[target].engine == item.engine and target != item.id for target in targets):
                raise ValueError(f"Ambiguous alias within engine: {alias}")
            targets.add(item.id)
    return MappingProxyType({alias: frozenset(targets) for alias, targets in index.items()})


_ALIASES = _make_index(_DEFINITIONS)
# Compatibility imports are derived read-only views, never separately maintained maps.
CANONICAL_TASK_MAP = MappingProxyType({key: next(iter(ids)) for key, ids in _ALIASES.items() if len(ids) == 1})
CANONICAL_TO_CAPABILITY_MAP = MappingProxyType({key: key for key in CAPABILITIES})


def normalize_capability(value, *, engine=None, inputs=None):
    """Resolve at an ingress boundary; ambiguous short names require context.

    Unknown values stay unknown so admission can report unsupported capability.
    Input-dependent refinement happens here before a job is persisted.
    """
    key = str(value or "").strip().lower()
    targets = _ALIASES.get(key, frozenset())
    context = engine_context(engine)
    if context:
        targets = frozenset(cid for cid in targets if CAPABILITIES[cid].engine == context)
    if len(targets) > 1:
        raise ValueError(f"Ambiguous capability alias '{key}'; specify its engine/service or canonical ID")
    cid = next(iter(targets), key)
    if inputs is not None:
        if cid == "image.reference_image" and isinstance(inputs.get("reference_image_urls"), list) and len(inputs["reference_image_urls"]) > 1:
            cid = "image.multi_reference"
        elif cid == "image.controlnet":
            kind = str(inputs.get("control_type", "pose")).strip().lower()
            if kind not in {"pose", "depth"}:
                raise ValueError(f"Unsupported control_type '{kind}'")
            cid = f"image.control_{kind}"
        elif cid in {"image.control_pose", "image.control_depth"} and "control_type" in inputs:
            if inputs["control_type"] != cid.removeprefix("image.control_"):
                raise ValueError("control_type conflicts with the explicit canonical capability")
    return cid


def require_canonical(value, *, engine=None):
    item = CAPABILITIES.get(value)
    if item is None or (engine is not None and item.engine != engine_context(engine)):
        raise ValueError(f"Expected canonical capability ID for {engine or 'a registered engine'}: {value}")
    return value


def capability_ids(engine):
    return tuple(cid for cid, item in CAPABILITIES.items() if item.engine == engine_context(engine))


def capability_aliases(cid):
    item = CAPABILITIES.get(cid)
    return [item.id, *item.aliases] if item else []


def canonical_model_capabilities(model):
    """Read canonical support; adapt v1 migration catalogs at the catalog boundary."""
    if model.apiVersion == "factory.ai/v2":
        return dict(model.capabilities)
    return {normalize_capability(key, engine=model.metadata.service_id): value for key, value in model.capabilities.items()}


def model_capability_truth(model, cid):
    if model.apiVersion == "factory.ai/v2":
        return model.get_capability_truth(cid)
    # v1 is a migration input. Its original fields remain readable by legacy tests.
    for key in model.capabilities:
        if normalize_capability(key, engine=model.metadata.service_id) == cid:
            return model.get_capability_truth(key)
    return model.get_capability_truth(cid)
