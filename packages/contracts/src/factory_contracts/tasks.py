"""
factory_contracts/tasks.py
Authoritative Canonical Identifier Resolution and Normalization Maps for Models and Tasks.
Provides bidirectional resolution between capability IDs, aliases, and canonical names.
"""
from typing import Dict, List, Optional

# =========================================================================
# Task Resolution Maps
# =========================================================================

CANONICAL_TASK_MAP: Dict[str, str] = {
    # Image Capability IDs & Aliases -> Canonical Task
    "image.t2i": "text_to_image",
    "t2i": "text_to_image",
    "text_to_image": "text_to_image",
    "text2image": "text_to_image",

    "image.i2i": "image_to_image",
    "i2i": "image_to_image",
    "image_to_image": "image_to_image",
    "image2image": "image_to_image",

    "image.inpaint": "inpaint",
    "inpaint": "inpaint",
    "inpainting": "inpaint",

    "image.outpaint": "outpaint",
    "outpaint": "outpaint",
    "outpainting": "outpaint",

    "image.instruction_edit": "instruction_edit",
    "instruction_edit": "instruction_edit",
    "edit": "instruction_edit",
    "image_edit": "instruction_edit",

    "image.reference": "reference_conditioning",
    "image.reference_conditioning": "reference_conditioning",
    "reference_conditioning": "reference_conditioning",
    "reference": "reference_conditioning",
    "ref2img": "reference_conditioning",

    "image.controlnet": "controlnet",
    "controlnet": "controlnet",
    "control_pose": "controlnet",
    "control_depth": "controlnet",

    # Video Capability IDs & Aliases -> Canonical Task
    "video.t2v": "text_to_video",
    "t2v": "text_to_video",
    "text_to_video": "text_to_video",
    "text2video": "text_to_video",

    "video.i2v": "image_to_video",
    "i2v": "image_to_video",
    "image_to_video": "image_to_video",
    "image2video": "image_to_video",

    "video.fl2v": "first_last_frame_to_video",
    "fl2v": "first_last_frame_to_video",
    "first_last_frame_to_video": "first_last_frame_to_video",
    "first_last_frame": "first_last_frame_to_video",

    "video.extend": "video_extension",
    "video.extension": "video_extension",
    "extension": "video_extension",
    "video_extension": "video_extension",

    "video.audio": "video_with_audio",
    "video.with_audio": "video_with_audio",
    "with_audio": "video_with_audio",
    "video_with_audio": "video_with_audio",

    "video.reference": "reference_conditioning",
    "video.ref2va": "reference_conditioning",
    "ref2va": "reference_conditioning",

    # Audio & Music Capability IDs & Aliases -> Canonical Task
    "music.t2m": "text_to_music",
    "audio.music": "text_to_music",
    "t2m": "text_to_music",
    "text_to_music": "text_to_music",

    "music.lyrics": "lyrics_to_music",
    "music.l2m": "lyrics_to_music",
    "l2m": "lyrics_to_music",
    "lyrics_to_music": "lyrics_to_music",

    "music.a2a": "audio_to_audio",
    "a2a": "audio_to_audio",
    "audio_to_audio": "audio_to_audio",

    # Speech Capability IDs & Aliases -> Canonical Task
    "speech.tts": "speech_synthesis",
    "speech.synthesis": "speech_synthesis",
    "tts": "speech_synthesis",
    "text_to_speech": "speech_synthesis",
    "text2speech": "speech_synthesis",
    "speech_synthesis": "speech_synthesis",

    "speech.asr": "speech_transcription",
    "speech.transcription": "speech_transcription",
    "asr": "speech_transcription",
    "speech_to_text": "speech_transcription",
    "speech2text": "speech_transcription",
    "speech_transcription": "speech_transcription",

    # Enhancement Capability IDs & Aliases -> Canonical Task
    "enhance.sr_image": "image_super_resolution",
    "enhancement.sr": "image_super_resolution",
    "super_resolution": "image_super_resolution",
    "image_super_resolution": "image_super_resolution",

    "enhance.sr_video": "video_super_resolution",
    "enhancement.vsr": "video_super_resolution",
    "video_super_resolution": "video_super_resolution",

    "enhance.interpolation": "frame_interpolation",
    "enhancement.interp": "frame_interpolation",
    "frame_interpolation": "frame_interpolation",

    "enhance.audio": "audio_enhancement",
    "enhancement.audio": "audio_enhancement",
    "audio_enhancement": "audio_enhancement",

    # Text / VLM / Decision / Embedding
    "chat": "chat_completion",
    "chat_completion": "chat_completion",
    "text.chat": "chat_completion",
    "vlm": "visual_language",
    "visual_language": "visual_language",
    "vision.vlm": "visual_language",
    "decision": "decision",
    "decision.classify": "decision",
    "embedding": "embedding",
    "embedding.text": "embedding",
    "rerank": "rerank",
    "rerank.text": "rerank",
}

# Reverse mapping: Canonical Task -> Primary Capability ID
CANONICAL_TO_CAPABILITY_MAP: Dict[str, str] = {
    "text_to_image": "image.t2i",
    "image_to_image": "image.i2i",
    "inpaint": "image.inpaint",
    "outpaint": "image.outpaint",
    "instruction_edit": "image.instruction_edit",
    "reference_conditioning": "image.reference_conditioning",
    "controlnet": "image.controlnet",
    "text_to_video": "video.t2v",
    "image_to_video": "video.i2v",
    "first_last_frame_to_video": "video.fl2v",
    "video_extension": "video.extension",
    "video_with_audio": "video.with_audio",
    "text_to_music": "music.t2m",
    "lyrics_to_music": "music.lyrics",
    "audio_to_audio": "music.a2a",
    "speech_synthesis": "speech.tts",
    "speech_transcription": "speech.asr",
    "image_super_resolution": "enhance.sr_image",
    "video_super_resolution": "enhance.sr_video",
    "frame_interpolation": "enhance.interpolation",
    "audio_enhancement": "enhance.audio",
    "chat_completion": "text.chat",
    "visual_language": "vision.vlm",
    "decision": "decision.classify",
    "embedding": "embedding.text",
    "rerank": "rerank.text",
}


# =========================================================================
# Model Resolution Maps
# =========================================================================

CANONICAL_MODEL_MAP: Dict[str, str] = {
    # Image
    "qwen-image": "qwen-image-2.1",
    "qwen-image-2.1": "qwen-image-2.1",
    "qwen-image-2.1-turbo": "qwen-image-2.1",
    "qwen-image-turbo": "qwen-image-2.1",
    "qwen2.1-image": "qwen-image-2.1",
    "qwen_image": "qwen-image-2.1",
    "qwen_image_2.1": "qwen-image-2.1",
    "qwen_image_2.1_turbo": "qwen-image-2.1",

    # Video
    "wan": "wan-2.2",
    "wan-2.2": "wan-2.2",
    "wan2.2": "wan-2.2",
    "wan_2.2": "wan-2.2",
    "minimax": "minimax-h3",
    "minimax-h3": "minimax-h3",
    "minimax_h3": "minimax-h3",

    # Music
    "ace-step": "ace-step-1.5",
    "ace-step-1.5": "ace-step-1.5",
    "ace_step": "ace-step-1.5",
    "ace_step_1.5": "ace-step-1.5",

    # Text & VLM
    "qwen-text": "qwen-text",
    "qwen-2.5-72b": "qwen-text",
    "qwen-2.5-72b-instruct": "qwen-text",
    "qwen_text": "qwen-text",
    "qwen-vlm": "qwen-vlm",
    "qwen-2.5-vl-72b": "qwen-vlm",
    "qwen_vlm": "qwen-vlm",

    # Decision & Embedding
    "qwen-decision": "qwen-decision",
    "qwen_decision": "qwen-decision",
    "qwen-embedding": "qwen-embedding",
    "qwen_embedding": "qwen-embedding",

    # Speech
    "qwen-speech": "qwen-speech",
    "qwen_speech": "qwen-speech",
    "qwen3-tts": "qwen-speech",
    "qwen3-asr": "qwen-speech",

    # Enhancement
    "enhancement-suite": "enhancement-suite-v1",
    "enhancement-suite-v1": "enhancement-suite-v1",
    "enhancement_suite": "enhancement-suite-v1",
    "enhancement_suite_v1": "enhancement-suite-v1",
}

MODEL_TO_FAMILY_MAP: Dict[str, str] = {
    "qwen-image-2.1": "qwen-image",
    "wan-2.2": "wan",
    "minimax-h3": "minimax",
    "ace-step-1.5": "ace-step",
    "qwen-text": "qwen-text",
    "qwen-vlm": "qwen-vlm",
    "qwen-decision": "qwen-decision",
    "qwen-speech": "qwen-speech",
    "qwen-embedding": "qwen-embedding",
    "enhancement-suite-v1": "enhancement-suite",
}


# =========================================================================
# Resolution & Normalization Helper Functions
# =========================================================================

def normalize_task(task: Optional[str]) -> str:
    """
    Normalizes any capability ID (e.g. 'image.t2i'), short alias ('t2i'),
    or canonical name into its canonical task string.
    If not found in CANONICAL_TASK_MAP, returns trimmed, lowercased string.
    """
    if not task:
        return ""
    key = str(task).strip().lower()
    return CANONICAL_TASK_MAP.get(key, key)


def task_to_capability_id(canonical_task: Optional[str]) -> Optional[str]:
    """
    Resolves a canonical task name (e.g. 'text_to_image') to its authoritative
    CapabilityManifest ID (e.g. 'image.t2i').
    """
    if not canonical_task:
        return None
    normalized = normalize_task(canonical_task)
    return CANONICAL_TO_CAPABILITY_MAP.get(normalized)


def get_task_aliases(canonical_task: str) -> List[str]:
    """Returns all aliases and capability IDs that resolve to the given canonical task."""
    normalized = normalize_task(canonical_task)
    return [alias for alias, target in CANONICAL_TASK_MAP.items() if target == normalized]


def normalize_model(model: Optional[str]) -> str:
    """
    Normalizes any model family, short alias, or checkpoint variant
    into its authoritative canonical model ID (e.g. 'qwen-image' -> 'qwen-image-2.1').
    If not found in CANONICAL_MODEL_MAP, returns trimmed, lowercased string.
    """
    if not model:
        return ""
    key = str(model).strip().lower()
    return CANONICAL_MODEL_MAP.get(key, key)


def model_to_family(canonical_model: Optional[str]) -> Optional[str]:
    """
    Resolves a canonical model ID (e.g. 'qwen-image-2.1') to its manifest
    model family (e.g. 'qwen-image').
    """
    if not canonical_model:
        return None
    normalized = normalize_model(canonical_model)
    return MODEL_TO_FAMILY_MAP.get(normalized)


def get_model_aliases(canonical_model: str) -> List[str]:
    """Returns all aliases and family names that resolve to the given canonical model."""
    normalized = normalize_model(canonical_model)
    return [alias for alias, target in CANONICAL_MODEL_MAP.items() if target == normalized]
