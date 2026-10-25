"""
factory_contracts/tasks.py
Authoritative Canonical Task ID Resolution and Normalization Map.
"""
from typing import Dict, Optional

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
    "speech_synthesis": "speech_synthesis",

    "speech.asr": "speech_transcription",
    "speech.transcription": "speech_transcription",
    "asr": "speech_transcription",
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
    "vlm": "visual_language",
    "visual_language": "visual_language",
    "decision": "decision",
    "embedding": "embedding",
    "rerank": "rerank",
}


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
