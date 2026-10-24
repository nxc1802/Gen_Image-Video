"""
Deprecated compatibility shim.

Voice cloning is executed by native Qwen3-TTS/vLLM-Omni using ref_audio and
ref_text. Factory must never derive a fake speaker embedding from a URL hash.
"""
from factory_contracts.errors import ModelUnavailableError


class VoiceCloner:
    def extract_speaker_profile(self, reference_audio_url: str):
        raise ModelUnavailableError(
            message=(
                "Local speaker-profile synthesis is disabled. "
                "Use native Qwen3-TTS voice cloning with reference audio and reference text."
            )
        )
