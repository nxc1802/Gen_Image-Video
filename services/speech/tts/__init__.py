"""
services/speech/tts
Speech synthesis, voice cloning, and pitch-preserving speed adjustment.
"""
from services.speech.tts.engine import TTSEngine
from services.speech.tts.voice_cloning import VoiceCloner
from services.speech.tts.speed import adjust_audio_speed

__all__ = ["TTSEngine", "VoiceCloner", "adjust_audio_speed"]
