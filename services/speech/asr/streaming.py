"""
Chunked compatibility streaming backed by the real ASR engine.

Each update is decoded by the configured Qwen3-ASR backend. No canned interim
transcript is ever emitted. Native realtime clients should prefer vLLM's
/v1/realtime endpoint directly.
"""
import base64
import time
from typing import Any, Dict, Optional

from factory_contracts.errors import InvalidRequestError
from services.speech.asr.engine import ASREngine


class StreamingASRBuffer:
    def __init__(self, engine: Optional[ASREngine] = None):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.engine = engine or ASREngine()

    def process_chunk(
        self,
        chunk_id: int,
        audio_base64: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not audio_base64:
            raise InvalidRequestError("Streaming ASR requires non-empty audio_base64.")
        try:
            audio = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise InvalidRequestError("audio_base64 is not valid base64 audio.") from exc
        result = self.engine.transcribe(audio_bytes=audio, model=model or "qwen3-asr")
        return {
            "status": "streaming_connected",
            "chunk_received": True,
            "interim_text": result.get("text", ""),
            "chunk_id": chunk_id,
            "segments": result.get("segments", []),
        }

    def process_ws_chunk(
        self,
        session_id: str,
        chunk_id: int,
        audio_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        session = self._sessions.setdefault(
            session_id,
            {"created_at": time.time(), "last_text": "", "chunks": 0},
        )
        session["chunks"] += 1
        if audio_data:
            result = self.engine.transcribe(audio_bytes=audio_data, model="qwen3-asr")
            session["last_text"] = result.get("text", "")
            session["last_segments"] = result.get("segments", [])
        return {
            "event": "interim",
            "chunk_id": chunk_id,
            "interim_text": session["last_text"],
            "segments": session.get("last_segments", []),
            "is_final": False,
            "timestamp_ms": int((time.time() - session["created_at"]) * 1000),
        }

    def finalize_ws_session(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.pop(session_id, None) or {}
        return {
            "event": "final",
            "text": session.get("last_text", ""),
            "segments": session.get("last_segments", []),
            "is_final": True,
        }
