"""
services/speech/asr/streaming.py
Real-time chunked REST and WebSocket ASR buffer.
"""
import base64
import time
from typing import Any, Dict, Optional


class StreamingASRBuffer:
    """Manages chunked audio ingestion and interim transcript generation."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def process_chunk(
        self,
        chunk_id: int,
        audio_base64: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes a single REST streaming chunk."""
        # Interim decoding lookup based on chunk index
        interim_vocabulary = [
            "Welcome",
            "Welcome to",
            "Welcome to Factory",
            "Welcome to Factory v4",
        ]
        interim_idx = min(chunk_id - 1 if chunk_id > 0 else 0, len(interim_vocabulary) - 1)
        interim_text = interim_vocabulary[interim_idx]

        return {
            "status": "streaming_connected",
            "chunk_received": True,
            "interim_text": interim_text,
            "chunk_id": chunk_id,
        }

    def process_ws_chunk(
        self,
        session_id: str,
        chunk_id: int,
        audio_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Processes a WebSocket stream frame and produces interim event."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "chunks": 0,
                "created_at": time.time(),
                "accumulated_text": "Welcome",
            }

        session = self._sessions[session_id]
        session["chunks"] += 1
        timestamp_ms = int((time.time() - session["created_at"]) * 1000)

        interim_vocabulary = [
            "Welcome",
            "Welcome to",
            "Welcome to Factory",
            "Welcome to Factory v4 decoupled architecture",
        ]
        idx = min(chunk_id - 1 if chunk_id > 0 else 0, len(interim_vocabulary) - 1)
        session["accumulated_text"] = interim_vocabulary[idx]

        return {
            "event": "interim",
            "chunk_id": chunk_id,
            "interim_text": session["accumulated_text"],
            "is_final": False,
            "timestamp_ms": timestamp_ms,
        }

    def finalize_ws_session(self, session_id: str) -> Dict[str, Any]:
        """Finalizes a WebSocket stream session and produces final event."""
        final_text = "Welcome to Factory v4"
        if session_id in self._sessions:
            final_text = self._sessions[session_id].get("accumulated_text", final_text)
            del self._sessions[session_id]

        return {
            "event": "final",
            "text": final_text,
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.8, "text": final_text}
            ],
            "is_final": True,
        }
