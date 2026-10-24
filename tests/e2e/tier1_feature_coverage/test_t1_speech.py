"""
Tier 1 Feature Coverage: Speech Subsystem (Features 15 - 18)
Validates happy path and contract adherence for:
- Feature 15: Speech Audio Transcription (POST /v1/audio/transcriptions)
- Feature 16: Speech Streaming ASR (POST /v1/audio/transcriptions/stream)
- Feature 17: Speech Forced Alignment (POST /v1/audio/alignments)
- Feature 18: Speech Synthesis TTS (POST /v1/audio/speech)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestSpeechSubsystem:
    # --- Feature 15: Speech Audio Transcription ---
    def test_f15_audio_transcription_text(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/sample_speech.wav"
        }
        res = client.post("/v1/audio/transcriptions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "text" in data
        assert len(data["text"]) > 0

    def test_f15_audio_transcription_language_detection(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/english_speech.wav"
        }
        res = client.post("/v1/audio/transcriptions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["language"] == "en"

    def test_f15_audio_transcription_duration(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/short_speech.wav"
        }
        res = client.post("/v1/audio/transcriptions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["duration"] > 0

    def test_f15_audio_transcription_segments(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/lecture.wav"
        }
        res = client.post("/v1/audio/transcriptions", headers=auth_headers, json=payload)
        assert res.status_code == 200
        segments = res.json()["segments"]
        assert len(segments) >= 2
        for seg in segments:
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert seg["end"] >= seg["start"]

    def test_f15_audio_transcription_binary_upload(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "audio/wav"}
        dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        res = client.post("/v1/audio/transcriptions", headers=headers, content=dummy_wav)
        assert res.status_code == 200
        assert "text" in res.json()

    # --- Feature 16: Speech Streaming ASR ---
    def test_f16_streaming_asr_chunk_status(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/audio/transcriptions/stream", headers=auth_headers, json={"chunk_id": 1, "audio_base64": "AAA="})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "streaming_connected"
        assert data["chunk_received"] is True

    def test_f16_streaming_asr_interim_text(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/audio/transcriptions/stream", headers=auth_headers, json={"chunk_id": 2, "audio_base64": "BBB="})
        assert res.status_code == 200
        assert "interim_text" in res.json()

    def test_f16_streaming_asr_sequential_chunks(self, client: httpx.Client, auth_headers):
        for idx in range(1, 4):
            res = client.post("/v1/audio/transcriptions/stream", headers=auth_headers, json={"chunk_id": idx, "audio_base64": "CCC="})
            assert res.status_code == 200
            assert res.json()["chunk_received"] is True

    def test_f16_streaming_asr_content_type(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/audio/transcriptions/stream", headers=auth_headers, json={"chunk_id": 1})
        assert "application/json" in res.headers.get("content-type", "")

    def test_f16_streaming_asr_model_header(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/audio/transcriptions/stream", headers=auth_headers, json={"model": "qwen3-asr", "chunk_id": 1})
        assert res.status_code == 200

    # --- Feature 17: Speech Forced Alignment ---
    def test_f17_forced_alignment_aligned_words(self, client: httpx.Client, auth_headers):
        payload = {
            "audio_url": "https://s3.ai.example.com/assets/aligned_audio.wav",
            "transcript_text": "Welcome to Factory v4"
        }
        res = client.post("/v1/audio/alignments", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "aligned_words" in data
        assert len(data["aligned_words"]) >= 2

    def test_f17_forced_alignment_timestamps(self, client: httpx.Client, auth_headers):
        payload = {
            "audio_url": "https://s3.ai.example.com/assets/aligned_audio.wav",
            "transcript_text": "Welcome to Factory"
        }
        res = client.post("/v1/audio/alignments", headers=auth_headers, json=payload)
        assert res.status_code == 200
        words = res.json()["aligned_words"]
        for w in words:
            assert "start_ms" in w
            assert "end_ms" in w
            assert w["end_ms"] >= w["start_ms"]

    def test_f17_forced_alignment_monotonic_order(self, client: httpx.Client, auth_headers):
        payload = {
            "audio_url": "https://s3.ai.example.com/assets/aligned_audio.wav",
            "transcript_text": "Welcome Factory"
        }
        res = client.post("/v1/audio/alignments", headers=auth_headers, json=payload)
        words = res.json()["aligned_words"]
        assert words[1]["start_ms"] >= words[0]["end_ms"]

    def test_f17_forced_alignment_scores(self, client: httpx.Client, auth_headers):
        payload = {
            "audio_url": "https://s3.ai.example.com/assets/aligned_audio.wav",
            "transcript_text": "Welcome"
        }
        res = client.post("/v1/audio/alignments", headers=auth_headers, json=payload)
        words = res.json()["aligned_words"]
        for w in words:
            assert 0.0 <= w["score"] <= 1.0

    def test_f17_forced_alignment_word_strings(self, client: httpx.Client, auth_headers):
        payload = {
            "audio_url": "https://s3.ai.example.com/assets/aligned_audio.wav",
            "transcript_text": "Welcome Factory"
        }
        res = client.post("/v1/audio/alignments", headers=auth_headers, json=payload)
        words = res.json()["aligned_words"]
        assert words[0]["word"] == "Welcome"
        assert words[1]["word"] == "Factory"

    # --- Feature 18: Speech Synthesis TTS ---
    def test_f18_tts_wav_output(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-tts",
            "input": "Factory v4 decoupled speech synthesis active.",
            "voice": "alloy"
        }
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert "audio/wav" in res.headers.get("content-type", "")
        assert len(res.content) > 0

    def test_f18_tts_voice_cloning_reference(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-tts",
            "input": "Cloned voice speaking now.",
            "reference_audio_url": "https://s3.ai.example.com/assets/speaker_sample.wav"
        }
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert len(res.content) > 0

    def test_f18_tts_custom_voice_preset(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-tts",
            "input": "Testing preset voice.",
            "voice": "nova"
        }
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert res.status_code == 200

    def test_f18_tts_binary_header_check(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-tts",
            "input": "WAV file signature check."
        }
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert res.status_code == 200
        # Check RIFF header bytes
        assert res.content.startswith(b"RIFF")

    def test_f18_tts_long_sentence(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen3-tts",
            "input": "Factory v4 eliminates the monolithic inference server and deploys native frameworks behind static ngrok domains with full security."
        }
        res = client.post("/v1/audio/speech", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert len(res.content) > 10
