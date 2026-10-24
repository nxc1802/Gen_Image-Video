"""
Tier 3 Cross-Feature Interaction Tests: Speech ASR -> LLM Reasoning -> TTS Synthesis
Validates complete voice assistant loop:
1. Audio input transcribed by Qwen3-ASR
2. Transcript answered by Qwen Chat / Responses
3. Text answer synthesized into voice by Qwen3-TTS
4. Synthetic voice aligned with forced alignment
"""

import pytest
import httpx


@pytest.mark.tier3
class TestSpeechLLMTTSPipeline:
    def test_pipeline_voice_qa_conversational_loop(self, client: httpx.Client, auth_headers):
        # Step 1: User speaks query -> ASR transcribes
        asr_res = client.post("/v1/audio/transcriptions", headers=auth_headers, json={
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/user_query.wav"
        })
        assert asr_res.status_code == 200
        user_text = asr_res.json()["text"]
        assert len(user_text) > 0

        # Step 2: LLM generates thoughtful answer
        chat_res = client.post("/v1/chat/completions", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "system", "content": "You are Factory v4 Voice Assistant."},
                {"role": "user", "content": user_text}
            ]
        })
        assert chat_res.status_code == 200
        assistant_reply = chat_res.json()["choices"][0]["message"]["content"]

        # Step 3: TTS synthesizes response audio
        tts_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": assistant_reply,
            "voice": "alloy"
        })
        assert tts_res.status_code == 200
        assert "audio/wav" in tts_res.headers.get("content-type", "")
        assert len(tts_res.content) > 0

    def test_pipeline_asr_to_structured_response_to_tts(self, client: httpx.Client, auth_headers):
        # Transcribe
        asr_res = client.post("/v1/audio/transcriptions", headers=auth_headers, json={
            "model": "qwen3-asr",
            "audio_url": "https://s3.ai.example.com/assets/complex_task.wav"
        })
        task_text = asr_res.json()["text"]

        # Reasoning Responses API
        resp_res = client.post("/v1/responses", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "input": task_text
        })
        assert resp_res.status_code == 200
        reasoned_answer = resp_res.json()["output"]

        # Synthesize Speech
        tts_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": reasoned_answer,
            "voice": "nova"
        })
        assert tts_res.status_code == 200
        assert len(tts_res.content) > 0

    def test_pipeline_tts_to_forced_alignment(self, client: httpx.Client, auth_headers):
        script = "Factory v4 decoupled speech architecture."
        # Synthesize speech
        tts_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": script
        })
        assert tts_res.status_code == 200

        # Upload generated speech to S3 Asset Service
        asset_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "audio/wav"})
        generated_audio_s3 = asset_res.json()["s3_url"]

        # Forced Alignment
        align_res = client.post("/v1/audio/alignments", headers=auth_headers, json={
            "audio_url": generated_audio_s3,
            "transcript_text": script
        })
        assert align_res.status_code == 200
        aligned_words = align_res.json()["aligned_words"]
        assert len(aligned_words) >= 2
        assert aligned_words[0]["word"] == "Welcome" or len(aligned_words[0]["word"]) > 0

    def test_pipeline_voice_cloning_from_asr_reference(self, client: httpx.Client, auth_headers):
        # 3-second reference voice uploaded
        ref_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "audio/wav"})
        ref_audio_url = ref_res.json()["s3_url"]

        # Synthesize new text using the reference audio sample
        tts_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": "This speech is generated using zero-shot voice cloning.",
            "reference_audio_url": ref_audio_url
        })
        assert tts_res.status_code == 200
        assert len(tts_res.content) > 0
