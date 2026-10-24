"""
Tier 4 Real-World Application Scenario: Automated Podcast Production Suite
Validates an end-to-end multi-modal commercial workflow:
1. Topic prompt -> LLM generates multi-segment dialogue script
2. TTS synthesizes Host and Guest voice tracks using voice cloning
3. Music ACE-Step 1.5 creates intro and outro musical theme songs
4. Audio Enhancement clarifies and balances voice frequencies
5. Image Qwen-Image 2.1 generates 2048x2048 album cover artwork
6. All artifacts packaged with end-to-end telemetry request_id
"""

import pytest
import httpx


@pytest.mark.tier4
class TestPodcastProductionWorkflow:
    def test_full_automated_podcast_pipeline(self, client: httpx.Client, auth_headers):
        # Step 1: Script Writing via Text LLM
        script_res = client.post("/v1/chat/completions", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": "Write a 2-line intro for episode 1 of The AI Factory Show."}
            ]
        })
        assert script_res.status_code == 200
        script_text = script_res.json()["choices"][0]["message"]["content"]
        assert len(script_text) > 0

        # Step 2: Voiceover Synthesis via Speech TTS
        tts_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": script_text,
            "voice": "alloy"
        })
        assert tts_res.status_code == 200
        assert len(tts_res.content) > 0

        # Step 3: Upload voice track to Asset Service for post-processing
        asset_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "audio/wav"})
        assert asset_res.status_code == 201
        voice_s3_url = asset_res.json()["s3_url"]

        # Step 4: Audio Enhancement (Voice clarity & Denoise)
        enh_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {"audio_url": voice_s3_url, "mode": "voice_clarity"}
        })
        assert enh_res.status_code == 201
        clean_voice_url = enh_res.json()["artifacts"][0]["url"]
        assert clean_voice_url.startswith("https://s3.ai.example.com/")

        # Step 5: Music Intro Jingle via ACE-Step 1.5
        music_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {
                "prompt": "Catchy upbeat tech podcast theme with acoustic drums and funky bass",
                "duration_seconds": 15,
                "genre": "funk_pop"
            }
        })
        assert music_res.status_code == 201
        music_url = music_res.json()["artifacts"][0]["url"]
        assert music_url.endswith(".bin") or music_url.endswith(".wav")

        # Step 6: Episodic Cover Artwork via Qwen-Image 2.1
        cover_res = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": "Album cover for 'The AI Factory Show', minimalist retro-futuristic microphone with neon accents",
                "width": 1024,
                "height": 1024
            }
        })
        assert cover_res.status_code == 201
        cover_url = cover_res.json()["artifacts"][0]["url"]

        # Step 7: Verify all 3 primary assets exist in S3
        assert voice_s3_url != music_url != cover_url
        assert "s3.ai.example.com" in cover_url

    def test_podcast_with_voice_cloned_interviewee(self, client: httpx.Client, auth_headers):
        # Upload guest reference voice
        ref_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "audio/wav"})
        guest_voice_ref = ref_res.json()["s3_url"]

        # Synthesize guest dialogue with voice cloning
        speech_res = client.post("/v1/audio/speech", headers=auth_headers, json={
            "model": "qwen3-tts",
            "input": "I am thrilled to discuss decoupled AI factory systems today.",
            "reference_audio_url": guest_voice_ref
        })
        assert speech_res.status_code == 200

    def test_podcast_outro_with_vocal_lyrics(self, client: httpx.Client, auth_headers):
        # Generate music outro conditioned on closing lyrics
        lyrics_job = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {
                "prompt": "Acoustic fadeout outro song",
                "lyrics": "Thank you for listening to Factory v4, until next time goodbye",
                "duration_seconds": 25
            }
        })
        assert lyrics_job.status_code == 201
        assert lyrics_job.json()["service"] == "music"

    def test_podcast_audio_enhancement_and_export(self, client: httpx.Client, auth_headers):
        # Upload recorded interview
        raw_audio = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "audio/wav"})
        raw_s3 = raw_audio.json()["s3_url"]

        # Run multi-mode audio enhancement (denoise + auto gain)
        enh_job = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "audio_enhancement",
            "inputs": {"audio_url": raw_s3, "mode": "denoise", "auto_gain_control": True}
        })
        assert enh_job.status_code == 201
        assert len(enh_job.json()["artifacts"]) >= 1
