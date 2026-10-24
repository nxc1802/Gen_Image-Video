"""
Tier 1 Feature Coverage: Music Subsystem (Features 33 - 35)
Validates happy path and contract adherence for:
- Feature 33: Text-to-Music (T2M)
- Feature 34: Lyrics-to-Music
- Feature 35: Music Audio-to-Audio
"""

import pytest
import httpx
from tests.e2e.conftest import assert_valid_job


@pytest.mark.tier1
class TestMusicSubsystem:
    # --- Feature 33: Text-to-Music ---
    def test_f33_t2m_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {
                "prompt": "Uplifting synthpop soundtrack with 80s drums and bright arpeggiated synths",
                "duration_seconds": 60,
                "genre": "synthpop",
                "tempo": 120
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "music", "text_to_music")

    def test_f33_t2m_short_sample(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Acoustic guitar jingle", "duration_seconds": 15}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f33_t2m_genre_conditioning(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Lo-fi hip hop beat with warm vinyl crackle", "genre": "lo-fi"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f33_t2m_tempo_bpm(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Fast drum and bass groove", "tempo": 174}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f33_t2m_audio_artifact_returned(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Artifact check"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        art = res.json()["artifacts"][0]
        assert art["mime_type"] == "audio/wav"
        assert art["url"].startswith("https://s3.ai.example.com/")

    # --- Feature 34: Lyrics-to-Music ---
    def test_f34_lyrics_to_music_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {
                "prompt": "Pop ballad with soaring female vocals and grand piano",
                "lyrics": "[Verse 1]\nStars alight across the midnight blue\n[Chorus]\nFactory v4 dreams come true",
                "vocal_style": "female_soprano"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "music", "lyrics_to_music")

    def test_f34_lyrics_to_music_rock_genre(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {
                "prompt": "Hard rock anthem with distorted guitars",
                "lyrics": "Reaching for the sky / Never looking back",
                "vocal_style": "male_rock"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f34_lyrics_to_music_acoustic(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {
                "prompt": "Folk duet with acoustic guitar and cello",
                "lyrics": "Down by the riverside we walk alone"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f34_lyrics_to_music_duration(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {
                "prompt": "Jingle",
                "lyrics": "Factory v4, build fast and secure",
                "duration_seconds": 20
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f34_lyrics_to_music_artifact(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "lyrics_to_music",
            "inputs": {"prompt": "Song", "lyrics": "Test lyrics line"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.json()["artifacts"][0]["size_bytes"] > 0

    # --- Feature 35: Music Audio-to-Audio ---
    def test_f35_audio_to_audio_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "audio_to_audio",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/piano_track.wav",
                "prompt": "Remix this piano track into a modern EDM banger",
                "audio_strength": 0.6
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert_valid_job(res.json(), "music", "audio_to_audio")

    def test_f35_audio_to_audio_low_strength(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "audio_to_audio",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/sample.wav",
                "prompt": "Subtle polish and orchestral backing",
                "audio_strength": 0.3
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f35_audio_to_audio_genre_shift(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "audio_to_audio",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/vocal.wav",
                "prompt": "Transform into an 8-bit chiptune version",
                "audio_strength": 0.75
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f35_audio_to_audio_seed(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "audio_to_audio",
            "inputs": {
                "audio_url": "https://s3.ai.example.com/assets/track.wav",
                "prompt": "Reggae dub remix",
                "seed": 1010
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201

    def test_f35_audio_to_audio_artifact_checksum(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "audio_to_audio",
            "inputs": {"audio_url": "https://s3.ai.example.com/assets/melody.wav", "prompt": "Remix"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        art = res.json()["artifacts"][0]
        assert "sha256" in art
        assert len(art["sha256"]) == 64
