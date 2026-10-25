"""
tests/adversarial/test_adversarial_m3_it2_challenger2.py
Factory v4 Milestone 3 Iteration 2 Gate Adversarial Verification Suite (Challenger 2).

Adversarially probes:
1. Speech WebSocket Security & Disconnect Crash Resilience (/v1/audio/transcriptions/ws).
2. Image & Video Gateways /ws Auth Bypass Defense (including FACTORY_API_KEY="").
3. SSRF Protection Across Subsystems (Speech, Music, Enhancement external URLs).
4. Error Sanitization, X-Request-ID Tracing, and Zero Exception String Leaks.
5. Workflow Registries Deep Copy Isolation & Task Alias Access.
"""
import copy
import json
import os
import uuid
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from factory_auth.config import AuthConfig
from services.speech.server import app as speech_app
from services.music.server import app as music_app
from services.enhancement.server import app as enhancement_app
from services.image.gateway import app as image_gateway_app
from services.video.gateway import app as video_gateway_app
import services.image.gateway as image_gateway_mod
import services.video.gateway as video_gateway_mod
from workflows.image.registry import ImageWorkflowRegistry
from workflows.video.registry import VideoWorkflowRegistry


VALID_API_KEY = "sk-factory-m3-it2-challenger2-valid"
VALID_JWT_SECRET = "jwt-secret-m3-it2-challenger2-test-32b"


@pytest.fixture(autouse=True)
def setup_clean_auth_state():
    """Ensure apps have clean, predictable auth configurations for testing."""
    auth_config = AuthConfig(api_key=VALID_API_KEY, jwt_secret=VALID_JWT_SECRET, environment="test")
    speech_app.state.auth_config = auth_config
    music_app.state.auth_config = auth_config
    enhancement_app.state.auth_config = auth_config
    image_gateway_app.state.auth_config = auth_config
    video_gateway_app.state.auth_config = auth_config

    orig_image_cfg_key = getattr(image_gateway_mod.config, "api_key", None)
    orig_video_cfg_key = getattr(video_gateway_mod.config, "api_key", None)
    image_gateway_mod.config.api_key = VALID_API_KEY
    video_gateway_mod.config.api_key = VALID_API_KEY

    yield

    image_gateway_mod.config.api_key = orig_image_cfg_key
    video_gateway_mod.config.api_key = orig_video_cfg_key
    for app in [speech_app, music_app, enhancement_app, image_gateway_app, video_gateway_app]:
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")


# ==============================================================================
# SECTION 1: Speech WebSocket Security Perimeter & Disconnect Resilience
# ==============================================================================

class TestSpeechWebSocketAdversarialSecurity:
    """Probes /v1/audio/transcriptions/ws for unauthorized ingress and abrupt disconnects."""

    def test_anonymous_websocket_connection_rejected_with_1008(self):
        """Probe WebSocket without credentials -> must be rejected with close code 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws"):
                    pass
            assert exc_info.value.code == 1008

    def test_empty_token_query_rejected_with_1008(self):
        """Probe WebSocket with ?token= -> must be rejected with 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws?token="):
                    pass
            assert exc_info.value.code == 1008

    def test_invalid_bearer_token_query_rejected_with_1008(self):
        """Probe WebSocket with ?token=wrong-secret -> must be rejected with 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws?token=hacker-evil-token"):
                    pass
            assert exc_info.value.code == 1008

    def test_invalid_authorization_header_rejected_with_1008(self):
        """Probe WebSocket with Authorization: Bearer bad-token -> rejected with 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/v1/audio/transcriptions/ws",
                    headers={"Authorization": "Bearer bad-token-xyz"},
                ):
                    pass
            assert exc_info.value.code == 1008

    def test_empty_authorization_header_rejected_with_1008(self):
        """Probe WebSocket with Authorization: Bearer -> rejected with 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/v1/audio/transcriptions/ws",
                    headers={"Authorization": "Bearer "},
                ):
                    pass
            assert exc_info.value.code == 1008

    def test_malformed_jwt_token_rejected_with_1008(self):
        """Probe WebSocket with malformed 3-part JWT token -> rejected with 1008."""
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws?token=abc.def.ghi"):
                    pass
            assert exc_info.value.code == 1008

    def test_valid_token_query_connects_and_streams(self):
        """Valid token query param successfully authenticates and streams interim & final events."""
        with TestClient(speech_app) as client:
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={VALID_API_KEY}") as ws:
                ws.send_json({"chunk_id": 1})
                resp = ws.receive_json()
                assert resp.get("event") == "interim"
                assert resp.get("chunk_id") == 1

                ws.send_json({"action": "end"})
                final_resp = ws.receive_json()
                assert final_resp.get("event") == "final"
                assert "text" in final_resp

    def test_valid_authorization_header_connects_and_streams(self):
        """Valid Authorization: Bearer header successfully authenticates."""
        with TestClient(speech_app) as client:
            with client.websocket_connect(
                "/v1/audio/transcriptions/ws",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            ) as ws:
                ws.send_json({"chunk_id": 1})
                resp = ws.receive_json()
                assert resp.get("event") == "interim"
                ws.send_json({"action": "end"})
                final_resp = ws.receive_json()
                assert final_resp.get("event") == "final"

    def test_abrupt_disconnect_mid_stream_causes_zero_crashes(self):
        """
        Adversarially probe abrupt client disconnect mid-stream:
        Client opens session, sends chunk, and terminates abruptly without sending action: end.
        Verifies zero unhandled RuntimeError or WebSocketDisconnected exceptions escape.
        Verifies immediate fresh reconnection succeeds on same server.
        """
        with TestClient(speech_app) as client:
            # Session 1: abrupt disconnect mid-stream
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={VALID_API_KEY}") as ws1:
                ws1.send_json({"chunk_id": 1})
                resp = ws1.receive_json()
                assert resp.get("event") == "interim"
                # Exiting context without sending action: end triggers client disconnect

            # Session 2: Immediate follow-up connection must succeed normally (zero server corruption)
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={VALID_API_KEY}") as ws2:
                ws2.send_json({"chunk_id": 100})
                resp2 = ws2.receive_json()
                assert resp2.get("event") == "interim"
                assert resp2.get("chunk_id") == 100
                ws2.send_json({"action": "end"})
                final2 = ws2.receive_json()
                assert final2.get("event") == "final"

    def test_binary_audio_chunk_streaming_and_disconnect(self):
        """Stream raw binary PCM audio bytes over WebSocket, then abruptly drop."""
        from unittest.mock import patch
        with patch("services.speech.server.streaming_buffer.engine.transcribe", return_value={"text": "interim transcript", "segments": []}):
            with TestClient(speech_app) as client:
                with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={VALID_API_KEY}") as ws:
                    ws.send_bytes(b"\x00\x01\x02\x03" * 256)
                    resp = ws.receive_json()
                    assert resp.get("event") == "interim"
                    assert resp.get("chunk_id") == 1
                    # Abrupt disconnect with binary frames in buffer


# ==============================================================================
# SECTION 2: Image & Video Gateways /ws Auth Bypass Defense
# ==============================================================================

class TestGatewayWebSocketAdversarialAuth:
    """Probes ComfyUI Image and Video edge gateways for auth bypass vulnerabilities."""

    def test_image_gateway_empty_api_key_blocks_arbitrary_tokens(self):
        """
        Adversarial scenario: gateway configured with FACTORY_API_KEY="" or None.
        Random client attempts to connect with ?token=anything.
        Must strictly reject with close code 1008 (zero bypass).
        """
        image_gateway_mod.config.api_key = ""
        with TestClient(image_gateway_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws?token=random-attacker-token"):
                    pass
            assert exc_info.value.code == 1008

    def test_video_gateway_empty_api_key_blocks_arbitrary_tokens(self):
        """Video gateway with FACTORY_API_KEY="" must strictly reject with code 1008."""
        video_gateway_mod.config.api_key = ""
        with TestClient(video_gateway_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws?token=random-attacker-token"):
                    pass
            assert exc_info.value.code == 1008

    def test_image_gateway_rejects_missing_and_invalid_tokens(self):
        """Image gateway rejects anonymous and wrong tokens with code 1008."""
        image_gateway_mod.config.api_key = VALID_API_KEY
        with TestClient(image_gateway_app) as client:
            # Anonymous
            with pytest.raises(WebSocketDisconnect) as exc1:
                with client.websocket_connect("/ws"):
                    pass
            assert exc1.value.code == 1008

            # Wrong token query
            with pytest.raises(WebSocketDisconnect) as exc2:
                with client.websocket_connect("/ws?token=wrong-token"):
                    pass
            assert exc2.value.code == 1008

            # Wrong token header
            with pytest.raises(WebSocketDisconnect) as exc3:
                with client.websocket_connect("/ws", headers={"Authorization": "Bearer wrong-token"}):
                    pass
            assert exc3.value.code == 1008

    def test_video_gateway_rejects_missing_and_invalid_tokens(self):
        """Video gateway rejects anonymous and wrong tokens with code 1008."""
        video_gateway_mod.config.api_key = VALID_API_KEY
        with TestClient(video_gateway_app) as client:
            # Anonymous
            with pytest.raises(WebSocketDisconnect) as exc1:
                with client.websocket_connect("/ws"):
                    pass
            assert exc1.value.code == 1008

            # Wrong token query
            with pytest.raises(WebSocketDisconnect) as exc2:
                with client.websocket_connect("/ws?token=wrong-token"):
                    pass
            assert exc2.value.code == 1008

            # Wrong token header
            with pytest.raises(WebSocketDisconnect) as exc3:
                with client.websocket_connect("/ws", headers={"Authorization": "Bearer wrong-token"}):
                    pass
            assert exc3.value.code == 1008


# ==============================================================================
# SECTION 3: SSRF Protection Across Subsystems (Speech, Music, Enhancement)
# ==============================================================================

class TestSSRFDefenseAcrossSubsystems:
    """Probes external URL ingestion parameters for SSRF vulnerabilities."""

    HEADERS = {"Authorization": f"Bearer {VALID_API_KEY}", "Content-Type": "application/json"}

    MALICIOUS_SSRF_URLS = [
        # Loopback
        "http://127.0.0.1/audio.wav",
        "http://127.0.0.1:8080/secret",
        "http://localhost/bad.wav",
        "http://localhost:5000/internal",
        "http://[::1]/leak",
        "http://0.0.0.0:8000/admin",
        # RFC1918 Private ranges
        "http://10.0.0.1/internal.wav",
        "http://10.254.1.99/secret.png",
        "http://192.168.1.1/router.mp4",
        "http://192.168.0.254/config",
        "http://172.16.0.1/payroll.wav",
        "http://172.31.255.255/db.mp4",
        # Cloud Metadata endpoints
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ]

    def test_speech_transcription_ssrf_blocked(self):
        """Probe Speech /v1/audio/transcriptions with SSRF targets -> 400 SSRF_BLOCKED."""
        with TestClient(speech_app) as client:
            for bad_url in self.MALICIOUS_SSRF_URLS:
                res = client.post(
                    "/v1/audio/transcriptions",
                    headers=self.HEADERS,
                    json={"model": "qwen3-asr", "audio_url": bad_url},
                )
                assert res.status_code == 400, f"Expected 400 for {bad_url}, got {res.status_code}"
                body = res.json()
                assert body["error"]["code"] == "SSRF_BLOCKED", f"Failed for {bad_url}: {body}"
                assert "Traceback" not in body["error"]["message"]

    def test_speech_forced_alignment_ssrf_blocked(self):
        """Probe Speech /v1/audio/alignments with SSRF targets -> 400 SSRF_BLOCKED."""
        with TestClient(speech_app) as client:
            for bad_url in ["http://127.0.0.1/align.wav", "http://169.254.169.254/cred", "http://10.0.0.5/p.wav"]:
                res = client.post(
                    "/v1/audio/alignments",
                    headers=self.HEADERS,
                    json={"audio_url": bad_url, "transcript_text": "hello"},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_speech_synthesis_reference_audio_ssrf_blocked(self):
        """Probe Speech /v1/audio/speech with SSRF reference_audio_url -> 400 SSRF_BLOCKED."""
        with TestClient(speech_app) as client:
            for bad_url in ["http://localhost/ref.wav", "http://192.168.1.1/ref.wav", "http://169.254.169.254/ref"]:
                res = client.post(
                    "/v1/audio/speech",
                    headers=self.HEADERS,
                    json={"input": "test speech", "voice": "female_1", "reference_audio_url": bad_url},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_music_audio_to_audio_ssrf_blocked(self):
        """Probe Music /v1/audio_to_audio with SSRF targets -> 400 SSRF_BLOCKED or 410."""
        with TestClient(music_app) as client:
            for bad_url in self.MALICIOUS_SSRF_URLS:
                res = client.post(
                    "/v1/audio_to_audio",
                    headers=self.HEADERS,
                    json={"prompt": "remix", "audio_url": bad_url},
                )
                assert res.status_code in (400, 410)
                if res.status_code == 400:
                    assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_enhancement_image_sr_ssrf_blocked(self):
        """Probe Enhancement /v1/image_super_resolution with SSRF image_url -> 400 SSRF_BLOCKED."""
        with TestClient(enhancement_app) as client:
            for bad_url in ["http://127.0.0.1/i.png", "http://169.254.169.254/meta", "http://10.1.2.3/sr.png"]:
                res = client.post(
                    "/v1/image_super_resolution",
                    headers=self.HEADERS,
                    json={"image_url": bad_url, "scale_factor": 2},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_enhancement_video_sr_ssrf_blocked(self):
        """Probe Enhancement /v1/video_super_resolution with SSRF video_url -> 400 SSRF_BLOCKED."""
        with TestClient(enhancement_app) as client:
            for bad_url in ["http://localhost/v.mp4", "http://192.168.1.50/v.mp4"]:
                res = client.post(
                    "/v1/video_super_resolution",
                    headers=self.HEADERS,
                    json={"video_url": bad_url, "scale_factor": 2},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_enhancement_frame_interpolation_ssrf_blocked(self):
        """Probe Enhancement /v1/frame_interpolation with SSRF video_url -> 400 SSRF_BLOCKED."""
        with TestClient(enhancement_app) as client:
            for bad_url in ["http://127.0.0.1/interp.mp4", "http://172.16.0.10/v.mp4"]:
                res = client.post(
                    "/v1/frame_interpolation",
                    headers=self.HEADERS,
                    json={"video_url": bad_url, "multiplier": 2},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_enhancement_audio_enhancement_ssrf_blocked(self):
        """Probe Enhancement /v1/audio_enhancement with SSRF audio_url -> 400 SSRF_BLOCKED."""
        with TestClient(enhancement_app) as client:
            for bad_url in ["http://127.0.0.1/noise.wav", "http://169.254.169.254/auth"]:
                res = client.post(
                    "/v1/audio_enhancement",
                    headers=self.HEADERS,
                    json={"audio_url": bad_url, "mode": "denoise"},
                )
                assert res.status_code == 400
                assert res.json()["error"]["code"] == "SSRF_BLOCKED"


# ==============================================================================
# SECTION 4: Error Sanitization, X-Request-ID, and Zero Exception String Leaks
# ==============================================================================

class TestErrorSanitizationAndTraceability:
    """Verifies X-Request-ID propagation, unified error schemas, and zero raw exception leaks."""

    def test_x_request_id_custom_propagation_and_echo(self):
        """Client-provided X-Request-ID header must be preserved in response header and error payload."""
        custom_id = f"req-{uuid.uuid4()}"
        with TestClient(speech_app) as client:
            # Trigger 401 unauthorized with custom request ID
            res = client.post(
                "/v1/audio/transcriptions",
                headers={"X-Request-ID": custom_id, "Authorization": "Bearer bad-token"},
                json={"audio_url": "https://s3.ai.example.com/audio.wav"},
            )
            assert res.status_code == 401
            assert res.headers.get("X-Request-ID") == custom_id
            body = res.json()
            assert body["error"]["request_id"] == custom_id

    def test_x_request_id_generated_when_omitted(self):
        """When client omits X-Request-ID, server must generate req-... and synchronize header and body."""
        with TestClient(speech_app) as client:
            res = client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer bad-token"},
                json={"audio_url": "https://s3.ai.example.com/audio.wav"},
            )
            assert res.status_code == 401
            header_id = res.headers.get("X-Request-ID")
            assert header_id and header_id.startswith("req-")
            body = res.json()
            assert body["error"]["request_id"] == header_id

    def test_gateway_sanitizes_unreachable_comfyui_without_str_e_leaks(self):
        """
        Probe Image and Video Gateway proxy routes when internal ComfyUI is offline.
        Must return sanitized fallback responses without leaking raw str(e) exception strings.
        """
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        # Point to unreachable port
        image_gateway_mod.config.internal_comfy_port = 59999
        video_gateway_mod.config.internal_comfy_port = 59999

        with TestClient(image_gateway_app) as client:
            # POST /interrupt fallback returns 502 fail-closed
            res = client.post("/interrupt", headers=headers)
            assert res.status_code in (502, 503)
            data = res.json()
            # Check zero raw python connection exceptions
            assert "ConnectError" not in str(data)
            assert "Errno" not in str(data)

            # GET /system_stats fallback returns 502 fail-closed
            res_stats = client.get("/system_stats", headers=headers)
            assert res_stats.status_code in (502, 503)
            stats = res_stats.json()
            assert "ConnectError" not in str(stats)
            assert "Errno" not in str(stats)

        with TestClient(video_gateway_app) as client:
            res_vid = client.post("/interrupt", headers=headers)
            assert res_vid.status_code in (502, 503)
            data_vid = res_vid.json()
            assert "ConnectError" not in str(data_vid)
            assert "Errno" not in str(data_vid)

    def test_zero_raw_filesystem_paths_or_tracebacks_in_error_responses(self):
        """Ensure no responses expose absolute filesystem paths or Python tracebacks."""
        with TestClient(speech_app) as client:
            # Trigger validation error (422)
            res = client.post(
                "/v1/audio/alignments",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"},
                json={"audio_url": 12345},  # Invalid type
            )
            text = res.text
            assert "Traceback (most recent call last)" not in text
            assert "/Volumes/WorkSpace" not in text
            assert "/Users/" not in text
            assert "site-packages" not in text


# ==============================================================================
# SECTION 5: Workflow Registries Deep Copy Isolation & Task Aliases
# ==============================================================================

class TestWorkflowRegistriesIsolationAndAliases:
    """Probes ImageWorkflowRegistry and VideoWorkflowRegistry for cache mutation leaks."""

    def test_image_registry_deep_copy_isolation_against_mutation(self):
        """
        Mutating returned workflow dict from ImageWorkflowRegistry must NEVER corrupt the cached graph.
        """
        registry = ImageWorkflowRegistry()
        wf1 = registry.get_workflow("text_to_image")
        original_node_count = len(wf1)
        original_prompt_node = copy.deepcopy(wf1.get("6", {}))

        # Perform malicious in-place mutations
        wf1["MALICIOUS_INJECTED_KEY"] = "ATTACK_PAYLOAD"
        if "6" in wf1:
            wf1["6"]["inputs"]["text"] = "MUTATED_PROMPT_POISON"
            wf1["6"]["NEW_FIELD"] = 999
        wf1["COMPLETELY_NEW_NODE"] = {"class_type": "ExploitNode", "inputs": {}}

        # Re-fetch from registry
        wf2 = registry.get_workflow("text_to_image")

        # Verify pristine isolation
        assert "MALICIOUS_INJECTED_KEY" not in wf2, "Registry cache was mutated by caller!"
        assert "COMPLETELY_NEW_NODE" not in wf2, "Registry cache retained injected node!"
        assert len(wf2) == original_node_count, "Node count was altered in cache!"
        if "6" in wf2:
            assert wf2["6"] == original_prompt_node, "Internal node inputs were mutated in cache!"

    def test_image_registry_task_aliases(self):
        """Verify task aliases (t2i, i2i, edit, reference) map correctly."""
        registry = ImageWorkflowRegistry()
        for alias, canonical in [
            ("t2i", "text_to_image"),
            ("i2i", "image_to_image"),
            ("edit", "instruction_edit"),
            ("reference", "reference_conditioning"),
        ]:
            wf_alias = registry.get_workflow(alias)
            wf_canon = registry.get_workflow(canonical)
            assert wf_alias == wf_canon, f"Alias '{alias}' did not match canonical '{canonical}'"

    def test_video_registry_deep_copy_isolation_against_mutation(self):
        """
        Mutating returned workflow dict from VideoWorkflowRegistry must NEVER corrupt the cached graph.
        """
        registry = VideoWorkflowRegistry()
        wf1 = registry.get_workflow("wan-2.2", "text_to_video")
        original_count = len(wf1)
        original_sampler = copy.deepcopy(wf1.get("3", {}))

        # Perform aggressive mutations
        wf1["POISON_TAG"] = 0xDEADBEEF
        if "3" in wf1:
            wf1["3"]["inputs"]["steps"] = -999
            del wf1["3"]

        # Re-fetch
        wf2 = registry.get_workflow("wan-2.2", "text_to_video")
        assert "POISON_TAG" not in wf2, "Video registry cache was poisoned!"
        assert len(wf2) == original_count
        if "3" in wf2:
            assert wf2["3"] == original_sampler

    def test_video_registry_task_aliases_and_unified_access(self):
        """Verify video task aliases and unified single-arg / two-arg get_workflow."""
        registry = VideoWorkflowRegistry()

        # Single-arg invocation defaults to wan-2.2
        wf_single = registry.get_workflow("t2v")
        wf_explicit = registry.get_workflow("wan-2.2", "text_to_video")
        assert wf_single == wf_explicit

        # Aliases
        for alias, canonical in [
            ("t2v", "text_to_video"),
            ("i2v", "image_to_video"),
            ("fl2v", "first_last_frame_to_video"),
            ("extension", "video_extension"),
        ]:
            assert registry.get_workflow("wan-2.2", alias) == registry.get_workflow("wan-2.2", canonical)

        # MiniMax models
        assert "9" in registry.get_workflow("minimax-h3", "t2v")
        assert "9" in registry.get_workflow("minimax-h3", "with_audio")
