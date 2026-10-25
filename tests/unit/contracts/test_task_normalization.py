"""
tests/unit/contracts/test_task_normalization.py
Unit tests verifying canonical task normalization and capability ID mappings across
contracts, registries, and job lifecycle manager.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from factory_contracts.tasks import CANONICAL_TASK_MAP, normalize_task
from factory_contracts.jobs import JobSubmissionRequest
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from workflows.image.registry import ImageWorkflowRegistry
from workflows.video.registry import VideoWorkflowRegistry


def test_canonical_task_map_coverage():
    """Verify CANONICAL_TASK_MAP contains all required capability IDs and aliases."""
    # Image capabilities
    assert "image.t2i" in CANONICAL_TASK_MAP
    assert "image.i2i" in CANONICAL_TASK_MAP
    assert "image.inpaint" in CANONICAL_TASK_MAP
    assert "image.outpaint" in CANONICAL_TASK_MAP
    assert "image.instruction_edit" in CANONICAL_TASK_MAP
    assert "image.reference" in CANONICAL_TASK_MAP
    assert "image.controlnet" in CANONICAL_TASK_MAP

    # Video capabilities
    assert "video.t2v" in CANONICAL_TASK_MAP
    assert "video.i2v" in CANONICAL_TASK_MAP
    assert "video.fl2v" in CANONICAL_TASK_MAP
    assert "video.extend" in CANONICAL_TASK_MAP
    assert "video.audio" in CANONICAL_TASK_MAP
    assert "video.reference" in CANONICAL_TASK_MAP

    # Music & Audio capabilities
    assert "music.t2m" in CANONICAL_TASK_MAP
    assert "music.l2m" in CANONICAL_TASK_MAP
    assert "music.a2a" in CANONICAL_TASK_MAP

    # Speech capabilities
    assert "speech.tts" in CANONICAL_TASK_MAP
    assert "speech.asr" in CANONICAL_TASK_MAP

    # Enhancement capabilities
    assert "enhance.sr_image" in CANONICAL_TASK_MAP
    assert "enhance.sr_video" in CANONICAL_TASK_MAP
    assert "enhance.interpolation" in CANONICAL_TASK_MAP
    assert "enhance.audio" in CANONICAL_TASK_MAP


def test_normalize_task_image_capabilities():
    """Verify all image capability IDs and aliases resolve to canonical task names."""
    # Text to image
    assert normalize_task("image.t2i") == "text_to_image"
    assert normalize_task("t2i") == "text_to_image"
    assert normalize_task("text2image") == "text_to_image"
    assert normalize_task("text_to_image") == "text_to_image"

    # Image to image
    assert normalize_task("image.i2i") == "image_to_image"
    assert normalize_task("i2i") == "image_to_image"
    assert normalize_task("image2image") == "image_to_image"
    assert normalize_task("image_to_image") == "image_to_image"

    # Inpaint & Outpaint
    assert normalize_task("image.inpaint") == "inpaint"
    assert normalize_task("inpaint") == "inpaint"
    assert normalize_task("inpainting") == "inpaint"

    assert normalize_task("image.outpaint") == "outpaint"
    assert normalize_task("outpaint") == "outpaint"
    assert normalize_task("outpainting") == "outpaint"

    # Instruction edit
    assert normalize_task("image.instruction_edit") == "instruction_edit"
    assert normalize_task("instruction_edit") == "instruction_edit"
    assert normalize_task("edit") == "instruction_edit"
    assert normalize_task("image_edit") == "instruction_edit"

    # Reference conditioning
    assert normalize_task("image.reference") == "reference_conditioning"
    assert normalize_task("image.reference_conditioning") == "reference_conditioning"
    assert normalize_task("reference_conditioning") == "reference_conditioning"
    assert normalize_task("reference") == "reference_conditioning"
    assert normalize_task("ref2img") == "reference_conditioning"

    # ControlNet
    assert normalize_task("image.controlnet") == "controlnet"
    assert normalize_task("controlnet") == "controlnet"
    assert normalize_task("control_pose") == "controlnet"
    assert normalize_task("control_depth") == "controlnet"


def test_normalize_task_video_capabilities():
    """Verify all video capability IDs and aliases resolve to canonical task names."""
    # Text to video
    assert normalize_task("video.t2v") == "text_to_video"
    assert normalize_task("t2v") == "text_to_video"
    assert normalize_task("text2video") == "text_to_video"
    assert normalize_task("text_to_video") == "text_to_video"

    # Image to video
    assert normalize_task("video.i2v") == "image_to_video"
    assert normalize_task("i2v") == "image_to_video"
    assert normalize_task("image2video") == "image_to_video"
    assert normalize_task("image_to_video") == "image_to_video"

    # First last frame to video
    assert normalize_task("video.fl2v") == "first_last_frame_to_video"
    assert normalize_task("fl2v") == "first_last_frame_to_video"
    assert normalize_task("first_last_frame_to_video") == "first_last_frame_to_video"
    assert normalize_task("first_last_frame") == "first_last_frame_to_video"

    # Video extension
    assert normalize_task("video.extend") == "video_extension"
    assert normalize_task("video.extension") == "video_extension"
    assert normalize_task("extension") == "video_extension"
    assert normalize_task("video_extension") == "video_extension"

    # Video with audio
    assert normalize_task("video.audio") == "video_with_audio"
    assert normalize_task("video.with_audio") == "video_with_audio"
    assert normalize_task("with_audio") == "video_with_audio"
    assert normalize_task("video_with_audio") == "video_with_audio"

    # Video reference
    assert normalize_task("video.reference") == "reference_conditioning"
    assert normalize_task("video.ref2va") == "reference_conditioning"
    assert normalize_task("ref2va") == "reference_conditioning"


def test_normalize_task_music_speech_enhance():
    """Verify music, speech, and enhancement task normalization."""
    # Music
    assert normalize_task("music.t2m") == "text_to_music"
    assert normalize_task("audio.music") == "text_to_music"
    assert normalize_task("music.lyrics") == "lyrics_to_music"
    assert normalize_task("music.l2m") == "lyrics_to_music"
    assert normalize_task("music.a2a") == "audio_to_audio"

    # Speech
    assert normalize_task("speech.tts") == "speech_synthesis"
    assert normalize_task("tts") == "speech_synthesis"
    assert normalize_task("speech.asr") == "speech_transcription"
    assert normalize_task("asr") == "speech_transcription"

    # Enhancement
    assert normalize_task("enhance.sr_image") == "image_super_resolution"
    assert normalize_task("enhancement.sr") == "image_super_resolution"
    assert normalize_task("enhance.sr_video") == "video_super_resolution"
    assert normalize_task("enhance.interpolation") == "frame_interpolation"
    assert normalize_task("enhance.audio") == "audio_enhancement"


def test_normalize_task_edge_cases():
    """Verify edge cases: whitespace, casing, unknown tasks, empty values."""
    # Case insensitivity and whitespace stripping
    assert normalize_task("  IMAGE.T2I  ") == "text_to_image"
    assert normalize_task("Video.T2V") == "text_to_video"
    assert normalize_task("  Inpaint  ") == "inpaint"
    assert normalize_task("MUSIC.T2M") == "text_to_music"

    # Unknown custom tasks pass through as stripped lowercase
    assert normalize_task("custom_inference_task") == "custom_inference_task"
    assert normalize_task("  MY_CUSTOM_TASK  ") == "my_custom_task"

    # Empty and None values return empty string
    assert normalize_task("") == ""
    assert normalize_task(None) == ""


def test_image_registry_resolves_capability_ids():
    """Verify ImageWorkflowRegistry resolves capability IDs identically to canonical tasks."""
    registry = ImageWorkflowRegistry()
    assert registry.get_workflow("image.t2i") == registry.get_workflow("text_to_image")
    assert registry.get_workflow("image.i2i") == registry.get_workflow("image_to_image")
    assert registry.get_workflow("image.inpaint") == registry.get_workflow("inpaint")
    assert registry.get_workflow("image.outpaint") == registry.get_workflow("outpaint")
    assert registry.get_workflow("image.instruction_edit") == registry.get_workflow("instruction_edit")
    assert registry.get_workflow("image.reference") == registry.get_workflow("reference_conditioning")
    assert registry.get_workflow("image.reference_conditioning") == registry.get_workflow("reference_conditioning")
    assert registry.get_workflow("image.controlnet") == registry.get_workflow("controlnet")


def test_video_registry_resolves_capability_ids():
    """Verify VideoWorkflowRegistry resolves capability IDs identically to canonical tasks."""
    registry = VideoWorkflowRegistry()
    # Wan2.2
    assert registry.get_workflow("wan-2.2", "video.t2v") == registry.get_workflow("wan-2.2", "text_to_video")
    assert registry.get_workflow("wan-2.2", "video.i2v") == registry.get_workflow("wan-2.2", "image_to_video")
    assert registry.get_workflow("wan-2.2", "video.fl2v") == registry.get_workflow("wan-2.2", "first_last_frame_to_video")
    assert registry.get_workflow("wan-2.2", "video.extension") == registry.get_workflow("wan-2.2", "video_extension")
    assert registry.get_workflow("wan-2.2", "video.extend") == registry.get_workflow("wan-2.2", "video_extension")
    with pytest.raises(KeyError):
        registry.get_workflow("wan-2.2", "video.with_audio")

    # MiniMax H3
    assert registry.get_workflow("minimax-h3", "video.t2v") == registry.get_workflow("minimax-h3", "text_to_video")
    assert registry.get_workflow("minimax-h3", "video.with_audio") == registry.get_workflow("minimax-h3", "video_with_audio")
    assert registry.get_workflow("minimax-h3", "video.ref2va") == registry.get_workflow("minimax-h3", "reference_conditioning")


@pytest.mark.asyncio
async def test_job_lifecycle_manager_normalizes_task_on_submit():
    """Verify JobLifecycleManager.submit_job stores canonical task in the database."""
    mock_repo = MagicMock()
    mock_repo.get_by_idempotency_key = AsyncMock(return_value=None)
    mock_repo.create_job = AsyncMock()
    mock_repo.count_active_and_queued = AsyncMock(return_value=(0, 1))

    mock_broker = MagicMock()
    mock_broker.enqueue = AsyncMock()

    mock_stream = MagicMock()
    mock_stream.publish = AsyncMock()

    manager = JobLifecycleManager(
        repository=mock_repo,
        broker=mock_broker,
        stream_manager=mock_stream,
    )

    req = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="image.t2i",
        inputs={"prompt": "A beautiful sunset"},
    )

    job, created = await manager.submit_job(req, request_id="req-12345678-1234-1234-1234-123456789abc")
    assert created is True
    assert job.task == "text_to_image"
    mock_repo.create_job.assert_called_once()
    saved_job = mock_repo.create_job.call_args[0][0]
    assert saved_job.task == "text_to_image"
