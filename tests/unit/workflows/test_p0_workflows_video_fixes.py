"""
tests/unit/workflows/test_p0_workflows_video_fixes.py
Targeted regression tests for Phase 0 fixes:
- FIX-11: Standard vs Turbo checkpoint selection based on variant in image parameter mapping.
- FIX-12: MiniMax H3 8_ref node bound to actual reference image asset.
- FIX-13: VideoExtensionHelper concatenates original and generated segments for video extension.
"""
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.parameter_mappings import VideoParameterMapper
from services.video.extension_helper import VideoExtensionHelper


def test_fix_11_image_variant_checkpoint_selection():
    """FIX-11: Turbo variant loads Turbo weights, Standard loads Standard weights."""
    template = {
        "476": {"inputs": {"steps": 20, "seed": 0, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}},
        "478": {"inputs": {"unet_name": "placeholder.gguf"}},
    }

    # 1. Turbo variant
    wf_turbo = ImageParameterMapper.apply(
        workflow=dict(template),
        task="text_to_image",
        inputs={"prompt": "A coastline", "variant": "turbo"},
    )
    assert wf_turbo["478"]["inputs"]["unet_name"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    assert wf_turbo["476"]["inputs"]["steps"] == 8

    # 2. Standard variant
    wf_std = ImageParameterMapper.apply(
        workflow=dict(template),
        task="text_to_image",
        inputs={"prompt": "A coastline", "variant": "standard"},
    )
    assert wf_std["478"]["inputs"]["unet_name"] == "qwen-image-2.1-Q4_K_M.gguf"
    assert wf_std["476"]["inputs"]["steps"] == 20


def test_fix_12_minimax_ref2va_binds_8_ref():
    """FIX-12: MiniMax H3 ref2va binds reference image into node 8_ref."""
    workflow = {
        "8_ref": {"class_type": "LoadImage", "inputs": {"image": "default.png"}},
        "9": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"reference_image": ["8_ref", 0]}},
    }

    res = VideoParameterMapper.apply(
        workflow=workflow,
        model="minimax-h3",
        task="reference_conditioning",
        inputs={"prompt": "Cinematic motion", "reference_image_url": "https://example.com/character.png"},
        staged_files={"reference_image_url": "staged_character_123.png"},
    )
    assert res["8_ref"]["inputs"]["image"] == "staged_character_123.png"


def test_fix_13_video_extension_concatenation_order():
    """FIX-13: VideoExtensionHelper.concatenate_videos invokes ffmpeg with both clips in order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        output = tmp_path / "concat.mp4"

        # Create dummy non-empty files
        clip1.write_bytes(b"CLIP1" * 100)
        clip2.write_bytes(b"CLIP2" * 100)

        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_run.return_value = mock_res
            # Simulate output created by ffmpeg
            output.write_bytes(b"CONCAT_RESULT" * 100)

            result = VideoExtensionHelper.concatenate_videos(clip1, clip2, output)
            assert result == output
            assert mock_run.called
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-f" in cmd and "concat" in cmd
