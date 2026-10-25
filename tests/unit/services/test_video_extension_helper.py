"""
tests/unit/services/test_video_extension_helper.py
Unit and regression tests for VideoExtensionHelper ensuring strict fail-closed
behavior and zero mock file or fake byte fallbacks.
"""
from pathlib import Path
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from services.video.extension_helper import VideoExtensionHelper


class TestVideoExtensionHelper:
    def test_extract_anchor_frame_missing_source_raises_runtime_error(self, tmp_path: Path):
        source = tmp_path / "non_existent.mp4"
        output = tmp_path / "anchor.png"

        with pytest.raises(RuntimeError) as exc_info:
            VideoExtensionHelper.extract_anchor_frame(source, output)

        assert "Source video does not exist" in str(exc_info.value)
        assert not output.exists()

    def test_extract_anchor_frame_corrupt_or_empty_source_raises_runtime_error(self, tmp_path: Path):
        source = tmp_path / "corrupt.mp4"
        source.write_bytes(b"too small")
        output = tmp_path / "anchor.png"

        with pytest.raises(RuntimeError) as exc_info:
            VideoExtensionHelper.extract_anchor_frame(source, output)

        assert "Source video is corrupt or too small" in str(exc_info.value)
        assert not output.exists()

    def test_extract_anchor_frame_ffmpeg_failure_raises_runtime_error(self, tmp_path: Path):
        source = tmp_path / "valid.mp4"
        source.write_bytes(b"0" * 200)
        output = tmp_path / "anchor.png"

        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = b"ffmpeg decoding error"

        with patch("subprocess.run", return_value=mock_res):
            with pytest.raises(RuntimeError) as exc_info:
                VideoExtensionHelper.extract_anchor_frame(source, output)

        assert "Failed to extract video frame from" in str(exc_info.value)
        assert str(source) in str(exc_info.value)
        assert not output.exists()

    def test_extract_anchor_frame_success(self, tmp_path: Path):
        source = tmp_path / "valid.mp4"
        source.write_bytes(b"0" * 200)
        output = tmp_path / "anchor.png"

        def fake_run(*args, **kwargs):
            output.write_bytes(b"\x89PNG\r\n\x1a\nfake-real-png-bytes")
            res = MagicMock()
            res.returncode = 0
            return res

        with patch("subprocess.run", side_effect=fake_run):
            res_path = VideoExtensionHelper.extract_anchor_frame(source, output)

        assert res_path == output
        assert output.exists()

    def test_concatenate_videos_missing_source_raises_runtime_error(self, tmp_path: Path):
        source = tmp_path / "missing_source.mp4"
        extension = tmp_path / "extension.mp4"
        extension.write_bytes(b"0" * 200)
        output = tmp_path / "concatenated.mp4"

        with pytest.raises(RuntimeError) as exc_info:
            VideoExtensionHelper.concatenate_videos(source, extension, output)

        assert "Source video does not exist" in str(exc_info.value)
        assert not output.exists()

    def test_concatenate_videos_missing_extension_raises_runtime_error(self, tmp_path: Path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"0" * 200)
        extension = tmp_path / "missing_extension.mp4"
        output = tmp_path / "concatenated.mp4"

        with pytest.raises(RuntimeError) as exc_info:
            VideoExtensionHelper.concatenate_videos(source, extension, output)

        assert "Extension video does not exist" in str(exc_info.value)
        assert not output.exists()

    def test_concatenate_videos_ffmpeg_failure_raises_runtime_error_no_mock_bytes(self, tmp_path: Path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"0" * 200)
        extension = tmp_path / "extension.mp4"
        extension.write_bytes(b"1" * 200)
        output = tmp_path / "concatenated.mp4"

        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = b"ffmpeg concat stream mismatch"

        with patch("subprocess.run", return_value=mock_res):
            with pytest.raises(RuntimeError) as exc_info:
                VideoExtensionHelper.concatenate_videos(source, extension, output)

        assert "Failed to concatenate videos" in str(exc_info.value)
        # Verify no fake MP4 container was created
        assert not output.exists()

    def test_concatenate_videos_success(self, tmp_path: Path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"0" * 200)
        extension = tmp_path / "extension.mp4"
        extension.write_bytes(b"1" * 200)
        output = tmp_path / "concatenated.mp4"

        def fake_run(*args, **kwargs):
            output.write_bytes(b"valid_concatenated_mp4_stream")
            res = MagicMock()
            res.returncode = 0
            return res

        with patch("subprocess.run", side_effect=fake_run):
            res_path = VideoExtensionHelper.concatenate_videos(source, extension, output)

        assert res_path == output
        assert output.exists()
        assert output.read_bytes() == b"valid_concatenated_mp4_stream"
