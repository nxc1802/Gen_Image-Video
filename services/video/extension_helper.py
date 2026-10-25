"""
services/video/extension_helper.py
Video frame extraction and clip concatenation utilities for Video Extension & Chaining.
"""
import logging
import os
from pathlib import Path
import subprocess

logger = logging.getLogger("services.video.extension_helper")


class VideoExtensionHelper:
    """Helper utilities for extracting anchor keyframes and stitching extension clips."""

    @staticmethod
    def extract_anchor_frame(source_video_path: Path, output_frame_path: Path) -> Path:
        """
        Extracts the final frame of source_video_path to output_frame_path using ffmpeg.
        Raises RuntimeError if ffmpeg is unavailable, extraction fails, or source video is corrupt.
        """
        output_frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_video_path.exists():
            raise RuntimeError(f"Source video does not exist: {source_video_path}")
        if source_video_path.stat().st_size <= 100:
            raise RuntimeError(
                f"Source video is corrupt or too small ({source_video_path.stat().st_size} bytes): {source_video_path}"
            )

        cmd = [
            "ffmpeg",
            "-sseof",
            "-1",
            "-i",
            str(source_video_path),
            "-update",
            "1",
            "-q:v",
            "1",
            "-y",
            str(output_frame_path),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10.0)
            if res.returncode == 0 and output_frame_path.exists() and output_frame_path.stat().st_size > 0:
                return output_frame_path
            logger.warning(
                f"ffmpeg frame extraction failed with code {res.returncode}: {res.stderr.decode('utf-8', errors='replace')}"
            )
        except Exception as e:
            logger.warning(f"ffmpeg frame extraction failed: {e}")

        raise RuntimeError(
            f"Failed to extract video frame from {source_video_path}: ffmpeg extraction failed or video is corrupt."
        )

    @staticmethod
    def concatenate_videos(source_video_path: Path, extension_video_path: Path, output_video_path: Path) -> Path:
        """
        Concatenates source and extension video clips into output_video_path using ffmpeg.
        Raises RuntimeError if source or extension video is missing or ffmpeg fails.
        """
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_video_path.exists():
            raise RuntimeError(f"Source video does not exist: {source_video_path}")
        if not extension_video_path.exists():
            raise RuntimeError(f"Extension video does not exist: {extension_video_path}")

        concat_list = output_video_path.parent / f"concat_{os.getpid()}.txt"
        concat_list.write_text(f"file '{source_video_path.resolve()}'\nfile '{extension_video_path.resolve()}'\n")
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-y",
            str(output_video_path),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15.0)
            if res.returncode == 0 and output_video_path.exists() and output_video_path.stat().st_size > 0:
                return output_video_path
            logger.warning(
                f"ffmpeg concatenation failed with code {res.returncode}: {res.stderr.decode('utf-8', errors='replace')}"
            )
        except Exception as e:
            logger.warning(f"ffmpeg concatenation failed: {e}")
        finally:
            try:
                concat_list.unlink(missing_ok=True)
            except Exception:
                pass

        raise RuntimeError(
            f"Failed to concatenate videos {source_video_path} and {extension_video_path}: ffmpeg concatenation failed."
        )
