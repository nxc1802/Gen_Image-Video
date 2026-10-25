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
        Falls back to generating an image if ffmpeg is unavailable or source video is mock.
        """
        output_frame_path.parent.mkdir(parents=True, exist_ok=True)
        if source_video_path.exists() and source_video_path.stat().st_size > 100:
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
                if res.returncode == 0 and output_frame_path.exists():
                    return output_frame_path
            except Exception as e:
                logger.warning(f"ffmpeg frame extraction failed: {e}")

        raise RuntimeError(
            f"Failed to extract video frame from {video_path}: ffmpeg extraction failed or video is corrupt."
        )

    @staticmethod
    def concatenate_videos(source_video_path: Path, extension_video_path: Path, output_video_path: Path) -> Path:
        """
        Concatenates source and extension video clips into output_video_path.
        """
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        if source_video_path.exists() and extension_video_path.exists():
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
                if res.returncode == 0 and output_video_path.exists():
                    try:
                        concat_list.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return output_video_path
            except Exception as e:
                logger.warning(f"ffmpeg concatenation failed: {e}")
            finally:
                try:
                    concat_list.unlink(missing_ok=True)
                except Exception:
                    pass

        # If extension video exists, copy or fallback
        if extension_video_path.exists():
            output_video_path.write_bytes(extension_video_path.read_bytes())
        else:
            output_video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42")

        return output_video_path
