"""
benchmarks.validators
Robust multi-modal artifact validation engine for Factory v4 Lite benchmarks.

Verifies:
1. Binary container magic headers (PNG, JPEG, WebP, MP4, WebM, WAV, MP3, FLAC).
2. MIME types and container formats.
3. Non-empty payload bytes and minimum byte thresholds.
4. Cryptographic SHA-256 integrity checksums.
5. Exact output dimensions (width x height) for image/video and durations for audio/video.
6. Domain schemas for structured text, decisions, embeddings, reranking, and ABC scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import wave

# Known Magic Header Signatures
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
RIFF_MAGIC = b"RIFF"
WEBP_SUBTYPE = b"WEBP"
WAVE_SUBTYPE = b"WAVE"
WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
FLAC_MAGIC = b"fLaC"
MP3_ID3_MAGIC = b"ID3"
MP3_SYNC_WORDS = (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"\xff\xf2")


@dataclass
class ValidationResult:
    valid: bool
    error_code: Optional[str] = None
    mime_type: Optional[str] = None
    format: Optional[str] = None
    size_bytes: int = 0
    sha256: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None  # (width, height)
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def detect_binary_container(data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Detects (mime_type, format) from binary magic header.
    Returns (None, None) if not recognized.
    """
    if len(data) < 4:
        return None, None

    # PNG
    if data.startswith(PNG_MAGIC):
        return "image/png", "png"

    # JPEG
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg", "jpeg"

    # RIFF containers (WebP or WAV)
    if data.startswith(RIFF_MAGIC) and len(data) >= 12:
        subtype = data[8:12]
        if subtype == WEBP_SUBTYPE:
            return "image/webp", "webp"
        if subtype == WAVE_SUBTYPE:
            return "audio/wav", "wav"

    # MP4 / QuickTime (ftyp box in bytes 4..32)
    if len(data) >= 12:
        box_size = struct.unpack(">I", data[0:4])[0]
        box_type = data[4:8]
        if box_type == b"ftyp":
            return "video/mp4", "mp4"
        # Check first 64 bytes for ftyp
        if b"ftyp" in data[:64]:
            return "video/mp4", "mp4"

    # WebM / Matroska
    if data.startswith(WEBM_MAGIC):
        return "video/webm", "webm"

    # FLAC
    if data.startswith(FLAC_MAGIC):
        return "audio/flac", "flac"

    # MP3
    if data.startswith(MP3_ID3_MAGIC) or any(data.startswith(s) for s in MP3_SYNC_WORDS):
        return "audio/mpeg", "mp3"

    return None, None


def extract_image_metadata(data: bytes, format_hint: Optional[str] = None) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Decodes image and extracts dimensions and metadata using Pillow.
    Pass 1: img.verify() checks container markers and chunk structure.
    Pass 2: Reopen stream and call img.load() to force full decompression of all raster scanlines.
    Detects truncated JPEGs and corrupted WebP bitstreams.
    """
    meta: Dict[str, Any] = {}
    try:
        from PIL import Image
        # Pass 1: Container verify
        bio = io.BytesIO(data)
        with Image.open(bio) as img:
            img.verify()

        # Pass 2: Full pixel raster decompression
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            if format_hint and fmt and fmt.upper() != format_hint.upper():
                meta["error_code"] = "IMAGE_FORMAT_MISMATCH"
                return None, meta

            # Force complete decompression of all raster scanlines
            img.load()

            w, h = img.size
            if w <= 0 or h <= 0:
                meta["error_code"] = "IMAGE_DIMENSION_OUT_OF_BOUNDS"
                return None, meta

            meta.update({
                "format": fmt,
                "mode": img.mode,
                "width": w,
                "height": h,
            })
            return (w, h), meta
    except OSError as e:
        err_msg = str(e).lower()
        if "truncated" in err_msg or "failed to read next frame" in err_msg:
            meta["error_code"] = "IMAGE_TRUNCATED"
        else:
            meta["error_code"] = "IMAGE_DECODE_FAILED"
    except Exception:
        meta["error_code"] = "IMAGE_DECODE_FAILED"
    return None, meta


def extract_png_metadata(data: bytes) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Parses and verifies PNG image container using Pillow decoding verification.
    Extracts dimensions, bit depth, and color type.
    """
    if len(data) < 29 or not data.startswith(PNG_MAGIC):
        return None, {"error_code": "IMAGE_CORRUPTED"}

    # Strict decodability verification via Pillow
    dims, meta = extract_image_metadata(data, format_hint="PNG")
    if dims is None:
        # Fallback only if Pillow is not available
        try:
            import PIL  # noqa: F401
            # If PIL is installed and failed verification, the image is corrupt
            return None, meta
        except ImportError:
            pass

        try:
            ihdr_type = data[12:16]
            if ihdr_type == b"IHDR":
                width, height = struct.unpack(">II", data[16:24])
                if width <= 0 or height <= 0:
                    return None, {"error_code": "IMAGE_DIMENSION_OUT_OF_BOUNDS"}
                bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                    ">BBBBB", data[24:29]
                )
                return (width, height), {
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                    "compression": compression,
                    "interlace": interlace,
                }
        except Exception:
            return None, {"error_code": "IMAGE_CORRUPTED"}

    # Augment with PNG-specific chunk attributes if available
    try:
        ihdr_type = data[12:16]
        if ihdr_type == b"IHDR" and len(data) >= 29:
            bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">BBBBB", data[24:29]
            )
            meta.update({
                "bit_depth": bit_depth,
                "color_type": color_type,
                "compression": compression,
                "interlace": interlace,
            })
    except Exception:
        pass

    return dims, meta


def extract_wav_metadata(data: bytes, check_silence: bool = False) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Parses WAV audio using stdlib wave to extract duration, format details,
    and detects digital silence via PCM sample RMS amplitude.
    """
    meta: Dict[str, Any] = {}
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if framerate <= 0 or channels <= 0 or sample_width <= 0:
                meta["error_code"] = "AUDIO_CONTAINER_CORRUPT"
                return None, meta
            if n_frames == 0:
                meta["error_code"] = "AUDIO_ZERO_SAMPLES"
                return None, meta

            duration = round(n_frames / float(framerate), 3)

            # Read sample frames to detect digital silence
            raw_frames = wf.readframes(n_frames)
            rms = 0.0
            max_amp = 0.0
            if raw_frames:
                total_samples = n_frames * channels
                if sample_width == 2:
                    sample_count = min(total_samples, len(raw_frames) // 2)
                    samples = struct.unpack(f"<{sample_count}h", raw_frames[:sample_count * 2])
                    max_amp = max(abs(s) for s in samples) / 32768.0
                    sum_sq = sum(float(s * s) for s in samples)
                    rms = math.sqrt(sum_sq / sample_count) / 32768.0
                elif sample_width == 1:
                    sample_count = min(total_samples, len(raw_frames))
                    samples = [(b - 128) / 128.0 for b in raw_frames[:sample_count]]
                    max_amp = max(abs(s) for s in samples)
                    sum_sq = sum(s * s for s in samples)
                    rms = math.sqrt(sum_sq / sample_count)
                elif sample_width in (3, 4):
                    max_amp = 1.0 if any(b != 0 for b in raw_frames) else 0.0
                    rms = max_amp

            meta.update({
                "channels": channels,
                "sample_width_bytes": sample_width,
                "framerate_hz": framerate,
                "n_frames": n_frames,
                "rms_amplitude": round(rms, 6),
                "peak_amplitude": round(max_amp, 6),
            })

            if rms < 1e-4 and max_amp == 0.0:
                meta["error_code"] = "AUDIO_SILENT"
                meta["is_silent"] = True
                if check_silence:
                    return None, meta

            return duration, meta
    except Exception:
        meta["error_code"] = "AUDIO_CONTAINER_CORRUPT"
    return None, meta


def extract_mp4_metadata(data: bytes, require_mdat: bool = False) -> Tuple[Optional[float], Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Parses MP4 / ISO BMFF container to validate box offsets, atom containment,
    ftyp brand, mvhd duration, tkhd track dimensions, and mdat media payload.
    """
    duration: Optional[float] = None
    dimensions: Optional[Tuple[int, int]] = None
    meta: Dict[str, Any] = {}

    try:
        offset = 0
        total_len = len(data)
        found_ftyp = False
        found_moov = False
        found_mdat = False
        mdat_size = 0

        while offset + 8 <= total_len:
            box_size = struct.unpack(">I", data[offset:offset+4])[0]
            box_type = data[offset+4:offset+8]

            if box_size == 1:
                # 64-bit box size
                if offset + 16 > total_len:
                    meta["error_code"] = "VIDEO_CONTAINER_CORRUPT"
                    return None, None, meta
                box_size = struct.unpack(">Q", data[offset+8:offset+16])[0]
                payload_offset = offset + 16
            elif box_size == 0:
                box_size = total_len - offset
                payload_offset = offset + 8
            else:
                payload_offset = offset + 8

            if box_size < 8:
                meta["error_code"] = "VIDEO_CONTAINER_CORRUPT"
                return None, None, meta

            # Atom containment check: box boundary must stay within total_len
            if offset + box_size > total_len:
                meta["error_code"] = "VIDEO_CONTAINER_CORRUPT"
                return None, None, meta

            if box_type == b"ftyp":
                found_ftyp = True
                major_brand = data[payload_offset:payload_offset+4].decode("ascii", errors="ignore")
                meta["major_brand"] = major_brand

            elif box_type == b"mdat":
                found_mdat = True
                mdat_size = box_size - (payload_offset - offset)
                meta["mdat_size"] = mdat_size

            elif box_type == b"moov":
                found_moov = True
                # Recurse into moov container
                moov_end = min(offset + box_size, total_len)
                sub_offset = payload_offset
                while sub_offset + 8 <= moov_end:
                    sub_size = struct.unpack(">I", data[sub_offset:sub_offset+4])[0]
                    sub_type = data[sub_offset+4:sub_offset+8]
                    if sub_size < 8 or sub_offset + sub_size > moov_end:
                        break

                    if sub_type == b"mvhd":
                        ver = data[sub_offset+8]
                        if ver == 0 and sub_offset + 28 <= moov_end:
                            timescale, dur = struct.unpack(">II", data[sub_offset+20:sub_offset+28])
                            if timescale > 0:
                                duration = round(dur / float(timescale), 3)
                                meta["timescale"] = timescale
                        elif ver == 1 and sub_offset + 40 <= moov_end:
                            timescale = struct.unpack(">I", data[sub_offset+28:sub_offset+32])[0]
                            dur = struct.unpack(">Q", data[sub_offset+32:sub_offset+40])[0]
                            if timescale > 0:
                                duration = round(dur / float(timescale), 3)
                                meta["timescale"] = timescale

                    elif sub_type == b"trak":
                        # Look for tkhd inside trak to find width/height
                        trak_end = sub_offset + sub_size
                        t_offset = sub_offset + 8
                        while t_offset + 8 <= trak_end:
                            t_size = struct.unpack(">I", data[t_offset:t_offset+4])[0]
                            t_type = data[t_offset+4:t_offset+8]
                            if t_size < 8 or t_offset + t_size > trak_end:
                                break
                            if t_type == b"tkhd":
                                ver = data[t_offset+8]
                                dim_offset = (t_offset + 88) if ver == 0 else (t_offset + 100)
                                if dim_offset + 8 <= trak_end:
                                    w_fixed, h_fixed = struct.unpack(">II", data[dim_offset:dim_offset+8])
                                    w = w_fixed >> 16
                                    h = h_fixed >> 16
                                    if w > 0 and h > 0 and dimensions is None:
                                        dimensions = (w, h)
                            t_offset += t_size

                    sub_offset += sub_size

            offset += box_size

        if not found_ftyp or not found_moov:
            meta["error_code"] = "VIDEO_CONTAINER_CORRUPT"
            return None, None, meta

        meta["has_mdat"] = found_mdat
        if not found_mdat or mdat_size <= 0:
            meta["error_code"] = "VIDEO_ZERO_FRAMES"
            if require_mdat:
                return None, None, meta

        # Optional deep decode check via cv2 or av if available
        try:
            import cv2
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(data)
                tmp.flush()
                cap = cv2.VideoCapture(tmp.name)
                if cap.isOpened():
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    meta["frame_count"] = fc
                    ret, frame = cap.read()
                    cap.release()
                    if not ret or frame is None or fc <= 0:
                        meta["decode_error"] = "Failed to decode frame via cv2"
        except (ImportError, Exception):
            pass

    except Exception:
        meta["error_code"] = "VIDEO_CONTAINER_CORRUPT"
        return None, None, meta

    return duration, dimensions, meta


def validate_decision_schema(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates structured decision / routing / ranking response schema."""
    if not isinstance(payload, dict):
        return False, "Decision payload must be a JSON object"

    # Accept common decision schemas: choice, predicate, scores, ranking
    has_decision = any(k in payload for k in ("decision", "choice", "predicate", "selected", "route", "result"))
    has_confidence = any(k in payload for k in ("confidence", "probability", "score", "scores", "rankings", "logits"))

    if not (has_decision or has_confidence):
        return False, "Decision payload missing decision/choice/score fields"

    if "confidence" in payload:
        conf_val = payload["confidence"]
        if not isinstance(conf_val, (int, float)) or isinstance(conf_val, bool):
            return False, "Confidence must be a numerical float in [0.0, 1.0]"
        conf = float(conf_val)
        if not (0.0 <= conf <= 1.0):
            return False, f"Confidence {conf} out of valid [0.0, 1.0] range"

    if "probability" in payload:
        prob_val = payload["probability"]
        if not isinstance(prob_val, (int, float)) or isinstance(prob_val, bool):
            return False, "Probability must be a numerical float in [0.0, 1.0]"
        prob = float(prob_val)
        if not (0.0 <= prob <= 1.0):
            return False, f"Probability {prob} out of valid [0.0, 1.0] range"

    return True, None


def validate_embedding_vector(
    vector: Any,
    expected_dim: Optional[int] = None,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validates embedding float array, dimensions, and L2 norm."""
    if not isinstance(vector, (list, tuple)):
        return False, "Embedding vector must be a list or tuple of floats", {"error_code": "EMBEDDING_INVALID_TYPE"}

    dim = len(vector)
    if dim == 0:
        return False, "Embedding vector is empty (0 dimensions)", {"dim": dim, "error_code": "EMBEDDING_ZERO_NORM"}

    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector):
        return False, "Embedding elements must be numerical floats", {"dim": dim, "error_code": "EMBEDDING_NON_NUMERIC"}

    if any(math.isnan(float(x)) or math.isinf(float(x)) for x in vector):
        return False, "Embedding contains NaN or Inf values", {"dim": dim, "error_code": "EMBEDDING_NAN_OR_INF"}

    if expected_dim is not None and dim != expected_dim:
        return False, f"Embedding dimension mismatch: expected {expected_dim}, got {dim}", {"dim": dim, "error_code": "EMBEDDING_DIMENSION_MISMATCH"}

    # Compute norm
    try:
        l2_sum = sum(float(x) ** 2 for x in vector)
        l2_norm = math.sqrt(l2_sum)
    except (ValueError, TypeError, OverflowError) as e:
        return False, f"Embedding computation failed: {e}", {"dim": dim, "error_code": "EMBEDDING_NAN_OR_INF"}

    if math.isnan(l2_norm) or math.isinf(l2_norm):
        return False, "Embedding contains NaN or Inf values", {"dim": dim, "error_code": "EMBEDDING_NAN_OR_INF"}

    if l2_norm == 0.0:
        return False, "Embedding vector is all zeros (zero L2 norm)", {"dim": dim, "error_code": "EMBEDDING_ZERO_NORM"}

    return True, None, {"dim": dim, "l2_norm": round(l2_norm, 5)}


def validate_abc_score(text: str) -> Tuple[bool, Optional[str]]:
    """Validates YuE2 ABC symbolic score headers and body."""
    if not isinstance(text, str) or not text.strip():
        return False, "ABC score text is empty"

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    header_keys = set()
    for line in lines:
        if len(line) >= 2 and line[1] == ":" and line[0].isupper():
            header_keys.add(line[0])

    # Check for mandatory ABC headers: X: (reference), T: (title), M: (meter), K: (key)
    mandatory = {"X", "T", "K"}
    if not mandatory.issubset(header_keys):
        return False, f"Missing required ABC score headers {mandatory - header_keys} in:\n{text[:200]}"

    return True, None


def validate_artifact(
    content: Union[bytes, str, Dict[str, Any], List[Any], Path],
    expected_mime: Optional[str] = None,
    expected_format: Optional[str] = None,
    min_bytes: int = 1,
    rules: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Unified entrypoint for validating benchmark artifacts.
    Supports raw bytes, file paths, JSON strings, parsed dicts/lists.
    """
    rules = rules or {}

    # Resolve content to raw bytes or structured data
    raw_bytes: Optional[bytes] = None
    structured_data: Optional[Any] = None

    if isinstance(content, Path) or (isinstance(content, str) and (content.startswith("/") or content.startswith("./")) and os.path.exists(content)):
        path_obj = Path(content)
        if not path_obj.exists() or not path_obj.is_file():
            return ValidationResult(
                valid=False,
                error_code="FILE_NOT_FOUND",
                error=f"Artifact file not found: {content}",
            )
        raw_bytes = path_obj.read_bytes()

    elif isinstance(content, (bytes, bytearray)):
        raw_bytes = bytes(content)
        # Attempt to parse as structured JSON if possible
        stripped = raw_bytes.strip()
        if (stripped.startswith(b"{") and stripped.endswith(b"}")) or (stripped.startswith(b"[") and stripped.endswith(b"]")):
            try:
                structured_data = json.loads(stripped.decode("utf-8"))
            except Exception:
                pass

    elif isinstance(content, str):
        # Try JSON parsing
        stripped = content.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                structured_data = json.loads(stripped)
            except Exception:
                structured_data = None
        raw_bytes = content.encode("utf-8")

    elif isinstance(content, (dict, list)):
        structured_data = content
        raw_bytes = json.dumps(content).encode("utf-8")

    elif content is None:
        return ValidationResult(valid=False, error_code="ARTIFACT_NULL", error="Artifact content is None")

    else:
        return ValidationResult(valid=False, error_code="UNSUPPORTED_CONTENT_TYPE", error=f"Unsupported artifact content type: {type(content).__name__}")

    size_bytes = len(raw_bytes) if raw_bytes else 0
    sha256 = compute_sha256(raw_bytes) if raw_bytes else None

    # 1. Non-empty check
    if size_bytes < min_bytes:
        return ValidationResult(
            valid=False,
            error_code="ARTIFACT_EMPTY",
            size_bytes=size_bytes,
            sha256=sha256,
            error=f"Artifact size {size_bytes} bytes is below minimum threshold of {min_bytes} bytes",
        )

    # 2. Binary container header detection
    detected_mime, detected_fmt = detect_binary_container(raw_bytes)
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = {}

    if detected_mime == "image/png":
        dimensions, metadata = extract_png_metadata(raw_bytes)
    elif detected_mime in ("image/jpeg", "image/webp"):
        dimensions, metadata = extract_image_metadata(raw_bytes)
    elif detected_mime == "audio/wav":
        check_sil = bool(rules.get("reject_silence") or rules.get("check_silence"))
        duration, metadata = extract_wav_metadata(raw_bytes, check_silence=check_sil)
    elif detected_mime == "video/mp4":
        req_mdat = bool(rules.get("require_media_payload") or rules.get("require_mdat") or rules.get("check_frames"))
        duration, dimensions, metadata = extract_mp4_metadata(raw_bytes, require_mdat=req_mdat)

    # 3. Text / JSON handling
    if detected_mime is None:
        if structured_data is not None:
            detected_mime = "application/json"
            detected_fmt = "json"
        else:
            # Check if valid UTF-8 plain text
            try:
                text_content = raw_bytes.decode("utf-8")
                detected_mime = "text/plain"
                detected_fmt = "text"
                if not text_content.strip() and expected_mime == "text/plain":
                    return ValidationResult(
                        valid=False,
                        error_code="TEXT_EMPTY",
                        mime_type=detected_mime,
                        format=detected_fmt,
                        size_bytes=size_bytes,
                        sha256=sha256,
                        error="Text artifact is empty or contains only whitespace",
                    )
            except UnicodeDecodeError:
                return ValidationResult(
                    valid=False,
                    error_code="UNRECOGNIZED_FORMAT",
                    size_bytes=size_bytes,
                    sha256=sha256,
                    error="Binary artifact does not match any recognized media container magic header and is not valid UTF-8",
                )

    # 4. Check expected MIME type
    if expected_mime:
        expected_mime_clean = expected_mime.lower().split(";")[0].strip()
        detected_clean = (detected_mime or "").lower().split(";")[0].strip()
        if expected_mime_clean != detected_clean:
            # Allow compatible types: application/json vs text/plain, image/png vs image/*
            compatible = (
                (expected_mime_clean.endswith("/*") and detected_clean.startswith(expected_mime_clean[:-1]))
                or (expected_mime_clean == "application/json" and detected_clean == "text/plain" and structured_data is not None)
            )
            if not compatible:
                return ValidationResult(
                    valid=False,
                    error_code="MIME_TYPE_MISMATCH",
                    mime_type=detected_mime,
                    format=detected_fmt,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    error=f"MIME type mismatch: expected '{expected_mime}', detected '{detected_mime}'",
                )

    # 5. Check expected format
    if expected_format:
        if detected_fmt != expected_format.lower():
            return ValidationResult(
                valid=False,
                error_code="FORMAT_MISMATCH",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error=f"Format mismatch: expected '{expected_format}', detected '{detected_fmt}'",
            )

    # 6. Apply domain rules
    # SHA-256 integrity rule
    expected_sha = rules.get("sha256") or rules.get("expected_sha256")
    if expected_sha and (sha256 is None or sha256.lower() != str(expected_sha).strip().lower()):
        return ValidationResult(
            valid=False,
            error_code="CHECKSUM_MISMATCH",
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            dimensions=dimensions,
            duration_seconds=duration,
            metadata=metadata,
            error=f"SHA-256 mismatch: expected '{expected_sha}', got '{sha256}'",
        )

    # Exact size rule
    expected_size = rules.get("expected_size_bytes") or rules.get("exact_size")
    if expected_size is not None and size_bytes != int(expected_size):
        return ValidationResult(
            valid=False,
            error_code="SIZE_MISMATCH",
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            dimensions=dimensions,
            duration_seconds=duration,
            metadata=metadata,
            error=f"Size mismatch: expected {expected_size} bytes, got {size_bytes} bytes",
        )

    # Dimension rules gating
    has_dim_rules = any(k in rules for k in ("exact_width", "exact_height", "min_width", "min_height"))
    if has_dim_rules and dimensions is None:
        return ValidationResult(
            valid=False,
            error_code="IMAGE_DIMENSION_EXTRACTION_FAILED",
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Unable to extract image dimensions from corrupted or unsupported container",
        )

    # Duration rules gating
    has_dur_rules = any(k in rules for k in ("min_duration", "max_duration"))
    if has_dur_rules and duration is None:
        err_cd = "AUDIO_DURATION_EXTRACTION_FAILED" if detected_mime == "audio/wav" else "VIDEO_DURATION_EXTRACTION_FAILED"
        return ValidationResult(
            valid=False,
            error_code=err_cd,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Unable to extract media duration from corrupted or unsupported container",
        )

    # Container integrity verification for recognized media containers
    if detected_mime in ("image/png", "image/jpeg", "image/webp") and dimensions is None:
        err_cd = metadata.get("error_code") or "IMAGE_CORRUPTED"
        return ValidationResult(
            valid=False,
            error_code=err_cd,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    if detected_mime == "audio/wav" and duration is None:
        err_cd = metadata.get("error_code") or "AUDIO_CONTAINER_CORRUPT"
        return ValidationResult(
            valid=False,
            error_code=err_cd,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    if detected_mime == "video/mp4" and duration is None and dimensions is None:
        err_cd = metadata.get("error_code") or "VIDEO_CONTAINER_CORRUPT"
        return ValidationResult(
            valid=False,
            error_code=err_cd,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    # Specific audio silence rejection
    if detected_mime == "audio/wav" and (rules.get("reject_silence") or rules.get("check_silence")):
        if metadata.get("error_code") == "AUDIO_SILENT":
            return ValidationResult(
                valid=False,
                error_code="AUDIO_SILENT",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                duration_seconds=duration,
                metadata=metadata,
                error="Audio is digitally silent (RMS amplitude < 1e-4 and peak == 0.0)",
            )

    # Specific hollow video rejection
    if detected_mime == "video/mp4" and (rules.get("require_media_payload") or rules.get("require_mdat") or rules.get("check_frames")):
        if metadata.get("error_code") == "VIDEO_ZERO_FRAMES":
            return ValidationResult(
                valid=False,
                error_code="VIDEO_ZERO_FRAMES",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                duration_seconds=duration,
                metadata=metadata,
                error="MP4 video container missing non-empty mdat media payload box",
            )

    # Dimension rules
    if dimensions:
        w, h = dimensions
        if "exact_width" in rules and w != rules["exact_width"]:
            return ValidationResult(
                valid=False,
                error_code="IMAGE_DIMENSION_MISMATCH",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                dimensions=dimensions,
                error=f"Image width {w} != expected exact {rules['exact_width']}",
            )
        if "exact_height" in rules and h != rules["exact_height"]:
            return ValidationResult(
                valid=False,
                error_code="IMAGE_DIMENSION_MISMATCH",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                dimensions=dimensions,
                error=f"Image height {h} != expected exact {rules['exact_height']}",
            )
        if "min_width" in rules and w < rules["min_width"]:
            return ValidationResult(
                valid=False,
                error_code="IMAGE_DIMENSION_OUT_OF_BOUNDS",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                dimensions=dimensions,
                error=f"Image width {w} < min {rules['min_width']}",
            )
        if "min_height" in rules and h < rules["min_height"]:
            return ValidationResult(
                valid=False,
                error_code="IMAGE_DIMENSION_OUT_OF_BOUNDS",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                dimensions=dimensions,
                error=f"Image height {h} < min {rules['min_height']}",
            )

    # Duration rules
    if duration is not None:
        err_dur_code = "AUDIO_DURATION_OUT_OF_BOUNDS" if detected_mime == "audio/wav" else "VIDEO_DURATION_OUT_OF_BOUNDS"
        if "min_duration" in rules and duration < rules["min_duration"]:
            return ValidationResult(
                valid=False,
                error_code=err_dur_code,
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                duration_seconds=duration,
                error=f"Media duration {duration}s < min {rules['min_duration']}s",
            )
        if "max_duration" in rules and duration > rules["max_duration"]:
            return ValidationResult(
                valid=False,
                error_code=err_dur_code,
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                duration_seconds=duration,
                error=f"Media duration {duration}s > max {rules['max_duration']}s",
            )

    # Semantic structured rules
    if rules.get("is_decision"):
        if not isinstance(structured_data, dict):
            return ValidationResult(
                valid=False,
                error_code="DECISION_SCHEMA_INVALID",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Decision rule specified but payload is not a valid JSON dictionary",
            )
        ok, err = validate_decision_schema(structured_data)
        if not ok:
            err_cd = "CONFIDENCE_OUT_OF_RANGE" if ("out of valid" in str(err) or "Confidence must be" in str(err) or "Probability must be" in str(err)) else "DECISION_SCHEMA_INVALID"
            return ValidationResult(
                valid=False,
                error_code=err_cd,
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error=f"Decision validation failed: {err}",
            )

    if rules.get("is_embedding"):
        vec = None
        if isinstance(structured_data, list):
            vec = structured_data
        elif isinstance(structured_data, dict):
            # Check OpenAI standard response: {"data": [{"embedding": [...]}]}
            if "data" in structured_data and isinstance(structured_data["data"], list) and structured_data["data"]:
                first_item = structured_data["data"][0]
                if isinstance(first_item, dict) and "embedding" in first_item:
                    vec = first_item["embedding"]
            elif "embedding" in structured_data:
                vec = structured_data["embedding"]

        exp_dim = rules.get("expected_dim")
        ok, err, emb_meta = validate_embedding_vector(vec, expected_dim=exp_dim)
        metadata.update(emb_meta)
        if not ok:
            err_cd = emb_meta.get("error_code") or "EMBEDDING_INVALID_TYPE"
            return ValidationResult(
                valid=False,
                error_code=err_cd,
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                metadata=metadata,
                error=f"Embedding validation failed: {err}",
            )

    if rules.get("is_abc_score"):
        text = raw_bytes.decode("utf-8", errors="ignore")
        ok, err = validate_abc_score(text)
        if not ok:
            return ValidationResult(
                valid=False,
                error_code="ABC_SCORE_INVALID",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error=f"ABC score validation failed: {err}",
            )

    # JSON schema / required fields enforcement
    if rules.get("schema") or rules.get("required_fields"):
        if not isinstance(structured_data, dict):
            return ValidationResult(
                valid=False,
                error_code="INVALID_JSON",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Schema validation failed: Structured payload is not a valid JSON dictionary",
            )
        req_fields: List[str] = list(rules.get("required_fields") or [])
        if isinstance(rules.get("schema"), dict) and "required" in rules["schema"]:
            req_fields.extend(rules["schema"]["required"])
        for req_k in set(req_fields):
            if req_k not in structured_data:
                return ValidationResult(
                    valid=False,
                    error_code="REQUIRED_FIELD_MISSING",
                    mime_type=detected_mime,
                    format=detected_fmt,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    error=f"Required field '{req_k}' missing from JSON payload",
                )
            val = structured_data[req_k]
            if val is None or val == "" or val == [] or val == {}:
                return ValidationResult(
                    valid=False,
                    error_code="EMPTY_REQUIRED_FIELD",
                    mime_type=detected_mime,
                    format=detected_fmt,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    error=f"Required field '{req_k}' is empty or null in JSON payload",
                )

    # Chat completion response non-empty enforcement
    if rules.get("is_chat"):
        if not isinstance(structured_data, dict):
            return ValidationResult(
                valid=False,
                error_code="INVALID_JSON",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Chat completion payload is not a valid JSON dictionary",
            )
        choices = structured_data.get("choices")
        if not choices or not isinstance(choices, list):
            return ValidationResult(
                valid=False,
                error_code="CHAT_RESPONSE_EMPTY",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Chat response missing valid choices array",
            )
        first_choice = choices[0]
        msg = first_choice.get("message") if isinstance(first_choice, dict) else None
        text_val = first_choice.get("text") if isinstance(first_choice, dict) else None
        content_val = msg.get("content") if isinstance(msg, dict) else text_val
        if not content_val or not str(content_val).strip():
            return ValidationResult(
                valid=False,
                error_code="CHAT_RESPONSE_EMPTY",
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Chat response choice has empty message content",
            )

    return ValidationResult(
        valid=True,
        mime_type=detected_mime,
        format=detected_fmt,
        size_bytes=size_bytes,
        sha256=sha256,
        dimensions=dimensions,
        duration_seconds=duration,
        metadata=metadata,
    )
