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
    Fully verifies chunk structure and container decode integrity.
    Rejects truncated headers, invalid chunks, or corrupted image streams.
    """
    try:
        from PIL import Image
        bio = io.BytesIO(data)
        with Image.open(bio) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            if format_hint and fmt and fmt.upper() != format_hint.upper():
                return None, {}
            w, h = img.size
            if w <= 0 or h <= 0:
                return None, {}
            return (w, h), {
                "format": fmt,
                "mode": img.mode,
                "width": w,
                "height": h,
            }
    except Exception:
        pass
    return None, {}


def extract_png_metadata(data: bytes) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Parses and verifies PNG image container using Pillow decoding verification.
    Extracts dimensions, bit depth, and color type.
    """
    if len(data) < 29 or not data.startswith(PNG_MAGIC):
        return None, {}

    # Strict decodability verification via Pillow
    dims, meta = extract_image_metadata(data, format_hint="PNG")
    if dims is None:
        # Fallback only if Pillow is not available
        try:
            import PIL  # noqa: F401
            # If PIL is installed and failed verification, the image is corrupt
            return None, {}
        except ImportError:
            pass

        try:
            ihdr_type = data[12:16]
            if ihdr_type == b"IHDR":
                width, height = struct.unpack(">II", data[16:24])
                if width <= 0 or height <= 0:
                    return None, {}
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
            return None, {}

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


def extract_wav_metadata(data: bytes) -> Tuple[Optional[float], Dict[str, Any]]:
    """Parses WAV audio using stdlib wave to extract duration and format details."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if framerate <= 0 or channels <= 0 or sample_width <= 0:
                return None, {}
            duration = n_frames / float(framerate)
            return round(duration, 3), {
                "channels": channels,
                "sample_width_bytes": sample_width,
                "framerate_hz": framerate,
                "n_frames": n_frames,
            }
    except Exception:
        pass
    return None, {}


def extract_mp4_metadata(data: bytes) -> Tuple[Optional[float], Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Parses MP4 / ISO BMFF container to locate ftyp brand, mvhd duration,
    and tkhd track dimensions without third-party dependencies.
    """
    duration: Optional[float] = None
    dimensions: Optional[Tuple[int, int]] = None
    meta: Dict[str, Any] = {}

    try:
        offset = 0
        total_len = len(data)

        while offset + 8 <= total_len:
            box_size = struct.unpack(">I", data[offset:offset+4])[0]
            box_type = data[offset+4:offset+8]

            if box_size == 1:
                # 64-bit box size
                if offset + 16 > total_len:
                    break
                box_size = struct.unpack(">Q", data[offset+8:offset+16])[0]
                payload_offset = offset + 16
            elif box_size == 0:
                box_size = total_len - offset
                payload_offset = offset + 8
            else:
                payload_offset = offset + 8

            if box_size < 8:
                break

            if box_type == b"ftyp":
                major_brand = data[payload_offset:payload_offset+4].decode("ascii", errors="ignore")
                meta["major_brand"] = major_brand

            elif box_type == b"moov":
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
    except Exception:
        pass

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
        return False, "Embedding vector must be a list or tuple of floats", {}

    dim = len(vector)
    if dim == 0:
        return False, "Embedding vector is empty (0 dimensions)", {}

    if expected_dim is not None and dim != expected_dim:
        return False, f"Embedding dimension mismatch: expected {expected_dim}, got {dim}", {"dim": dim}

    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector):
        return False, "Embedding elements must be numerical floats", {"dim": dim}

    # Compute norm
    try:
        l2_sum = sum(float(x) ** 2 for x in vector)
        l2_norm = math.sqrt(l2_sum)
    except (ValueError, TypeError, OverflowError) as e:
        return False, f"Embedding computation failed: {e}", {"dim": dim}

    if math.isnan(l2_norm) or math.isinf(l2_norm):
        return False, "Embedding contains NaN or Inf values", {"dim": dim}

    if l2_norm == 0.0:
        return False, "Embedding vector is all zeros (zero L2 norm)", {"dim": dim}

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
                error=f"Artifact file not found: {content}",
            )
        raw_bytes = path_obj.read_bytes()

    elif isinstance(content, (bytes, bytearray)):
        raw_bytes = bytes(content)

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
        return ValidationResult(valid=False, error="Artifact content is None")

    else:
        return ValidationResult(valid=False, error=f"Unsupported artifact content type: {type(content).__name__}")

    size_bytes = len(raw_bytes) if raw_bytes else 0
    sha256 = compute_sha256(raw_bytes) if raw_bytes else None

    # 1. Non-empty check
    if size_bytes < min_bytes:
        return ValidationResult(
            valid=False,
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
        duration, metadata = extract_wav_metadata(raw_bytes)
    elif detected_mime == "video/mp4":
        duration, dimensions, metadata = extract_mp4_metadata(raw_bytes)

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
            except UnicodeDecodeError:
                return ValidationResult(
                    valid=False,
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
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            dimensions=dimensions,
            duration_seconds=duration,
            metadata=metadata,
            error=f"SHA-256 mismatch: expected '{expected_sha}', got '{sha256}'",
        )

    # Dimension rules gating
    has_dim_rules = any(k in rules for k in ("exact_width", "exact_height", "min_width", "min_height"))
    if has_dim_rules and dimensions is None:
        return ValidationResult(
            valid=False,
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
        return ValidationResult(
            valid=False,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Unable to extract media duration from corrupted or unsupported container",
        )

    # Container integrity verification for recognized media containers
    if detected_mime in ("image/png", "image/jpeg", "image/webp") and dimensions is None:
        return ValidationResult(
            valid=False,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    if detected_mime == "audio/wav" and duration is None:
        return ValidationResult(
            valid=False,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    if detected_mime == "video/mp4" and duration is None and dimensions is None:
        return ValidationResult(
            valid=False,
            mime_type=detected_mime,
            format=detected_fmt,
            size_bytes=size_bytes,
            sha256=sha256,
            metadata=metadata,
            error="Corrupted or truncated media container",
        )

    # Dimension rules
    if dimensions:
        w, h = dimensions
        if "exact_width" in rules and w != rules["exact_width"]:
            return ValidationResult(
                valid=False,
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
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                dimensions=dimensions,
                error=f"Image height {h} < min {rules['min_height']}",
            )

    # Duration rules
    if duration is not None:
        if "min_duration" in rules and duration < rules["min_duration"]:
            return ValidationResult(
                valid=False,
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
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error="Decision rule specified but payload is not a valid JSON dictionary",
            )
        ok, err = validate_decision_schema(structured_data)
        if not ok:
            return ValidationResult(
                valid=False,
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error=f"Decision validation failed: {err}",
            )

    if rules.get("is_embedding"):
        vec = structured_data if isinstance(structured_data, list) else None
        if vec is None and isinstance(structured_data, dict) and "embedding" in structured_data:
            vec = structured_data["embedding"]
        exp_dim = rules.get("expected_dim")
        ok, err, emb_meta = validate_embedding_vector(vec, expected_dim=exp_dim)
        metadata.update(emb_meta)
        if not ok:
            return ValidationResult(
                valid=False,
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
                mime_type=detected_mime,
                format=detected_fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                error=f"ABC score validation failed: {err}",
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
