"""
factory/apps/control-api/src/assets/media_validator.py
Magic Byte Sniffing and Media Integrity Validator
"""
import io
import struct
from typing import Optional
from factory_contracts.errors import InvalidMediaError

try:
    from PIL import Image
except ImportError:
    Image = None

MAX_IMAGE_DIMENSION = 4096
MAX_TOTAL_PIXELS = 16_777_216  # 16 Megapixels decompression bomb limit

MIN_MEDIA_SIZES = {
    # Video
    "video/mp4": 32,
    "video/webm": 24,
    # Audio
    "audio/wav": 36,
    "audio/mpeg": 32,
    "audio/flac": 42,
    "audio/ogg": 28,
    # Image
    "image/png": 24,
    "image/jpeg": 24,
    "image/webp": 24,
}


def detect_mime_type(header_bytes: bytes) -> Optional[str]:
    """Sniffs MIME type from initial header bytes."""
    if header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif header_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif header_bytes.startswith(b"RIFF") and len(header_bytes) >= 12:
        if header_bytes[8:12] == b"WEBP":
            return "image/webp"
        elif header_bytes[8:12] == b"WAVE":
            return "audio/wav"
    elif any(header_bytes.startswith(sig) for sig in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3")):
        return "audio/mpeg"
    elif header_bytes.startswith(b"OggS"):
        return "audio/ogg"
    elif header_bytes.startswith(b"fLaC"):
        return "audio/flac"
    elif len(header_bytes) >= 8 and header_bytes[4:8] == b"ftyp":
        return "video/mp4"
    elif header_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return None


def _validate_mp4_stream(stream: io.BytesIO, size: int) -> None:
    """Validates ISO BMFF / MP4 container structure."""
    if size < MIN_MEDIA_SIZES["video/mp4"]:
        raise InvalidMediaError(
            message=f"MP4 video stream truncated: size {size} bytes below minimum required {MIN_MEDIA_SIZES['video/mp4']} bytes."
        )
    stream.seek(0)
    header = stream.read(min(size, 64))
    if len(header) < 8:
        raise InvalidMediaError(message="Truncated MP4 box header.")
    atom_size, atom_type = struct.unpack(">I4s", header[:8])
    if atom_type != b"ftyp":
        raise InvalidMediaError(message=f"Invalid MP4 container: initial box must be 'ftyp', found {atom_type!r}.")
    if atom_size == 1:
        if len(header) < 16:
            raise InvalidMediaError(message="Truncated MP4 64-bit box header.")
        actual_atom_size = struct.unpack(">Q", header[8:16])[0]
        if actual_atom_size < 16:
            raise InvalidMediaError(message="Malformed MP4 64-bit atom size.")
        if size < actual_atom_size:
            raise InvalidMediaError(
                message=f"Truncated MP4 video file: declared 64-bit atom size {actual_atom_size} exceeds total stream size {size} bytes."
            )
    elif atom_size < 16:
        raise InvalidMediaError(
            message=f"Malformed MP4 ftyp atom: declared size {atom_size} is less than minimum 16 bytes."
        )
    elif size < atom_size:
        raise InvalidMediaError(
            message=f"Truncated MP4 video file: declared ftyp atom size {atom_size} exceeds total stream size {size} bytes."
        )


def _validate_wav_stream(stream: io.BytesIO, size: int) -> None:
    """Validates RIFF WAVE audio container structure."""
    if size < MIN_MEDIA_SIZES["audio/wav"]:
        raise InvalidMediaError(
            message=f"WAV audio stream truncated or empty: size {size} bytes below minimum required {MIN_MEDIA_SIZES['audio/wav']} bytes."
        )
    stream.seek(0)
    header = stream.read(12)
    if len(header) < 12:
        raise InvalidMediaError(message="Truncated RIFF header.")
    riff_tag, riff_size, wave_tag = struct.unpack("<4sI4s", header)
    if riff_tag != b"RIFF" or wave_tag != b"WAVE":
        raise InvalidMediaError(message="Invalid RIFF WAVE container signature.")

    # Scan RIFF chunks for mandatory fmt chunk
    offset = 12
    found_fmt = False
    while offset + 8 <= size:
        stream.seek(offset)
        chunk_header = stream.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id, chunk_sz = struct.unpack("<4sI", chunk_header)
        if chunk_id == b"fmt ":
            found_fmt = True
            if chunk_sz < 16:
                raise InvalidMediaError(
                    message=f"Malformed WAV fmt chunk: declared size {chunk_sz} is less than minimum 16 bytes."
                )
            if offset + 8 + 16 > size:
                raise InvalidMediaError(
                    message="Truncated WAV audio stream: fmt chunk data extends beyond total stream size."
                )
            break
        offset += 8 + chunk_sz + (chunk_sz % 2)

    if not found_fmt:
        raise InvalidMediaError(
            message="Empty or invalid RIFF WAVE audio stream: missing mandatory 'fmt ' chunk."
        )


def _validate_webm_stream(stream: io.BytesIO, size: int) -> None:
    """Validates WebM EBML container header."""
    if size < MIN_MEDIA_SIZES["video/webm"]:
        raise InvalidMediaError(
            message=f"WebM video stream truncated: size {size} bytes below minimum required {MIN_MEDIA_SIZES['video/webm']} bytes."
        )
    stream.seek(0)
    magic = stream.read(4)
    if magic != b"\x1a\x45\xdf\xa3":
        raise InvalidMediaError(message="Invalid WebM EBML container header signature.")


def _validate_flac_stream(stream: io.BytesIO, size: int) -> None:
    """Validates FLAC container and STREAMINFO block."""
    if size < MIN_MEDIA_SIZES["audio/flac"]:
        raise InvalidMediaError(
            message=f"FLAC audio stream truncated: size {size} bytes below minimum required {MIN_MEDIA_SIZES['audio/flac']} bytes."
        )
    stream.seek(0)
    hdr = stream.read(8)
    if not hdr.startswith(b"fLaC"):
        raise InvalidMediaError(message="Invalid FLAC container signature.")
    block_type = hdr[4] & 0x7F
    if block_type != 0:
        raise InvalidMediaError(message="Invalid FLAC metadata: first block must be STREAMINFO (type 0).")
    block_len = struct.unpack(">I", b"\x00" + hdr[5:8])[0]
    if block_len != 34:
        raise InvalidMediaError(
            message=f"Malformed FLAC STREAMINFO block: declared size {block_len} != expected 34 bytes."
        )


def _validate_ogg_stream(stream: io.BytesIO, size: int) -> None:
    """Validates OGG page header."""
    if size < MIN_MEDIA_SIZES["audio/ogg"]:
        raise InvalidMediaError(
            message=f"Ogg audio stream truncated: size {size} bytes below minimum required {MIN_MEDIA_SIZES['audio/ogg']} bytes."
        )
    stream.seek(0)
    hdr = stream.read(27)
    if not hdr.startswith(b"OggS"):
        raise InvalidMediaError(message="Invalid Ogg container signature.")
    if hdr[4] != 0:
        raise InvalidMediaError(message=f"Unsupported Ogg bitstream structure version: {hdr[4]}.")


def _validate_mp3_stream(stream: io.BytesIO, size: int) -> None:
    """Validates MP3 audio frame / ID3 header."""
    if size < MIN_MEDIA_SIZES["audio/mpeg"]:
        raise InvalidMediaError(
            message=f"MP3 audio stream truncated: size {size} bytes below minimum required {MIN_MEDIA_SIZES['audio/mpeg']} bytes."
        )
    stream.seek(0)
    hdr = stream.read(10)
    if hdr.startswith(b"ID3"):
        if size < 10:
            raise InvalidMediaError(message="Truncated ID3 tag header in MP3 stream.")
    elif len(hdr) >= 2 and hdr[0] == 0xFF and hdr[1] in (0xFB, 0xF3, 0xF2):
        pass
    else:
        raise InvalidMediaError(message="Invalid MPEG audio sync word or ID3 container header.")


def validate_media_content(header_bytes: bytes, full_stream: io.BytesIO, claimed_mime: Optional[str] = None) -> str:
    """
    Validates media content against allowed formats, magic bytes, container atoms/chunks,
    minimum byte lengths, and image decompression limits.
    """
    detected_mime = detect_mime_type(header_bytes)
    if not detected_mime:
        raise InvalidMediaError(message="Unsupported or unrecognized media file header signature.")

    full_stream.seek(0, io.SEEK_END)
    stream_size = full_stream.tell()
    full_stream.seek(0)

    min_size = MIN_MEDIA_SIZES.get(detected_mime, 16)
    if stream_size < min_size:
        raise InvalidMediaError(
            message=f"Media file truncated: size {stream_size} bytes is less than minimum required {min_size} bytes for {detected_mime}."
        )

    if detected_mime.startswith("image/"):
        if Image is not None:
            full_stream.seek(0)
            try:
                with Image.open(full_stream) as img:
                    img.verify()
                    width, height = img.size
                    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                        raise InvalidMediaError(
                            message=f"Image dimensions {width}x{height} exceed maximum permitted {MAX_IMAGE_DIMENSION}px."
                        )
                    if width * height > MAX_TOTAL_PIXELS:
                        raise InvalidMediaError(
                            message=f"Image pixel count {width*height} exceeds decompression limit ({MAX_TOTAL_PIXELS}px)."
                        )
            except Exception as e:
                if isinstance(e, InvalidMediaError):
                    raise e
                raise InvalidMediaError(message=f"Image file integrity verification failed: {e}")
    elif detected_mime == "video/mp4":
        _validate_mp4_stream(full_stream, stream_size)
    elif detected_mime == "video/webm":
        _validate_webm_stream(full_stream, stream_size)
    elif detected_mime == "audio/wav":
        _validate_wav_stream(full_stream, stream_size)
    elif detected_mime == "audio/mpeg":
        _validate_mp3_stream(full_stream, stream_size)
    elif detected_mime == "audio/flac":
        _validate_flac_stream(full_stream, stream_size)
    elif detected_mime == "audio/ogg":
        _validate_ogg_stream(full_stream, stream_size)

    full_stream.seek(0)
    return detected_mime
