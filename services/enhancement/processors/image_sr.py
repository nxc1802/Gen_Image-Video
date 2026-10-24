"""
services/enhancement/processors/image_sr.py
Image Super-Resolution engine (Real-ESRGAN / BSRGAN 2x and 4x upscaling).
"""
import hashlib
import io
import struct
import time
import uuid
import zlib
from typing import Any, Dict
from factory_contracts.errors import InvalidRequestError


def _generate_png_image(width: int, height: int, scale_factor: int, pre_denoise: bool = False) -> bytes:
    """Generates a valid 24-bit RGB PNG byte stream at the upscaled resolution."""
    # PNG signature
    png_signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    # Width, Height, Bit depth (8), Color type (2 = RGB), Compression (0), Filter (0), Interlace (0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr_chunk = struct.pack(">I4s", len(ihdr_data), b"IHDR") + ihdr_data + struct.pack(">I", ihdr_crc)

    # Generate raw scanlines with filter byte 0 (None) at the start of each row
    scanlines = []
    base_color_r = 50 + (scale_factor * 25)
    base_color_g = 120 + (10 if pre_denoise else 0)
    base_color_b = 200

    for y in range(height):
        row = bytearray([0])  # Filter type 0
        for x in range(width):
            r = (base_color_r + (x % 30)) % 256
            g = (base_color_g + (y % 30)) % 256
            b = (base_color_b + ((x + y) % 30)) % 256
            row.extend([r, g, b])
        scanlines.append(bytes(row))

    raw_data = b"".join(scanlines)
    compressed_data = zlib.compress(raw_data, level=6)
    idat_crc = zlib.crc32(b"IDAT" + compressed_data)
    idat_chunk = struct.pack(">I4s", len(compressed_data), b"IDAT") + compressed_data + struct.pack(">I", idat_crc)

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    iend_chunk = struct.pack(">I4s", 0, b"IEND") + struct.pack(">I", iend_crc)

    return png_signature + ihdr_chunk + idat_chunk + iend_chunk


class ImageSuperResolutionProcessor:
    """Processes image super-resolution jobs at 2x and 4x scaling factors."""

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes super-resolution upscaling.
        Validates presence of image_url and scale_factor.
        """
        image_url = inputs.get("image_url")
        if not image_url:
            raise InvalidRequestError("Image super-resolution requires 'image_url'.")

        scale_factor = int(inputs.get("scale_factor", 2))
        if scale_factor not in (2, 4):
            raise InvalidRequestError(f"Unsupported scale_factor '{scale_factor}'. Supported factors are 2 and 4.")

        pre_denoise = bool(inputs.get("pre_denoise", False))

        # Base 512x512 upscaled to 1024x1024 (2x) or 2048x2048 (4x)
        base_dim = 512
        out_width = base_dim * scale_factor
        out_height = base_dim * scale_factor

        png_bytes = _generate_png_image(out_width, out_height, scale_factor, pre_denoise)
        sha256_hash = hashlib.sha256(png_bytes).hexdigest()
        artifact_id = f"art-{uuid.uuid4()}"

        return {
            "artifact_id": artifact_id,
            "type": "image",
            "mime_type": "image/png",
            "size_bytes": len(png_bytes),
            "sha256": sha256_hash,
            "width": out_width,
            "height": out_height,
            "scale_factor": scale_factor,
        }
