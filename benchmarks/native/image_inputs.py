"""Deterministic technical image inputs; no generated model output."""

import binascii
import io
import struct
import zlib

from PIL import Image


def png(image):
    image = image.convert("RGBA")

    def chunk(kind, raw):
        return (
            struct.pack(">I", len(raw))
            + kind
            + raw
            + struct.pack(">I", binascii.crc32(kind + raw) & 0xFFFFFFFF)
        )

    pixels = image.tobytes()
    stride = image.width * 4
    raw = b"".join(b"\0" + pixels[i : i + stride] for i in range(0, len(pixels), stride))
    blocks = [raw[i : i + 65535] for i in range(0, len(raw), 65535)]
    compressed = (
        b"\x78\x01"
        + b"".join(
            bytes([int(i == len(blocks) - 1)]) + struct.pack("<HH", len(b), 65535 - len(b)) + b
            for i, b in enumerate(blocks)
        )
        + struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def derive(recipe, originals):
    if recipe["kind"] == "alpha_mask":
        image = Image.new("RGBA", tuple(recipe["size"]), (255, 255, 255, 255))
        image.paste((255, 255, 255, 0), tuple(recipe["region"]))
    elif recipe["kind"] == "expanded_canvas":
        image = Image.new("RGBA", tuple(recipe["size"]), tuple(recipe["fill_rgba"]))
        original = Image.open(io.BytesIO(originals[recipe["source"]])).convert("RGBA")
        image.paste(original, (0, 0))
    else:
        raise ValueError("Unknown input recipe")
    return png(image)
