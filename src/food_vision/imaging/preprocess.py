"""Deterministic image normalization: decode, orient, resize, strip, encode.

docs/dev/pre-study-web-ui.md §7.5 / §4.2: refuse above a decoded-pixel
guard before full decode, apply EXIF orientation (otherwise portrait phone
photos are sent sideways), resize to a fixed long edge, drop all metadata,
re-encode as JPEG. No cropping, no reference-card detection — the card is
a scale cue for the *model*, not for our code.

``normalize()`` is a pure function: the same ``raw`` bytes and
:class:`ImageConfig` always produce the same output bytes (tested by
hash) — no timestamps, no EXIF, no random encoder state.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pillow_heif
from PIL import Image, ImageOps

#: Registers the HEIF/HEIC opener with Pillow's ``Image.open`` dispatch.
#: Safe to call more than once; done at import time so every caller gets
#: HEIC support without remembering to opt in.
pillow_heif.register_heif_opener()

__all__ = ["ImageConfig", "ImageTooLargeError", "normalize"]


class ImageTooLargeError(ValueError):
    """Raised when an image's decoded pixel count exceeds the configured guard."""


@dataclass(frozen=True)
class ImageConfig:
    """Normalization parameters. Part of a pre-study ``run_config`` (§5.3)."""

    max_decoded_pixels: int
    long_edge_px: int = 1024
    jpeg_quality: int = 90


def normalize(raw: bytes, cfg: ImageConfig) -> bytes:
    """Decode, orient, resize, strip metadata, and re-encode as JPEG.

    Raises:
        ImageTooLargeError: the image's width × height (as reported by the
            file header, before pixel data is fully decoded) exceeds
            ``cfg.max_decoded_pixels``.
    """
    with Image.open(io.BytesIO(raw)) as opened:
        width, height = opened.size
        if width * height > cfg.max_decoded_pixels:
            raise ImageTooLargeError(
                f"Image is {width}x{height} = {width * height} px, "
                f"exceeding the {cfg.max_decoded_pixels} px guard."
            )

        # Applies the EXIF Orientation tag physically and drops it from the
        # returned image, so a portrait phone photo isn't sent sideways.
        transposed = ImageOps.exif_transpose(opened)

    rgb = transposed.convert("RGB")
    resized = _resize_long_edge(rgb, cfg.long_edge_px)

    # Drop every other metadata key Pillow may have carried over (EXIF,
    # ICC profile, XMP, ...) by building a fresh in-memory copy with none
    # of it rather than trying to enumerate and strip each key.
    stripped = Image.new(resized.mode, resized.size)
    stripped.putdata(list(resized.getdata()))

    buffer = io.BytesIO()
    stripped.save(buffer, format="JPEG", quality=cfg.jpeg_quality)
    return buffer.getvalue()


def _resize_long_edge(image: Image.Image, long_edge_px: int) -> Image.Image:
    width, height = image.size
    if width >= height:
        new_width = long_edge_px
        new_height = max(1, round(height * (long_edge_px / width)))
    else:
        new_height = long_edge_px
        new_width = max(1, round(width * (long_edge_px / height)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
