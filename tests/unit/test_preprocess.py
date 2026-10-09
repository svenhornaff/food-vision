"""imaging.preprocess: determinism, orientation, metadata stripping, guards."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from food_vision.imaging.preprocess import ImageConfig, ImageTooLargeError, normalize

_ORIENTATION_TAG = 0x0112
_MAKE_TAG = 0x010F
_MODEL_TAG = 0x0110


def _jpeg_bytes(
    size: tuple[int, int] = (200, 100),
    color: tuple[int, int, int] = (10, 20, 30),
    exif_tags: dict[int, object] | None = None,
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    if exif_tags:
        exif = image.getexif()
        for tag, value in exif_tags.items():
            exif[tag] = value
        image.save(buffer, format="JPEG", exif=exif)
    else:
        image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _default_cfg(
    max_decoded_pixels: int = 10_000_000, long_edge_px: int = 64, jpeg_quality: int = 90
) -> ImageConfig:
    return ImageConfig(
        max_decoded_pixels=max_decoded_pixels, long_edge_px=long_edge_px, jpeg_quality=jpeg_quality
    )


def test_normalize_is_deterministic() -> None:
    raw = _jpeg_bytes()
    cfg = _default_cfg()

    first = normalize(raw, cfg)
    second = normalize(raw, cfg)

    assert first == second


def test_normalize_resizes_long_edge() -> None:
    raw = _jpeg_bytes(size=(200, 100))
    cfg = _default_cfg(long_edge_px=64)

    out = normalize(raw, cfg)

    with Image.open(io.BytesIO(out)) as result:
        assert result.size == (64, 32)


def test_normalize_applies_exif_orientation() -> None:
    """Orientation 6 = rotate 270 (phone held one way vs another):
    pixel data must be physically rotated, not just tagged."""
    portrait_landscape_tagged = _jpeg_bytes(size=(200, 100), exif_tags={_ORIENTATION_TAG: 6})
    cfg = _default_cfg(long_edge_px=64)

    out = normalize(portrait_landscape_tagged, cfg)

    with Image.open(io.BytesIO(out)) as result:
        # A 200x100 source rotated 90deg becomes taller than wide.
        assert result.size[1] > result.size[0]


def test_normalize_strips_exif_and_other_metadata() -> None:
    raw = _jpeg_bytes(exif_tags={_MAKE_TAG: "PhoneCo", _MODEL_TAG: "Model X"})

    out = normalize(raw, _default_cfg())

    with Image.open(io.BytesIO(out)) as result:
        assert len(result.getexif()) == 0
        assert "exif" not in result.info


def test_normalize_rejects_oversize_image() -> None:
    raw = _jpeg_bytes(size=(200, 100))
    cfg = _default_cfg(max_decoded_pixels=100)  # 200*100 = 20,000 >> 100

    with pytest.raises(ImageTooLargeError):
        normalize(raw, cfg)


def test_normalize_decodes_heic_fixture() -> None:
    import pillow_heif  # noqa: F401 (registers the opener as a side effect on import)

    image = Image.new("RGB", (120, 80), (5, 6, 7))
    buffer = io.BytesIO()
    image.save(buffer, format="HEIF", quality=80)
    raw = buffer.getvalue()

    out = normalize(raw, _default_cfg(long_edge_px=64))

    with Image.open(io.BytesIO(out)) as result:
        assert result.format == "JPEG"
        assert max(result.size) == 64
