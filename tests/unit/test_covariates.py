"""imaging.covariates: EXIF device/lens covariates, read before stripping."""

from __future__ import annotations

import io

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from food_vision.imaging.covariates import extract

_MAKE_TAG = 0x010F
_MODEL_TAG = 0x0110
_EXIF_IFD_POINTER = 0x8769
_FOCAL_LENGTH_TAG = 0x920A
_FOCAL_LENGTH_35MM_TAG = 0xA405


def _jpeg_with_exif(
    size: tuple[int, int] = (200, 100),
    make: str | None = "PhoneCo",
    model: str | None = "Model X",
    focal_length_mm: float | None = 4.25,
    focal_length_35mm: int | None = 26,
) -> bytes:
    image = Image.new("RGB", size, (1, 2, 3))
    exif = image.getexif()
    if make is not None:
        exif[_MAKE_TAG] = make
    if model is not None:
        exif[_MODEL_TAG] = model

    exif_ifd = exif.get_ifd(_EXIF_IFD_POINTER)
    if focal_length_mm is not None:
        # IFDRational(float) misparses a single-float argument as
        # numerator=float/denominator=1 — pass numerator/denominator.
        exif_ifd[_FOCAL_LENGTH_TAG] = IFDRational(round(focal_length_mm * 100), 100)
    if focal_length_35mm is not None:
        exif_ifd[_FOCAL_LENGTH_35MM_TAG] = focal_length_35mm

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_extract_reads_make_model_and_focal_length() -> None:
    raw = _jpeg_with_exif()

    covariates = extract(raw)

    assert covariates.make == "PhoneCo"
    assert covariates.model == "Model X"
    assert covariates.focal_length_mm == 4.25
    assert covariates.focal_length_35mm == 26
    assert covariates.width == 200
    assert covariates.height == 100


def test_extract_handles_missing_exif() -> None:
    image = Image.new("RGB", (64, 48), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    covariates = extract(buffer.getvalue())

    assert covariates.make is None
    assert covariates.model is None
    assert covariates.focal_length_mm is None
    assert covariates.focal_length_35mm is None
    assert covariates.width == 64
    assert covariates.height == 48


def test_extract_handles_exif_ifd_without_focal_length() -> None:
    """Exif sub-IFD present (35mm tag set) but no FocalLength tag at all."""
    raw = _jpeg_with_exif(focal_length_mm=None, focal_length_35mm=26)

    covariates = extract(raw)

    assert covariates.focal_length_mm is None
    assert covariates.focal_length_35mm == 26


def test_extract_never_mutates_or_strips_source_bytes() -> None:
    """extract() is read-only; the caller still has the untouched raw
    bytes to pass to preprocess.normalize() afterwards."""
    raw = _jpeg_with_exif()
    raw_copy = bytes(raw)

    extract(raw)

    assert raw == raw_copy
