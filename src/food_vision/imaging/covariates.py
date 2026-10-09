"""EXIF covariate extraction — read before ``preprocess.normalize`` strips it.

docs/dev/pre-study-web-ui.md §7.5 / §9 (Covariates): phone model and focal
length are cheap to record now and impossible to reconstruct later, but
must never be sent to a model or logged. Callers store
:class:`ExifCovariates` directly in the (M2) ``images`` table.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import ExifTags, Image

__all__ = ["ExifCovariates", "extract"]

_IFD_MAKE = 271  # ExifTags.Base.Make
_IFD_MODEL = 272  # ExifTags.Base.Model
_FOCAL_LENGTH = 37386  # ExifTags.Base.FocalLength (Exif sub-IFD)
_FOCAL_LENGTH_35MM = 41989  # ExifTags.Base.FocalLengthIn35mmFilm (Exif sub-IFD)


@dataclass(frozen=True)
class ExifCovariates:
    """Capture-device covariates. ``width``/``height`` are the source
    image's dimensions as decoded, *before* EXIF-orientation transpose."""

    make: str | None
    model: str | None
    focal_length_mm: float | None
    focal_length_35mm: int | None
    width: int
    height: int


def extract(raw: bytes) -> ExifCovariates:
    """Read device/lens covariates from an image's EXIF tags.

    Missing or absent EXIF data yields ``None`` fields rather than
    raising — most ad-hoc uploads won't have it, and that's expected.
    """
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        exif = image.getexif()

        make = _as_str(exif.get(_IFD_MAKE))
        model = _as_str(exif.get(_IFD_MODEL))

        focal_length_mm: float | None = None
        focal_length_35mm: int | None = None
        try:
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        except (KeyError, ValueError):
            exif_ifd = {}
        if exif_ifd:
            focal_length_mm = _as_float(exif_ifd.get(_FOCAL_LENGTH))
            focal_35mm_raw = exif_ifd.get(_FOCAL_LENGTH_35MM)
            focal_length_35mm = int(focal_35mm_raw) if focal_35mm_raw is not None else None

    return ExifCovariates(
        make=make,
        model=model,
        focal_length_mm=focal_length_mm,
        focal_length_35mm=focal_length_35mm,
        width=width,
        height=height,
    )


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\x00")
    return text or None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]  # IFDRational supports float()
