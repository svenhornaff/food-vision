"""Pure fruit-mass/kcal formulas for strategy S2 (and baseline B1).

docs/dev/pre-study-web-ui.md §4.3: S2 asks the model for ``length_cm`` and
``max_diameter_cm`` only; the edible-gram prediction is computed here, not
by the model. B1 runs the same formulas on *true* tape measurements as a
geometry-only ceiling for S2.

No fitted constant is baked in here. ``banana_volume_g``'s shape constant
and every fruit's density/edible-ratio are fitted on the dev split by
``prestudy fit`` (M2, not yet implemented) and passed in by the caller —
hard-coding a guessed value would make a scientifically material claim
this module has no business making. Callers without a fit yet (tests,
exploration) pass their own placeholder values explicitly.

No ranges, no low/mid/high — Appendix A metrics are point estimates only
(concept §A.6).
"""

from __future__ import annotations

import math

__all__ = ["banana_volume_g", "ellipsoid_mass_g", "kcal_from_edible_g"]


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number, got {value!r}.")


def banana_volume_g(
    length_cm: float,
    diameter_cm: float,
    shape_constant: float,
    density_g_cm3: float,
) -> float:
    """Banana mass from length and max diameter.

    Approximates the banana as a curved cylinder: volume ≈
    ``shape_constant * length_cm * diameter_cm**2``, mass = volume ×
    density. ``shape_constant`` is dimensionless and absorbs the
    deviation from a true cylinder (``pi/4`` would be a right cylinder);
    it is fitted on dev specimens, never guessed here.

    Raises:
        ValueError: any input is not finite and positive.
    """
    _require_positive("length_cm", length_cm)
    _require_positive("diameter_cm", diameter_cm)
    _require_positive("shape_constant", shape_constant)
    _require_positive("density_g_cm3", density_g_cm3)
    volume_cm3 = shape_constant * length_cm * diameter_cm**2
    return volume_cm3 * density_g_cm3


def ellipsoid_mass_g(
    length_cm: float,
    diameter_cm: float,
    density_g_cm3: float,
    edible_ratio: float,
) -> float:
    """Round-fruit (apple/orange/pear/kiwi/mandarin) mass.

    Models the fruit as a prolate spheroid with major axis ``length_cm``
    and equatorial diameter ``diameter_cm``:
    ``volume = (pi / 6) * length_cm * diameter_cm**2``. ``edible_ratio``
    is the fitted fraction of whole mass that's edible (removing peel,
    core, stem) for the fruit type, in ``(0, 1]``.

    Raises:
        ValueError: any length/diameter/density input is not finite and
            positive, or ``edible_ratio`` is not in ``(0, 1]``.
    """
    _require_positive("length_cm", length_cm)
    _require_positive("diameter_cm", diameter_cm)
    _require_positive("density_g_cm3", density_g_cm3)
    if not math.isfinite(edible_ratio) or not (0 < edible_ratio <= 1):
        raise ValueError(f"edible_ratio must be in (0, 1], got {edible_ratio!r}.")
    volume_cm3 = (math.pi / 6) * length_cm * diameter_cm**2
    return volume_cm3 * density_g_cm3 * edible_ratio


def kcal_from_edible_g(edible_g: float, kcal_100g: float) -> float:
    """kcal for a predicted/true edible mass, per BLS ``kcal_100g``.

    Raises:
        ValueError: ``edible_g`` or ``kcal_100g`` is negative, or either
            is not finite.
    """
    if not math.isfinite(edible_g) or edible_g < 0:
        raise ValueError(f"edible_g must be finite and non-negative, got {edible_g!r}.")
    if not math.isfinite(kcal_100g) or kcal_100g < 0:
        raise ValueError(f"kcal_100g must be finite and non-negative, got {kcal_100g!r}.")
    return edible_g * kcal_100g / 100
