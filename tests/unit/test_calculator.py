"""domain.calculator: pure S2/B1 formulas.

No fitted constant is hard-coded in the module under test — these tests
pass their own placeholder shape constants/densities/ratios and check the
formula's math properties (positivity, monotonicity, unit sanity), not any
specific fitted value (that's M2's job, not this module's).
"""

from __future__ import annotations

import math

import pytest

from food_vision.domain.calculator import (
    banana_volume_g,
    ellipsoid_mass_g,
    kcal_from_edible_g,
)


class TestBananaVolumeG:
    def test_positive_for_valid_inputs(self) -> None:
        mass = banana_volume_g(
            length_cm=20.0, diameter_cm=3.5, shape_constant=0.5, density_g_cm3=0.9
        )
        assert mass > 0

    def test_monotone_in_length(self) -> None:
        short = banana_volume_g(15.0, 3.5, 0.5, 0.9)
        long = banana_volume_g(25.0, 3.5, 0.5, 0.9)
        assert long > short

    def test_monotone_in_diameter(self) -> None:
        thin = banana_volume_g(20.0, 2.5, 0.5, 0.9)
        thick = banana_volume_g(20.0, 4.5, 0.5, 0.9)
        assert thick > thin

    def test_scales_linearly_with_density(self) -> None:
        low = banana_volume_g(20.0, 3.5, 0.5, 0.9)
        high = banana_volume_g(20.0, 3.5, 0.5, 1.8)
        assert math.isclose(high, 2 * low)

    @pytest.mark.parametrize(
        ("length_cm", "diameter_cm", "shape_constant", "density_g_cm3"),
        [
            (0.0, 3.5, 0.5, 0.9),
            (-1.0, 3.5, 0.5, 0.9),
            (20.0, 0.0, 0.5, 0.9),
            (20.0, 3.5, 0.0, 0.9),
            (20.0, 3.5, 0.5, 0.0),
            (float("nan"), 3.5, 0.5, 0.9),
            (float("inf"), 3.5, 0.5, 0.9),
        ],
    )
    def test_rejects_non_positive_or_non_finite(
        self, length_cm: float, diameter_cm: float, shape_constant: float, density_g_cm3: float
    ) -> None:
        with pytest.raises(ValueError):
            banana_volume_g(length_cm, diameter_cm, shape_constant, density_g_cm3)


class TestEllipsoidMassG:
    def test_positive_for_valid_inputs(self) -> None:
        mass = ellipsoid_mass_g(
            length_cm=7.0, diameter_cm=7.0, density_g_cm3=0.85, edible_ratio=0.9
        )
        assert mass > 0

    def test_monotone_in_length_and_diameter(self) -> None:
        base = ellipsoid_mass_g(7.0, 7.0, 0.85, 0.9)
        longer = ellipsoid_mass_g(9.0, 7.0, 0.85, 0.9)
        wider = ellipsoid_mass_g(7.0, 9.0, 0.85, 0.9)
        assert longer > base
        assert wider > base

    def test_scales_linearly_with_edible_ratio(self) -> None:
        full = ellipsoid_mass_g(7.0, 7.0, 0.85, 1.0)
        half = ellipsoid_mass_g(7.0, 7.0, 0.85, 0.5)
        assert math.isclose(full, 2 * half)

    def test_matches_known_sphere_volume_formula(self) -> None:
        """length == diameter degenerates to a sphere: V = (pi/6) * D**3."""
        diameter = 6.0
        mass = ellipsoid_mass_g(diameter, diameter, density_g_cm3=1.0, edible_ratio=1.0)
        expected_volume = (math.pi / 6) * diameter**3
        assert math.isclose(mass, expected_volume)

    @pytest.mark.parametrize(
        ("length_cm", "diameter_cm", "density_g_cm3", "edible_ratio"),
        [
            (0.0, 7.0, 0.85, 0.9),
            (7.0, 0.0, 0.85, 0.9),
            (7.0, 7.0, 0.0, 0.9),
            (7.0, 7.0, 0.85, 0.0),
            (7.0, 7.0, 0.85, 1.1),
            (7.0, 7.0, 0.85, float("nan")),
        ],
    )
    def test_rejects_invalid_inputs(
        self, length_cm: float, diameter_cm: float, density_g_cm3: float, edible_ratio: float
    ) -> None:
        with pytest.raises(ValueError):
            ellipsoid_mass_g(length_cm, diameter_cm, density_g_cm3, edible_ratio)


class TestKcalFromEdibleG:
    def test_known_value(self) -> None:
        assert kcal_from_edible_g(edible_g=200.0, kcal_100g=52.0) == pytest.approx(104.0)

    def test_zero_grams_is_zero_kcal(self) -> None:
        assert kcal_from_edible_g(0.0, 52.0) == 0.0

    def test_monotone_in_edible_g(self) -> None:
        low = kcal_from_edible_g(100.0, 52.0)
        high = kcal_from_edible_g(200.0, 52.0)
        assert high > low

    @pytest.mark.parametrize(
        ("edible_g", "kcal_100g"),
        [(-1.0, 52.0), (100.0, -1.0), (float("nan"), 52.0), (100.0, float("inf"))],
    )
    def test_rejects_invalid_inputs(self, edible_g: float, kcal_100g: float) -> None:
        with pytest.raises(ValueError):
            kcal_from_edible_g(edible_g, kcal_100g)
