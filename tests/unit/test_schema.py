"""observation.schema: strict per-strategy JSON schemas and validation."""

from __future__ import annotations

import pytest

from food_vision.observation.schema import (
    ObservationStrategy,
    ObservationValidationError,
    schema_for,
    validate_observation,
)


@pytest.mark.parametrize("strategy", list(ObservationStrategy))
def test_schema_requires_observations_field_first(strategy: ObservationStrategy) -> None:
    fmt = schema_for(strategy)
    schema = fmt.schema
    assert schema["additionalProperties"] is False
    assert schema["required"][0] == "observations"
    assert next(iter(schema["properties"])) == "observations"


def test_schema_s1_s3_require_mass_g() -> None:
    for strategy in (ObservationStrategy.S1, ObservationStrategy.S3):
        schema = schema_for(strategy).schema
        assert schema["required"] == ["observations", "mass_g"]


def test_schema_s2_requires_dimensions() -> None:
    schema = schema_for(ObservationStrategy.S2).schema
    assert schema["required"] == ["observations", "length_cm", "max_diameter_cm"]


def test_schema_bbox_requires_eight_box_fields() -> None:
    schema = schema_for(ObservationStrategy.BBOX).schema
    assert schema["required"] == [
        "observations",
        "coin_xmin",
        "coin_ymin",
        "coin_xmax",
        "coin_ymax",
        "fruit_xmin",
        "fruit_ymin",
        "fruit_xmax",
        "fruit_ymax",
    ]
    assert schema["properties"]["coin_xmin"] == {"type": "number", "minimum": 0, "maximum": 1}


def test_schema_has_no_confidence_field() -> None:
    for strategy in ObservationStrategy:
        assert "confidence" not in schema_for(strategy).schema["properties"]


def test_validate_observation_accepts_valid_s1() -> None:
    result = validate_observation(
        ObservationStrategy.S1, {"observations": "a ripe banana", "mass_g": 118.4}
    )
    assert result == {"observations": "a ripe banana", "mass_g": 118.4}


def test_validate_observation_accepts_valid_s2() -> None:
    result = validate_observation(
        ObservationStrategy.S2,
        {"observations": "banana, card visible", "length_cm": 19.8, "max_diameter_cm": 3.6},
    )
    assert result["length_cm"] == 19.8
    assert result["max_diameter_cm"] == 3.6


def _bbox_value(**overrides: float) -> dict[str, object]:
    base = {
        "observations": "a kiwi beside a coin",
        "coin_xmin": 0.10,
        "coin_ymin": 0.80,
        "coin_xmax": 0.20,
        "coin_ymax": 0.90,
        "fruit_xmin": 0.30,
        "fruit_ymin": 0.20,
        "fruit_xmax": 0.70,
        "fruit_ymax": 0.60,
    }
    base.update(overrides)
    return base


def test_validate_observation_accepts_valid_bbox() -> None:
    result = validate_observation(ObservationStrategy.BBOX, _bbox_value())
    assert result["coin_xmin"] == 0.10
    assert result["fruit_ymax"] == 0.60


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, float("nan"), float("inf")])
def test_validate_observation_rejects_out_of_unit_interval(bad_value: float) -> None:
    with pytest.raises(ObservationValidationError, match=r"\[0, 1\]"):
        validate_observation(ObservationStrategy.BBOX, _bbox_value(coin_xmin=bad_value))


def test_validate_observation_rejects_degenerate_box() -> None:
    """xmax <= xmin: zero or negative width."""
    with pytest.raises(ObservationValidationError, match="Degenerate"):
        validate_observation(
            ObservationStrategy.BBOX, _bbox_value(fruit_xmin=0.70, fruit_xmax=0.70)
        )


def test_validate_observation_rejects_inverted_box() -> None:
    """ymax < ymin: boxes the model reported back-to-front."""
    with pytest.raises(ObservationValidationError, match="Degenerate"):
        validate_observation(ObservationStrategy.BBOX, _bbox_value(coin_ymin=0.90, coin_ymax=0.80))


def test_validate_observation_rejects_non_dict() -> None:
    with pytest.raises(ObservationValidationError):
        validate_observation(ObservationStrategy.S1, "not a dict")


def test_validate_observation_rejects_missing_field() -> None:
    with pytest.raises(ObservationValidationError, match="Missing"):
        validate_observation(ObservationStrategy.S1, {"observations": "x"})


def test_validate_observation_rejects_extra_field() -> None:
    with pytest.raises(ObservationValidationError, match="Unexpected"):
        validate_observation(
            ObservationStrategy.S1,
            {"observations": "x", "mass_g": 100.0, "confidence": 0.9},
        )


@pytest.mark.parametrize("bad_value", [True, "118", None, float("nan"), float("inf"), -1.0])
def test_validate_observation_rejects_invalid_numeric(bad_value: object) -> None:
    with pytest.raises(ObservationValidationError):
        validate_observation(ObservationStrategy.S1, {"observations": "x", "mass_g": bad_value})


def test_validate_observation_rejects_non_string_observations() -> None:
    with pytest.raises(ObservationValidationError, match="observations"):
        validate_observation(ObservationStrategy.S1, {"observations": 7, "mass_g": 100.0})
