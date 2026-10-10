"""Strict output schemas, one per strategy (docs/dev/pre-study.md §4.1).

S1/S3 ask for ``mass_g`` directly (the prompt text, not this schema,
defines *which* mass — edible for the own-photo ``fruit_*`` prompts,
whole-fruit for the ``ecustfd_*`` prompts, since ECUSTFD has no edible-mass
ground truth); S2 asks for ``length_cm`` and ``max_diameter_cm`` (the
calculator does the rest). Every schema puts an ``observations`` free-text
field *before* the numeric answer — in-schema chain-of-thought without
spending reasoning tokens. No confidence field.

Renamed from ``edible_g`` (pre-study.md §4.1): the lean pre-study runs on
ECUSTFD, which only has whole-fruit weight, so a field named ``edible_g``
would misdescribe what's actually being asked for and measured there.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from food_vision.proxy.openrouter import JsonSchemaFormat

__all__ = [
    "ObservationStrategy",
    "StructuredOutputMode",
    "ObservationValidationError",
    "schema_for",
    "validate_observation",
]


class ObservationStrategy(StrEnum):
    """Which fields the model is asked for (pre-study.md §3)."""

    S1 = "S1"  # direct mass_g estimate
    S2 = "S2"  # length_cm + max_diameter_cm; calculator derives mass
    S3 = "S3"  # direct mass_g estimate, prompt states a dev-derived prior
    #: Not in pre-study.md's original §3 — added for the "E1" experiment
    #: in bench/reports/prestudy-lean/review.md §5: ask the model for
    #: normalised [0,1] bounding boxes of the coin and the fruit instead
    #: of a mass estimate; bench/scripts/bbox_vlm_eval.py does the
    #: coin-scaled geometry → mass_g conversion (same formulas as
    #: bbox_oracle.py), not this module.
    BBOX = "BBOX"


class StructuredOutputMode(StrEnum):
    """How structured output is requested (§7.3). Per-model capability
    resolution (which models support ``json_schema`` strict mode) is the
    ``prestudy/configs.py`` capability table — M2, not this module."""

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class ObservationValidationError(ValueError):
    """Raised by :func:`validate_observation` when a parsed value doesn't
    match its strategy's schema — the adapter classifies this as the
    ``invalid`` outcome (§7.2)."""


_BBOX_FIELDS = (
    "coin_xmin",
    "coin_ymin",
    "coin_xmax",
    "coin_ymax",
    "fruit_xmin",
    "fruit_ymin",
    "fruit_xmax",
    "fruit_ymax",
)

_GRAM_FIELDS: dict[ObservationStrategy, tuple[str, ...]] = {
    ObservationStrategy.S1: ("observations", "mass_g"),
    ObservationStrategy.S3: ("observations", "mass_g"),
    ObservationStrategy.S2: ("observations", "length_cm", "max_diameter_cm"),
    ObservationStrategy.BBOX: ("observations", *_BBOX_FIELDS),
}

_NUMERIC_FIELDS = frozenset({"mass_g", "length_cm", "max_diameter_cm"})
#: Normalised-coordinate fields (§BBOX): range is [0, 1], not "any
#: non-negative number" like the gram/cm fields above.
_UNIT_INTERVAL_FIELDS = frozenset(_BBOX_FIELDS)
#: (xmin, ymin, xmax, ymax) tuples that must each describe a positive-area
#: box — checked after the per-field range check.
_BBOX_BOXES = (
    ("coin_xmin", "coin_ymin", "coin_xmax", "coin_ymax"),
    ("fruit_xmin", "fruit_ymin", "fruit_xmax", "fruit_ymax"),
)


def schema_for(strategy: ObservationStrategy) -> JsonSchemaFormat:
    """The strict ``json_schema`` response format for a strategy."""
    fields = _GRAM_FIELDS[strategy]
    properties: dict[str, Any] = {"observations": {"type": "string"}}
    for field_name in fields[1:]:
        if field_name in _UNIT_INTERVAL_FIELDS:
            properties[field_name] = {"type": "number", "minimum": 0, "maximum": 1}
        else:
            properties[field_name] = {"type": "number"}
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }
    return JsonSchemaFormat(name=f"fruit_{strategy.value.lower()}_observation", schema=schema)


def validate_observation(strategy: ObservationStrategy, value: Any) -> dict[str, Any]:
    """Validate a parsed JSON value against ``strategy``'s schema.

    Raises:
        ObservationValidationError: ``value`` isn't a dict, is missing a
            required field, has an extra field, or a numeric field isn't
            a finite non-negative number.
    """
    if not isinstance(value, dict):
        raise ObservationValidationError(f"Expected a JSON object, got {type(value).__name__}.")

    fields = _GRAM_FIELDS[strategy]
    missing = [name for name in fields if name not in value]
    if missing:
        raise ObservationValidationError(f"Missing required field(s): {missing}.")

    extra = set(value) - set(fields)
    if extra:
        raise ObservationValidationError(f"Unexpected field(s): {sorted(extra)}.")

    if not isinstance(value["observations"], str):
        raise ObservationValidationError("'observations' must be a string.")

    result: dict[str, Any] = {"observations": value["observations"]}
    for field_name in fields:
        if field_name not in _NUMERIC_FIELDS and field_name not in _UNIT_INTERVAL_FIELDS:
            continue
        raw = value[field_name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ObservationValidationError(f"'{field_name}' must be a number.")
        number = float(raw)
        if field_name in _UNIT_INTERVAL_FIELDS:
            if not math.isfinite(number) or not (0.0 <= number <= 1.0):
                raise ObservationValidationError(
                    f"'{field_name}' must be a finite number in [0, 1], got {raw!r}."
                )
        elif not math.isfinite(number) or number < 0:
            raise ObservationValidationError(
                f"'{field_name}' must be a finite, non-negative number, got {raw!r}."
            )
        result[field_name] = number

    if strategy is ObservationStrategy.BBOX:
        for xmin_name, ymin_name, xmax_name, ymax_name in _BBOX_BOXES:
            if result[xmax_name] <= result[xmin_name] or result[ymax_name] <= result[ymin_name]:
                raise ObservationValidationError(
                    f"Degenerate or inverted box: {xmin_name}/{ymin_name}/"
                    f"{xmax_name}/{ymax_name} = {result[xmin_name]}, {result[ymin_name]}, "
                    f"{result[xmax_name]}, {result[ymax_name]}."
                )

    return result
