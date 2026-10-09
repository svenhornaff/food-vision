"""Build one observation request and classify its outcome.

docs/dev/pre-study.md §4.1. Thin: all transport/retry/telemetry is
``proxy.ModelProvider``'s job; this module only knows prompts, schemas,
and how to turn a :class:`~food_vision.proxy.openrouter.CompletionResult`
into a classified :class:`Observation`.

M1 scope note: the full ``observe()`` signature in §7.3 takes a
``RunConfig`` from ``prestudy/configs.py`` (M2 — content-addressed,
DB-backed, with a per-model capability table). That module doesn't exist
yet, so this is a self-contained, locally-defined substitute
(:class:`ObservationConfig`) carrying only what M1 needs. When M2 lands,
``configs.RunConfig`` replaces ``ObservationConfig`` as the caller-facing
type; ``observe()``'s message-building, parsing and outcome
classification are expected to carry over unchanged — only where the
config/capability-mode-per-model comes from changes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from food_vision.proxy.openrouter import (
    CompletionResult,
    JsonObjectFormat,
    ModelProvider,
    RoutingMode,
    RoutingPolicy,
)
from food_vision.utils.log_factory import get_logger

from .schema import ObservationStrategy, ObservationValidationError, StructuredOutputMode
from .schema import schema_for as _schema_for
from .schema import validate_observation as _validate_observation

logger = get_logger(__name__)

__all__ = [
    "ObservationConfig",
    "Routing",
    "ObservationOutcome",
    "Observation",
    "observe",
    "render_prompt",
]


class ObservationOutcome(StrEnum):
    """Per §7.2's outcome taxonomy. ``error`` (transport failure after the
    proxy's own retries) isn't produced here — it surfaces as the proxy's
    :class:`~food_vision.proxy.openrouter.OpenRouterError`, which the
    caller (the M2 executor) catches and classifies itself."""

    OK = "ok"
    INVALID = "invalid"
    REFUSED = "refused"
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class ObservationConfig:
    """M1-local substitute for M2's ``prestudy.configs.RunConfig``. See
    the module docstring for what changes when that lands."""

    strategy: ObservationStrategy
    #: Prompt filename prefix (pre-study.md §4.1): ``"fruit"`` for the
    #: own-photo transfer-check prompts (edible mass, ID-1 card), or
    #: ``"ecustfd"`` for the lean pre-study's prompts (whole mass, 25mm
    #: coin). Resolves to ``f"{prompt_set}_{strategy}_v1_en.md"``.
    prompt_set: str = "fruit"
    #: One prompt-image per view, in order; length must match ``images``
    #: passed to :func:`observe`. ``("primary",)`` or ``("c1", "c2")``.
    views: tuple[str, ...] = ("primary",)
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA
    #: Forwarded to ``provider.complete()`` as-is; ``None`` lets the
    #: provider/model default apply. The lean pre-study fixes this at
    #: 2048 (§3) so a reasoning model exhausting its budget is visible
    #: as ``truncated`` rather than silently retried with more tokens.
    max_tokens: int | None = None
    #: Disabled in protocol arms (§4.2); exploratory configs may enable
    #: one repair round once M2's executor implements it. Not implemented
    #: here — ``observe()`` rejects ``repair=True`.
    repair: bool = False
    #: Required, and only used, for ``ObservationStrategy.S3`` (the
    #: dev-derived prior stated in the prompt, §4.3). ``None`` for S1/S2.
    prior_low_g: float | None = None
    prior_high_g: float | None = None


@dataclass(frozen=True)
class Routing:
    """Model + routing for one observation call."""

    model: str
    mode: RoutingMode = RoutingMode.PRODUCTION
    policy: RoutingPolicy | None = None


@dataclass(frozen=True)
class Observation:
    outcome: ObservationOutcome
    #: The validated fields (§ schema.py) when ``outcome`` is ``OK``;
    #: ``None`` otherwise.
    values: dict[str, Any] | None
    completion: CompletionResult
    error: str | None = field(default=None)


#: finish_reason OpenRouter uses for a length-truncated response.
_FINISH_REASON_LENGTH = "length"


def observe(
    provider: ModelProvider,
    *,
    images: list[bytes],
    config: ObservationConfig,
    routing: Routing,
) -> Observation:
    """Run one observation call and classify its outcome.

    Raises:
        ValueError: ``len(images) != len(config.views)``, ``config.repair``
            is ``True`` (not implemented at this layer), or strategy ``S3``
            is used without both prior bounds set.
        OpenRouterError: propagated from the provider on a transport
            failure after its own retries — not caught here; the caller
            classifies it as the M2 executor's ``error`` outcome.
    """
    if len(images) != len(config.views):
        raise ValueError(
            f"Expected {len(config.views)} image(s) for views={config.views!r}, got {len(images)}."
        )
    if config.repair:
        raise ValueError("config.repair=True is not implemented by observation.adapter.observe().")

    prompt_text = render_prompt(config)
    message = _build_message(prompt_text, images)

    response_format: JsonObjectFormat | Any
    if config.structured_output_mode is StructuredOutputMode.JSON_SCHEMA:
        response_format = _schema_for(config.strategy)
    else:
        response_format = JsonObjectFormat()

    completion = provider.complete(
        messages=[message],
        response_format=response_format,
        model=routing.model,
        routing_mode=routing.mode,
        routing_policy=routing.policy,
        max_tokens=config.max_tokens,
    )

    return _classify(config, completion)


def render_prompt(config: ObservationConfig) -> str:
    """The exact prompt text :func:`observe` will send for ``config``.

    Public so callers that need the prompt's hash for provenance (e.g.
    ``prestudy.run``'s ``prompt_sha256`` field) don't have to duplicate
    prompt-loading/S3-prior-formatting logic.

    Raises:
        ValueError: strategy ``S3`` without both prior bounds set.
        FileNotFoundError: no prompt file for ``(config.prompt_set,
            config.strategy)``.
    """
    text = _load_prompt_text(config.prompt_set, config.strategy)
    if config.strategy is not ObservationStrategy.S3:
        return text
    if config.prior_low_g is None or config.prior_high_g is None:
        raise ValueError("ObservationStrategy.S3 requires both prior_low_g and prior_high_g.")
    return text.format(prior_low_g=config.prior_low_g, prior_high_g=config.prior_high_g)


def _load_prompt_text(prompt_set: str, strategy: ObservationStrategy) -> str:
    """Load ``{prompt_set}_{strategy}_v1_en.md`` as package data.

    Raises:
        FileNotFoundError: no prompt file exists for this
            ``(prompt_set, strategy)`` pair (e.g. ``prompt_set="ecustfd"``
            with ``strategy=S2``, which pre-study.md §3 explicitly drops).
    """
    from importlib import resources

    filename = f"{prompt_set}_{strategy.value.lower()}_v1_en.md"
    resource = resources.files("food_vision.observation.prompts").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(
            f"No prompt file {filename!r} for prompt_set={prompt_set!r}, strategy={strategy!r}."
        )
    return resource.read_text(encoding="utf-8")


def _build_message(prompt_text: str, images: list[bytes]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_bytes in images:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return {"role": "user", "content": content}


def _classify(config: ObservationConfig, completion: CompletionResult) -> Observation:
    if _is_refused(completion):
        return Observation(ObservationOutcome.REFUSED, None, completion, error="refused")

    if _is_truncated(completion):
        return Observation(ObservationOutcome.TRUNCATED, None, completion, error="truncated")

    if config.structured_output_mode is StructuredOutputMode.JSON_SCHEMA:
        candidate = completion.parsed
    else:
        candidate = _try_parse_json_object_mode(completion.content)

    if candidate is None:
        return Observation(
            ObservationOutcome.INVALID, None, completion, error="response was not valid JSON"
        )

    try:
        values = _validate_observation(config.strategy, candidate)
    except ObservationValidationError as exc:
        return Observation(ObservationOutcome.INVALID, None, completion, error=str(exc))

    return Observation(ObservationOutcome.OK, values, completion, error=None)


def _is_refused(completion: CompletionResult) -> bool:
    native = (completion.native_finish_reason or "").lower()
    return "content_filter" in native or "refus" in native


def _is_truncated(completion: CompletionResult) -> bool:
    """``finish_reason: length``. Empty content with reasoning tokens
    spent means the reasoning budget was exhausted (§7.2) — still
    classified as truncated either way."""
    return completion.finish_reason == _FINISH_REASON_LENGTH


def _try_parse_json_object_mode(content: str) -> dict[str, Any] | None:
    import json

    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
