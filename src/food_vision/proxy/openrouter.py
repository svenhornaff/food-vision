"""Thin OpenRouter chat-completions adapter.

Direct ``openai`` SDK usage against OpenRouter's OpenAI-compatible endpoint
via ``base_url`` override — no agent framework, per
docs/dev/food-vision-concept.md §5 ("AI gateway: OpenRouter via `openai`
client with `base_url` override; thin adapter; no agent framework") and §9
("OpenRouter Integration").

This module only owns the transport/request layer: building the request
(structured output, provider routing, determinism settings), bounded
retries, timeouts, and usage/cost telemetry. It knows nothing about meal
observation, prompts, or schemas — those are a later phase
(``observation/adapter.py`` in the concept's package layout) and are out of
scope here.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import httpx
from openai import APIStatusError, OpenAI

from food_vision.config.settings import Settings, get_settings
from food_vision.utils.log_factory import get_logger

logger = get_logger(__name__)

#: HTTP statuses worth retrying per concept §9 ("retry only 429/5xx").
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Valid keys for the completion-length cap. Some GPT-5-family endpoints
#: reject "max_tokens" and require "max_completion_tokens" instead.
_VALID_MAX_TOKENS_PARAMS = frozenset({"max_tokens", "max_completion_tokens"})


class RoutingMode(StrEnum):
    """Which OpenRouter ``provider`` routing policy to apply.

    - ``PRODUCTION``: fallbacks among the approved provider list.
    - ``BENCHMARK``: exactly one pinned provider, ``allow_fallbacks: false``
      — reproducibility for benchmark runs. Quantization is an *optional*
      extra pin (``RoutingPolicy.quantizations``): several benchmark
      candidates (e.g. Gemini, GPT, Claude endpoints on OpenRouter) don't
      publish quantization labels at all, so requiring one would silently
      exclude them.
    """

    PRODUCTION = "production"
    BENCHMARK = "benchmark"


_VALID_DATA_COLLECTION = frozenset({"allow", "deny"})


class OpenRouterError(RuntimeError):
    """Raised for a non-retryable OpenRouter failure or exhausted retries."""


@dataclass(frozen=True)
class RoutingPolicy:
    """Per-call override of OpenRouter provider-routing knobs.

    Every field left ``None`` falls back to the configured :class:`Settings`
    default. This exists so a caller (e.g. the pre-study benchmark harness)
    can vary privacy/routing per run — not just via process-wide env vars —
    and record the *effective* policy it used alongside the result, rather
    than one env-file edit and restart per candidate model/provider
    combination.

    ``provider_pin`` is required when used with ``RoutingMode.BENCHMARK``;
    ``quantizations`` is optional there (see :class:`RoutingMode`).
    """

    provider_pin: str | None = None
    quantizations: Sequence[str] | None = None
    zdr: bool | None = None
    data_collection: str | None = None

    def __post_init__(self) -> None:
        if self.data_collection is not None and self.data_collection not in _VALID_DATA_COLLECTION:
            raise OpenRouterError(
                f"RoutingPolicy.data_collection must be one of {sorted(_VALID_DATA_COLLECTION)}, "
                f"got {self.data_collection!r}."
            )


@dataclass(frozen=True)
class JsonSchemaFormat:
    """A ``response_format: json_schema`` request, caller-supplied.

    The schema itself (and whatever it models) is the caller's concern —
    this adapter does not define or version any observation schema.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True

    def as_response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": self.name, "strict": self.strict, "schema": self.schema},
        }


@dataclass(frozen=True)
class JsonObjectFormat:
    """A ``response_format: json_object`` request.

    Used instead of :class:`JsonSchemaFormat` for models that don't support
    strict JSON-schema structured output (pre-study §7.3/§7.4) — the caller
    is responsible for appending the schema to the prompt text and
    validating the parsed result itself.
    """

    def as_response_format(self) -> dict[str, Any]:
        return {"type": "json_object"}


@dataclass(frozen=True)
class Usage:
    """Provider-reported usage/cost. ``cost_is_estimate`` is ``True`` only
    when OpenRouter didn't report a cost and none can be derived — the
    caller must not silently treat a missing cost as free (concept §9:
    "Provider-reported usage and cost logged; estimates marked.").
    """

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    cost_is_estimate: bool


@dataclass(frozen=True)
class CompletionResult:
    content: str
    parsed: dict[str, Any] | None
    model: str
    provider: str | None
    latency_ms: float
    usage: Usage
    finish_reason: str | None
    attempts: int
    #: The OpenRouter ``provider`` block actually sent (pin, quantizations,
    #: zdr, data_collection, allow_fallbacks) — a caller that varies routing
    #: per call (e.g. the pre-study harness) records this alongside the
    #: result instead of having to reconstruct it from inputs.
    effective_routing_policy: dict[str, Any]
    #: Temperature actually sent, after ``Settings`` fallback -- ``None``
    #: when ``send_temperature=False`` omitted it entirely (some next-gen
    #: reasoning models, e.g. Claude Sonnet 5 / GPT-5 on OpenRouter, don't
    #: list ``temperature`` in any endpoint's ``supported_parameters`` at
    #: all; sending it with ``require_parameters: true`` filters out every
    #: endpoint regardless of provider pin).
    temperature: float | None
    seed: int | None
    #: Reproducibility/telemetry fields the pre-study harness needs
    #: (docs/dev/pre-study-web-ui.md §7.4). Defensive on extraction: mocked
    #: or minimal SDK responses that don't set these attributes yield
    #: ``None`` rather than raising.
    generation_id: str | None = None
    model_resolved: str | None = None
    system_fingerprint: str | None = None
    native_finish_reason: str | None = None
    reasoning_text: str | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    #: Full response body, best-effort JSON-able dict. Never logged — may
    #: contain model-generated text. Empty dict if it can't be derived.
    raw: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    """Swappable chat/vision completion provider.

    Kept minimal on purpose: this is the seam the concept's
    ``observation/adapter.py`` Protocol will eventually sit behind, but no
    observation-specific method is added until that phase exists.
    """

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_format: JsonSchemaFormat | JsonObjectFormat | None = None,
        model: str | None = None,
        routing_mode: RoutingMode = RoutingMode.PRODUCTION,
        routing_policy: RoutingPolicy | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_tokens_param: str = "max_tokens",
        reasoning: dict[str, Any] | None = None,
        send_temperature: bool = True,
    ) -> CompletionResult: ...


class OpenRouterClient:
    """Direct OpenRouter client built on the ``openai`` SDK.

    The SDK's own retry mechanism is disabled (``max_retries=0``); this
    class owns retries itself so the 429/5xx-only, max-2, jittered policy
    from concept §9 is explicit and testable rather than inherited SDK
    defaults.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.OPENROUTER_API_KEY:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        timeout = httpx.Timeout(
            connect=self._settings.OPENROUTER_CONNECT_TIMEOUT_S,
            read=self._settings.OPENROUTER_TOTAL_TIMEOUT_S,
            write=self._settings.OPENROUTER_TOTAL_TIMEOUT_S,
            pool=self._settings.OPENROUTER_TOTAL_TIMEOUT_S,
        )
        self._client = OpenAI(
            base_url=self._settings.OPENROUTER_API_URL,
            api_key=self._settings.OPENROUTER_API_KEY,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        self._sleep = sleep
        self._random_fn = random_fn

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_format: JsonSchemaFormat | JsonObjectFormat | None = None,
        model: str | None = None,
        routing_mode: RoutingMode = RoutingMode.PRODUCTION,
        routing_policy: RoutingPolicy | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_tokens_param: str = "max_tokens",
        reasoning: dict[str, Any] | None = None,
        send_temperature: bool = True,
    ) -> CompletionResult:
        """Run one chat-completion request against OpenRouter.

        ``routing_policy``, ``temperature`` and ``seed`` override the
        configured :class:`Settings` defaults for this call only — the
        pre-study benchmark harness varies these per run; production code
        can omit them and get the process-wide defaults. ``top_p``,
        ``max_tokens`` and ``reasoning`` (OpenRouter's reasoning-effort/
        budget block, forwarded as-is) are only sent when given — most
        callers don't need them.

        ``send_temperature=False`` omits ``temperature`` from the request
        entirely rather than sending a value — needed for models whose
        *every* OpenRouter endpoint omits ``temperature`` from
        ``supported_parameters`` (discovered live for
        ``anthropic/claude-sonnet-5`` and ``openai/gpt-5``: with
        ``require_parameters: true``, sending an unsupported param filters
        out every candidate endpoint regardless of provider pin, raising a
        404 with no provider available). ``max_tokens_param`` lets a caller
        send the cap under a different key (e.g. ``"max_completion_tokens"``
        for some GPT-5-family endpoints) instead of ``"max_tokens"``.

        Raises:
            OpenRouterError: model not on the allowlist, benchmark routing
                missing its required provider pin, ``max_tokens_param`` is
                not one of ``"max_tokens"``/``"max_completion_tokens"``, or
                the request failed (non-retryable status, or retries
                exhausted).
        """
        model_id = self._resolve_model(model)
        policy = routing_policy or RoutingPolicy()
        provider_payload = self._build_provider_payload(routing_mode, policy)
        effective_temperature: float | None = (
            temperature if temperature is not None else self._settings.OPENROUTER_TEMPERATURE
        )
        effective_seed = seed if seed is not None else self._settings.OPENROUTER_SEED
        if max_tokens_param not in _VALID_MAX_TOKENS_PARAMS:
            raise OpenRouterError(
                f"max_tokens_param must be one of {sorted(_VALID_MAX_TOKENS_PARAMS)}, "
                f"got {max_tokens_param!r}."
            )

        extra_body: dict[str, Any] = {"provider": provider_payload, "usage": {"include": True}}
        if reasoning is not None:
            extra_body["reasoning"] = reasoning

        request_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": list(messages),
            # ``usage.include`` asks OpenRouter to report actual provider
            # cost on this response; without it ``usage.cost`` is absent on
            # most requests and every row would be flagged
            # ``cost_is_estimate=True`` (concept §9: "Provider-reported usage
            # and cost logged; estimates marked." — this is what makes a
            # *reported*, not estimated, cost possible at all).
            "extra_body": extra_body,
        }
        if send_temperature:
            request_kwargs["temperature"] = effective_temperature
        else:
            effective_temperature = None
        if effective_seed is not None:
            request_kwargs["seed"] = effective_seed
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if max_tokens is not None:
            request_kwargs[max_tokens_param] = max_tokens
        if response_format is not None:
            request_kwargs["response_format"] = response_format.as_response_format()

        started = time.monotonic()
        response, attempts = self._request_with_retry(request_kwargs)
        latency_ms = (time.monotonic() - started) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        parsed = _try_parse_json(content) if response_format is not None else None
        usage = _extract_usage(response)
        provider_name = getattr(response, "provider", None)
        message = choice.message

        logger.info(
            "openrouter.completion model=%s provider=%s routing=%s attempts=%d latency_ms=%.0f "
            "temperature=%s seed=%s tokens_in=%s tokens_out=%s cost_usd=%s cost_is_estimate=%s "
            "finish_reason=%s",
            model_id,
            provider_name,
            routing_mode.value,
            attempts,
            latency_ms,
            effective_temperature,
            effective_seed,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
            usage.cost_is_estimate,
            choice.finish_reason,
        )

        return CompletionResult(
            content=content,
            parsed=parsed,
            model=model_id,
            provider=provider_name,
            latency_ms=latency_ms,
            usage=usage,
            finish_reason=choice.finish_reason,
            attempts=attempts,
            effective_routing_policy=provider_payload,
            temperature=effective_temperature,
            seed=effective_seed,
            generation_id=getattr(response, "id", None),
            model_resolved=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            native_finish_reason=getattr(choice, "native_finish_reason", None),
            reasoning_text=getattr(message, "reasoning", None),
            reasoning_tokens=_usage_detail(
                usage_obj=getattr(response, "usage", None),
                details_attr="completion_tokens_details",
                field_attr="reasoning_tokens",
            ),
            cached_tokens=_usage_detail(
                usage_obj=getattr(response, "usage", None),
                details_attr="prompt_tokens_details",
                field_attr="cached_tokens",
            ),
            raw=_response_as_dict(response),
        )

    def _resolve_model(self, model: str | None) -> str:
        model_id = model or self._settings.OPENROUTER_DEFAULT_MODEL
        if not model_id:
            raise OpenRouterError("No model given and OPENROUTER_DEFAULT_MODEL is not configured.")
        allowlist = self._settings.FOOD_VISION_MODELS
        if allowlist and model_id not in allowlist:
            raise OpenRouterError(f"Model {model_id!r} is not in the FOOD_VISION_MODELS allowlist.")
        return model_id

    def _build_provider_payload(
        self,
        routing_mode: RoutingMode,
        policy: RoutingPolicy,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "require_parameters": self._settings.OPENROUTER_REQUIRE_PARAMETERS,
            "data_collection": policy.data_collection or self._settings.OPENROUTER_DATA_COLLECTION,
            "zdr": policy.zdr if policy.zdr is not None else self._settings.OPENROUTER_ZDR,
        }
        if routing_mode is RoutingMode.BENCHMARK:
            if not policy.provider_pin:
                raise OpenRouterError("Benchmark routing requires routing_policy.provider_pin.")
            payload["order"] = [policy.provider_pin]
            payload["allow_fallbacks"] = False
            # Quantization is optional pinning, not a requirement: several
            # benchmark candidates don't publish quantization labels at all
            # (concept review fix #3) — requiring one would silently exclude
            # them rather than just doing nothing.
            if policy.quantizations:
                payload["quantizations"] = list(policy.quantizations)
        else:
            payload["allow_fallbacks"] = self._settings.OPENROUTER_ALLOW_FALLBACKS
            if self._settings.OPENROUTER_PROVIDER_ORDER:
                payload["order"] = list(self._settings.OPENROUTER_PROVIDER_ORDER)
        return payload

    def _request_with_retry(self, request_kwargs: dict[str, Any]) -> tuple[Any, int]:
        max_attempts = self._settings.OPENROUTER_MAX_RETRIES + 1
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(**request_kwargs)
                return response, attempt
            except APIStatusError as exc:
                status = exc.status_code
                if status not in _RETRYABLE_STATUS_CODES or attempt == max_attempts:
                    raise OpenRouterError(
                        f"OpenRouter request failed (status={status}): {exc}"
                    ) from exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "openrouter.retry attempt=%d status=%d delay_s=%.2f", attempt, status, delay
                )
                self._sleep(delay)
                last_exc = exc

        raise OpenRouterError("OpenRouter request failed after retries") from last_exc

    def _backoff_delay(self, attempt: int) -> float:
        base = min(2 ** (attempt - 1) * 0.5, 4.0)
        return float(base + self._random_fn() * 0.25)


def _usage_detail(*, usage_obj: Any, details_attr: str, field_attr: str) -> int | None:
    """Read a nested usage-detail field (reasoning/cached tokens).

    Defensive: most mocked responses in tests, and some real providers,
    don't set these nested objects at all.
    """
    details = getattr(usage_obj, details_attr, None)
    if details is None:
        return None
    value = getattr(details, field_attr, None)
    return int(value) if value is not None else None


def _response_as_dict(response: Any) -> dict[str, Any]:
    """Best-effort full response body as a plain dict.

    Real ``openai`` SDK responses are pydantic models (``model_dump``);
    test doubles are plain objects without it — those fall back to an
    empty dict rather than raising.
    """
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _try_parse_json(content: str) -> dict[str, Any] | None:
    try:
        parsed: dict[str, Any] = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("openrouter.parse_failed: response content was not valid JSON")
        return None
    return parsed


def _extract_usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(input_tokens=None, output_tokens=None, cost_usd=None, cost_is_estimate=True)
    cost = getattr(usage, "cost", None)
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        cost_usd=float(cost) if cost is not None else None,
        cost_is_estimate=cost is None,
    )
