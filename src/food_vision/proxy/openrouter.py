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
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import httpx
from openai import APIStatusError, OpenAI

from food_vision.config.settings import Settings, get_settings
from food_vision.utils.log_factory import get_logger

logger = get_logger(__name__)

#: HTTP statuses worth retrying per concept §9 ("retry only 429/5xx").
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RoutingMode(StrEnum):
    """Which OpenRouter ``provider`` routing policy to apply.

    - ``PRODUCTION``: fallbacks among the approved provider list.
    - ``BENCHMARK``: exactly one pinned provider, explicit quantizations,
      ``allow_fallbacks: false`` — reproducibility for benchmark runs.
    """

    PRODUCTION = "production"
    BENCHMARK = "benchmark"


class OpenRouterError(RuntimeError):
    """Raised for a non-retryable OpenRouter failure or exhausted retries."""


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
        response_format: JsonSchemaFormat | None = None,
        model: str | None = None,
        routing_mode: RoutingMode = RoutingMode.PRODUCTION,
        provider_pin: str | None = None,
        quantizations: Sequence[str] | None = None,
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
        response_format: JsonSchemaFormat | None = None,
        model: str | None = None,
        routing_mode: RoutingMode = RoutingMode.PRODUCTION,
        provider_pin: str | None = None,
        quantizations: Sequence[str] | None = None,
    ) -> CompletionResult:
        """Run one chat-completion request against OpenRouter.

        Raises:
            OpenRouterError: model not on the allowlist, benchmark routing
                missing its required pin/quantizations, or the request
                failed (non-retryable status, or retries exhausted).
        """
        model_id = self._resolve_model(model)
        provider_payload = self._build_provider_payload(routing_mode, provider_pin, quantizations)

        request_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": list(messages),
            "temperature": self._settings.OPENROUTER_TEMPERATURE,
            "extra_body": {"provider": provider_payload},
        }
        if self._settings.OPENROUTER_SEED is not None:
            request_kwargs["seed"] = self._settings.OPENROUTER_SEED
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

        logger.info(
            "openrouter.completion model=%s provider=%s routing=%s attempts=%d latency_ms=%.0f "
            "tokens_in=%s tokens_out=%s cost_usd=%s cost_is_estimate=%s finish_reason=%s",
            model_id,
            provider_name,
            routing_mode.value,
            attempts,
            latency_ms,
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
        provider_pin: str | None,
        quantizations: Sequence[str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "require_parameters": self._settings.OPENROUTER_REQUIRE_PARAMETERS,
            "data_collection": self._settings.OPENROUTER_DATA_COLLECTION,
            "zdr": self._settings.OPENROUTER_ZDR,
        }
        if routing_mode is RoutingMode.BENCHMARK:
            if not provider_pin:
                raise OpenRouterError("Benchmark routing requires provider_pin.")
            if not quantizations:
                raise OpenRouterError("Benchmark routing requires explicit quantizations.")
            payload["order"] = [provider_pin]
            payload["allow_fallbacks"] = False
            payload["quantizations"] = list(quantizations)
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
