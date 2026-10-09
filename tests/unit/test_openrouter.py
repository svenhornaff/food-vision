"""OpenRouterClient: request building, routing, retries, telemetry.

No network I/O — the ``openai`` SDK's HTTP layer is mocked out entirely by
monkeypatching ``OpenRouterClient._client`` after construction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError

from food_vision.config.settings import Settings
from food_vision.proxy.openrouter import (
    JsonSchemaFormat,
    OpenRouterClient,
    OpenRouterError,
    RoutingMode,
)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_DEFAULT_MODEL": "vendor/model-a",
        "FOOD_VISION_MODELS": ("vendor/model-a", "vendor/model-b"),
    }
    base.update(overrides)
    return Settings(**base)


def _fake_response(
    *,
    content: str = "{}",
    finish_reason: str = "stop",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    cost: float | None = 0.002,
    provider: str | None = "vendor-provider",
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, provider=provider)


def _api_status_error(status_code: int) -> APIStatusError:
    response = httpx.Response(status_code=status_code, request=httpx.Request("POST", "http://x"))
    return APIStatusError(f"status {status_code}", response=response, body=None)


def _client_with_mock(
    settings: Settings,
    *,
    sleep: Any = lambda _: None,
    random_fn: Any = lambda: 0.0,
) -> tuple[OpenRouterClient, MagicMock]:
    client = OpenRouterClient(settings, sleep=sleep, random_fn=random_fn)
    mock_create = MagicMock()
    fake_sdk_client: Any = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    client._client = fake_sdk_client
    return client, mock_create


def test_missing_api_key_raises() -> None:
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(_settings(OPENROUTER_API_KEY=""))


def test_complete_sends_deterministic_request_and_parses_json() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.return_value = _fake_response(content='{"foo": "bar"}')

    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        response_format=JsonSchemaFormat(name="obs", schema={"type": "object"}),
    )

    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "vendor/model-a"
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["extra_body"]["provider"]["require_parameters"] is True
    assert kwargs["extra_body"]["provider"]["data_collection"] == "deny"
    assert kwargs["extra_body"]["provider"]["zdr"] is True

    assert result.parsed == {"foo": "bar"}
    assert result.attempts == 1
    assert result.usage.cost_usd == 0.002
    assert result.usage.cost_is_estimate is False
    assert result.provider == "vendor-provider"


def test_complete_without_response_format_does_not_parse_json() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.return_value = _fake_response(content="not json")

    result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert "response_format" not in mock_create.call_args.kwargs
    assert result.parsed is None
    assert result.content == "not json"


def test_unknown_model_rejected_by_allowlist() -> None:
    client, _ = _client_with_mock(_settings())
    with pytest.raises(OpenRouterError, match="allowlist"):
        client.complete(messages=[], model="vendor/not-allowed")


def test_seed_included_when_configured() -> None:
    client, mock_create = _client_with_mock(_settings(OPENROUTER_SEED=42))
    mock_create.return_value = _fake_response()

    client.complete(messages=[{"role": "user", "content": "hi"}])

    assert mock_create.call_args.kwargs["seed"] == 42


def test_production_routing_uses_configured_provider_order() -> None:
    client, mock_create = _client_with_mock(
        _settings(
            OPENROUTER_PROVIDER_ORDER=("vendor-a", "vendor-b"),
            OPENROUTER_ALLOW_FALLBACKS=True,
        )
    )
    mock_create.return_value = _fake_response()

    client.complete(
        messages=[{"role": "user", "content": "hi"}], routing_mode=RoutingMode.PRODUCTION
    )

    provider = mock_create.call_args.kwargs["extra_body"]["provider"]
    assert provider["order"] == ["vendor-a", "vendor-b"]
    assert provider["allow_fallbacks"] is True


def test_benchmark_routing_pins_single_provider_and_quantization() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.return_value = _fake_response()

    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        routing_mode=RoutingMode.BENCHMARK,
        provider_pin="pinned-vendor",
        quantizations=["fp16"],
    )

    provider = mock_create.call_args.kwargs["extra_body"]["provider"]
    assert provider["order"] == ["pinned-vendor"]
    assert provider["allow_fallbacks"] is False
    assert provider["quantizations"] == ["fp16"]


def test_benchmark_routing_requires_provider_pin() -> None:
    client, _ = _client_with_mock(_settings())
    with pytest.raises(OpenRouterError, match="provider_pin"):
        client.complete(messages=[], routing_mode=RoutingMode.BENCHMARK, quantizations=["fp16"])


def test_benchmark_routing_requires_quantizations() -> None:
    client, _ = _client_with_mock(_settings())
    with pytest.raises(OpenRouterError, match="quantizations"):
        client.complete(
            messages=[], routing_mode=RoutingMode.BENCHMARK, provider_pin="pinned-vendor"
        )


def test_retries_on_429_then_succeeds() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.side_effect = [_api_status_error(429), _fake_response()]

    result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result.attempts == 2
    assert mock_create.call_count == 2


def test_retries_on_5xx_then_succeeds() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.side_effect = [_api_status_error(503), _fake_response()]

    result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result.attempts == 2


def test_does_not_retry_non_retryable_4xx() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.side_effect = [_api_status_error(400)]

    with pytest.raises(OpenRouterError, match="status=400"):
        client.complete(messages=[{"role": "user", "content": "hi"}])

    assert mock_create.call_count == 1


def test_gives_up_after_max_retries() -> None:
    client, mock_create = _client_with_mock(_settings(OPENROUTER_MAX_RETRIES=2))
    mock_create.side_effect = [
        _api_status_error(503),
        _api_status_error(503),
        _api_status_error(503),
    ]

    with pytest.raises(OpenRouterError):
        client.complete(messages=[{"role": "user", "content": "hi"}])

    assert mock_create.call_count == 3  # 1 initial attempt + 2 retries


def test_sleep_called_between_retries_with_jitter() -> None:
    sleeps: list[float] = []
    client, mock_create = _client_with_mock(_settings(), sleep=lambda delay: sleeps.append(delay))
    mock_create.side_effect = [_api_status_error(429), _fake_response()]

    client.complete(messages=[{"role": "user", "content": "hi"}])

    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_missing_cost_marks_usage_as_estimate() -> None:
    client, mock_create = _client_with_mock(_settings())
    mock_create.return_value = _fake_response(cost=None)

    result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result.usage.cost_usd is None
    assert result.usage.cost_is_estimate is True


def test_no_model_configured_raises() -> None:
    client, _ = _client_with_mock(_settings(OPENROUTER_DEFAULT_MODEL="", FOOD_VISION_MODELS=()))
    with pytest.raises(OpenRouterError, match="No model"):
        client.complete(messages=[])
