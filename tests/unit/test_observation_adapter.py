"""observation.adapter: message shape, mode selection, outcome classification.

No live network — ``_FakeProvider`` stands in for ``proxy.ModelProvider``
and returns a scripted :class:`CompletionResult`, matching the M1 exit
criterion: "observe() returns a classified Observation from a recorded
fixture".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from food_vision.observation.adapter import (
    ObservationConfig,
    ObservationOutcome,
    Routing,
    observe,
)
from food_vision.observation.schema import ObservationStrategy, StructuredOutputMode
from food_vision.proxy.openrouter import CompletionResult, JsonSchemaFormat, RoutingMode, Usage


def _completion(
    *,
    content: str = "{}",
    parsed: dict[str, Any] | None = None,
    finish_reason: str | None = "stop",
    native_finish_reason: str | None = None,
    reasoning_tokens: int | None = None,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        parsed=parsed,
        model="vendor/model-a",
        provider="vendor-provider",
        latency_ms=120.0,
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, cost_is_estimate=False),
        finish_reason=finish_reason,
        attempts=1,
        effective_routing_policy={},
        temperature=0.0,
        seed=None,
        native_finish_reason=native_finish_reason,
        reasoning_tokens=reasoning_tokens,
    )


@dataclass
class _FakeProvider:
    """Records the call it received and returns a scripted result."""

    result: CompletionResult
    received_kwargs: dict[str, Any] | None = None

    def complete(self, **kwargs: Any) -> CompletionResult:
        self.received_kwargs = kwargs
        return self.result


def _routing(model: str = "vendor/model-a") -> Routing:
    return Routing(model=model, mode=RoutingMode.PRODUCTION)


class TestPromptSetAndMaxTokens:
    def test_default_prompt_set_is_fruit(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        text = received["messages"][0]["content"][0]["text"]
        assert "edible" in text

    def test_ecustfd_prompt_set_selects_whole_mass_prompt(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1, prompt_set="ecustfd")

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        text = received["messages"][0]["content"][0]["text"]
        assert "whole" in text
        assert "coin" in text

    def test_ecustfd_s2_prompt_missing_raises_file_not_found(self) -> None:
        """pre-study.md §3: S2 is dropped for ECUSTFD (no length/diameter
        ground truth) — there is deliberately no ecustfd_s2_*.md file."""
        provider = _FakeProvider(_completion())
        config = ObservationConfig(strategy=ObservationStrategy.S2, prompt_set="ecustfd")

        with pytest.raises(FileNotFoundError, match="ecustfd"):
            observe(provider, images=[b"img"], config=config, routing=_routing())

    def test_max_tokens_forwarded_to_provider(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1, max_tokens=2048)

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        assert received["max_tokens"] == 2048

    def test_max_tokens_none_by_default(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        assert received["max_tokens"] is None


class TestMessageShape:
    def test_one_user_message_with_prompt_then_images_in_view_order(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        images = [b"c1-bytes", b"c2-bytes"]
        config = ObservationConfig(strategy=ObservationStrategy.S1, views=("c1", "c2"))

        observe(provider, images=images, config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        (message,) = received["messages"]
        assert message["role"] == "user"
        content = message["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"
        # view order preserved: c1 bytes come before c2 bytes
        import base64

        assert (
            base64.b64decode(content[1]["image_url"]["url"].removeprefix("data:image/jpeg;base64,"))
            == b"c1-bytes"
        )
        assert (
            base64.b64decode(content[2]["image_url"]["url"].removeprefix("data:image/jpeg;base64,"))
            == b"c2-bytes"
        )

    def test_raises_on_image_view_count_mismatch(self) -> None:
        provider = _FakeProvider(_completion())
        config = ObservationConfig(strategy=ObservationStrategy.S1, views=("c1", "c2"))

        with pytest.raises(ValueError, match="Expected 2 image"):
            observe(provider, images=[b"only-one"], config=config, routing=_routing())

    def test_repair_true_is_rejected(self) -> None:
        provider = _FakeProvider(_completion())
        config = ObservationConfig(strategy=ObservationStrategy.S1, repair=True)

        with pytest.raises(ValueError, match="repair"):
            observe(provider, images=[b"img"], config=config, routing=_routing())


class TestModeSelection:
    def test_json_schema_mode_sends_strict_schema(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 50.0}))
        config = ObservationConfig(
            strategy=ObservationStrategy.S1,
            structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
        )

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        assert isinstance(received["response_format"], JsonSchemaFormat)

    def test_json_object_mode_parses_content_manually(self) -> None:
        provider = _FakeProvider(
            _completion(content='{"observations": "x", "mass_g": 50.0}', parsed=None)
        )
        config = ObservationConfig(
            strategy=ObservationStrategy.S1,
            structured_output_mode=StructuredOutputMode.JSON_OBJECT,
        )

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.OK
        assert result.values == {"observations": "x", "mass_g": 50.0}

    def test_s3_requires_prior_bounds(self) -> None:
        provider = _FakeProvider(_completion())
        config = ObservationConfig(strategy=ObservationStrategy.S3)

        with pytest.raises(ValueError, match="prior"):
            observe(provider, images=[b"img"], config=config, routing=_routing())

    def test_s3_formats_prior_into_prompt(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 50.0}))
        config = ObservationConfig(
            strategy=ObservationStrategy.S3, prior_low_g=90.0, prior_high_g=140.0
        )

        observe(provider, images=[b"img"], config=config, routing=_routing())

        received = provider.received_kwargs
        assert received is not None
        text = received["messages"][0]["content"][0]["text"]
        assert "90.0" in text and "140.0" in text
        assert "{prior_low_g}" not in text


class TestOutcomeClassification:
    def test_ok_outcome_for_valid_response(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": 100.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.OK
        assert result.values == {"observations": "x", "mass_g": 100.0}
        assert result.error is None

    def test_invalid_outcome_for_unparseable_content(self) -> None:
        provider = _FakeProvider(_completion(parsed=None))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.INVALID

    def test_invalid_outcome_for_unparseable_json_object_mode_content(self) -> None:
        provider = _FakeProvider(_completion(content="not json at all", parsed=None))
        config = ObservationConfig(
            strategy=ObservationStrategy.S1,
            structured_output_mode=StructuredOutputMode.JSON_OBJECT,
        )

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.INVALID

    def test_invalid_outcome_for_schema_violation(self) -> None:
        provider = _FakeProvider(_completion(parsed={"observations": "x", "mass_g": -5.0}))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.INVALID
        assert result.values is None

    def test_refused_outcome_from_native_finish_reason(self) -> None:
        provider = _FakeProvider(_completion(parsed=None, native_finish_reason="CONTENT_FILTER"))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.REFUSED

    def test_truncated_outcome_for_length_finish_reason(self) -> None:
        provider = _FakeProvider(_completion(parsed=None, finish_reason="length"))
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.TRUNCATED

    def test_truncated_outcome_for_empty_content_with_reasoning_tokens(self) -> None:
        """Reasoning budget exhausted: finish_reason=length, empty content,
        reasoning_tokens > 0 (§7.2)."""
        provider = _FakeProvider(
            _completion(
                content="",
                parsed=None,
                finish_reason="length",
                reasoning_tokens=2048,
            )
        )
        config = ObservationConfig(strategy=ObservationStrategy.S1)

        result = observe(provider, images=[b"img"], config=config, routing=_routing())

        assert result.outcome == ObservationOutcome.TRUNCATED
        assert result.completion.reasoning_tokens == 2048
