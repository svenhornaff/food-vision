"""prestudy.run: key building, resume, budget/dry-run stops, priors.

No network — ``_FakeProvider`` stands in for ``proxy.ModelProvider``; a
tiny in-memory JPEG stands in for a dataset image.
"""

from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from food_vision.imaging.preprocess import ImageConfig, normalize
from food_vision.observation.schema import ObservationStrategy
from food_vision.prestudy.ecustfd import Item
from food_vision.prestudy.run import (
    ModelSpec,
    RunConfig,
    append_result,
    build_key,
    compute_priors,
    git_sha,
    load_existing_keys,
    load_models_toml,
    run_sweep,
)
from food_vision.proxy.openrouter import CompletionResult, Usage


def _jpeg_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    image = Image.new("RGB", (64, 48), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _color_for(seed: str) -> tuple[int, int, int]:
    """A distinct, deterministic color per image so each item normalizes
    to distinct bytes (and thus a distinct image_sha256/key) — using the
    same pixels for every fixture image would collapse every item onto
    one key."""
    digest = __import__("hashlib").sha256(seed.encode()).digest()
    return (digest[0], digest[1], digest[2])


@dataclass
class _FakeProvider:
    pred_g: float = 150.0
    cost_usd: float = 0.001
    calls: list[dict[str, Any]] | None = None

    def complete(self, **kwargs: Any) -> CompletionResult:
        if self.calls is None:
            self.calls = []
        self.calls.append(kwargs)
        return CompletionResult(
            content="{}",
            parsed={"observations": "x", "mass_g": self.pred_g},
            model="vendor/model-a",
            provider="vendor-provider",
            latency_ms=100.0,
            usage=Usage(
                input_tokens=10, output_tokens=5, cost_usd=self.cost_usd, cost_is_estimate=False
            ),
            finish_reason="stop",
            attempts=1,
            effective_routing_policy={},
            temperature=0.0,
            seed=None,
        )


def _items(n_objects: int = 2, split: str = "holdout") -> tuple[Item, ...]:
    items = []
    for i in range(n_objects):
        object_key = f"apple{i:03d}"
        for view, suffix in (("side", "S"), ("top", "T")):
            items.append(
                Item(
                    image_path=Path(f"{object_key}{suffix}(1).JPG"),
                    object_key=object_key,
                    fruit_type="apple",
                    view=view,
                    weight_g=200.0 + i,
                    split=split,
                    variant=1,
                )
            )
    return tuple(items)


def _write_images(snapshot_dir: Path, items: tuple[Item, ...]) -> None:
    images_dir = snapshot_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        color = _color_for(f"{item.object_key}-{item.view}")
        (images_dir / item.image_path).write_bytes(_jpeg_bytes(color))


def _model_spec(
    model: str = "vendor/model-a",
    *,
    temperature_zero_ok: bool = True,
    max_tokens_param: str = "max_tokens",
) -> ModelSpec:
    return ModelSpec(
        model=model,
        provider_pin="vendor-provider",
        temperature_zero_ok=temperature_zero_ok,
        max_tokens_param=max_tokens_param,
    )


class TestBuildKey:
    def test_format(self) -> None:
        key = build_key(
            model="vendor/a", strategy=ObservationStrategy.S1, image_sha256="abc", repeat=2
        )
        assert key == "vendor/a|S1|abc|2"


class TestGitSha:
    def test_returns_unknown_when_subprocess_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: Any, **kwargs: Any) -> None:
            raise FileNotFoundError("no git")

        monkeypatch.setattr(subprocess, "run", _raise)

        assert git_sha() == "unknown"

    def test_returns_real_sha_in_this_repo(self) -> None:
        sha = git_sha()
        assert sha == "unknown" or len(sha) == 40


class TestLoadExistingKeys:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_existing_keys(tmp_path / "nope.jsonl") == frozenset()

    def test_collects_keys_from_valid_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        path.write_text('{"key": "a"}\n{"key": "b"}\n')

        assert load_existing_keys(path) == frozenset({"a", "b"})

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        path.write_text('{"key": "a"}\nnot json\n{"key": "b"}\n')

        assert load_existing_keys(path) == frozenset({"a", "b"})

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        path.write_text('{"key": "a"}\n\n\n')

        assert load_existing_keys(path) == frozenset({"a"})


class TestAppendResult:
    def test_creates_parent_dirs_and_appends(self, tmp_path: Path) -> None:
        from food_vision.prestudy.run import ResultRecord

        path = tmp_path / "nested" / "results.jsonl"
        record = ResultRecord(
            key="k",
            model="m",
            provider="p",
            strategy="S1",
            repeat=0,
            object_key="apple001",
            fruit_type="apple",
            view="top",
            variant=1,
            split="holdout",
            true_g=200.0,
            pred_g=190.0,
            outcome="ok",
            model_resolved="m-v2",
            generation_id="gen-1",
            finish_reason="stop",
            prompt_sha256="p-hash",
            image_sent_sha256="i-hash",
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=0,
            cost_usd=0.001,
            latency_ms=100.0,
            git_sha="abc123",
            ts="2026-01-01T00:00:00+00:00",
        )

        append_result(path, record)
        append_result(path, record)

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["key"] == "k"


class TestComputePriors:
    def test_uses_only_dev_split_weights(self) -> None:
        items = (
            Item(Path("a.JPG"), "a", "apple", "top", 100.0, "dev", 1),
            Item(Path("b.JPG"), "b", "apple", "top", 300.0, "dev", 1),
            Item(Path("c.JPG"), "c", "apple", "top", 9999.0, "holdout", 1),
        )

        priors = compute_priors(items)

        assert priors["apple"] == (100.0, 300.0)

    def test_raises_when_type_has_no_dev_items(self) -> None:
        items = (Item(Path("a.JPG"), "a", "apple", "top", 100.0, "holdout", 1),)

        with pytest.raises(ValueError, match="apple"):
            compute_priors(items)


class TestLoadModelsToml:
    def test_loads_models_array(self, tmp_path: Path) -> None:
        path = tmp_path / "models.toml"
        path.write_text(
            """
            [[models]]
            model = "vendor/model-a"
            provider_pin = "vendor-provider"
            quantizations = ["fp8"]
            notes = "smoke ok"
            """
        )

        specs = load_models_toml(path)

        assert specs["vendor/model-a"].provider_pin == "vendor-provider"
        assert specs["vendor/model-a"].quantizations == ("fp8",)
        assert specs["vendor/model-a"].notes == "smoke ok"
        assert specs["vendor/model-a"].temperature_zero_ok is True
        assert specs["vendor/model-a"].max_tokens_param == "max_tokens"

    def test_loads_temperature_zero_ok_false_and_max_tokens_param(self, tmp_path: Path) -> None:
        """anthropic/claude-sonnet-5 and openai/gpt-5 need these set --
        see bench/runs/prestudy-lean/models.toml for the live finding."""
        path = tmp_path / "models.toml"
        path.write_text(
            """
            [[models]]
            model = "openai/gpt-5"
            provider_pin = "azure"
            temperature_zero_ok = false
            max_tokens_param = "max_completion_tokens"
            """
        )

        specs = load_models_toml(path)

        assert specs["openai/gpt-5"].temperature_zero_ok is False
        assert specs["openai/gpt-5"].max_tokens_param == "max_completion_tokens"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_models_toml(tmp_path / "nope.toml")

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "models.toml"
        path.write_text('[[models]]\nmodel = "vendor/model-a"\n')

        with pytest.raises(KeyError):
            load_models_toml(path)


class TestRunSweep:
    def _base_config(self, tmp_path: Path, items: tuple[Item, ...], **overrides: Any) -> RunConfig:
        snapshot_dir = tmp_path / "snapshot"
        _write_images(snapshot_dir, items)
        defaults: dict[str, Any] = {
            "items": items,
            "models": (_model_spec(),),
            "strategies": (ObservationStrategy.S1,),
            "results_path": tmp_path / "results.jsonl",
            "snapshot_dir": snapshot_dir,
            "image_config": ImageConfig(max_decoded_pixels=10_000_000, long_edge_px=32),
            "split_filter": "all",
        }
        defaults.update(overrides)
        return RunConfig(**defaults)

    def test_runs_all_combos_and_writes_results(self, tmp_path: Path) -> None:
        items = _items(n_objects=2)
        config = self._base_config(tmp_path, items)
        provider = _FakeProvider()

        stats = run_sweep(provider, config)

        assert stats.stopped_reason is None
        assert stats.by_model["vendor/model-a"].completed == len(items)
        lines = config.results_path.read_text().splitlines()
        assert len(lines) == len(items)
        first = json.loads(lines[0])
        assert first["outcome"] == "ok"
        assert first["pred_g"] == 150.0

    def test_model_spec_temperature_and_max_tokens_param_reach_provider(
        self, tmp_path: Path
    ) -> None:
        """anthropic/claude-sonnet-5 / openai/gpt-5 fix: ModelSpec's
        temperature_zero_ok=False and a non-default max_tokens_param must
        flow through ObservationConfig into the provider.complete() call,
        not just sit unused in models.toml."""
        items = _items(n_objects=1)
        model_spec = _model_spec(
            temperature_zero_ok=False, max_tokens_param="max_completion_tokens"
        )
        config = self._base_config(tmp_path, items, models=(model_spec,))
        provider = _FakeProvider()

        run_sweep(provider, config)

        assert provider.calls is not None and len(provider.calls) > 0
        for call in provider.calls:
            assert call["send_temperature"] is False
            assert call["max_tokens_param"] == "max_completion_tokens"

    def test_resume_skips_existing_keys(self, tmp_path: Path) -> None:
        items = _items(n_objects=2)
        config = self._base_config(tmp_path, items)

        # First run: populate results.jsonl.
        run_sweep(_FakeProvider(), config)
        lines_after_first = config.results_path.read_text().splitlines()

        # Second run against the same config/results file: must skip everything.
        stats = run_sweep(_FakeProvider(), config)

        assert stats.by_model["vendor/model-a"].completed == 0
        assert stats.by_model["vendor/model-a"].skipped_existing == len(items)
        lines_after_second = config.results_path.read_text().splitlines()
        assert lines_after_second == lines_after_first  # nothing appended

    def test_budget_stop(self, tmp_path: Path) -> None:
        items = _items(n_objects=3)  # 6 items
        config = self._base_config(tmp_path, items, budget_usd=0.0015)  # ~1-2 calls at $0.001 each
        provider = _FakeProvider(cost_usd=0.001)

        stats = run_sweep(provider, config)

        assert stats.stopped_reason == "budget"
        assert stats.total_completed() < len(items)
        assert stats.total_cost_usd() >= 0.0015

    def test_dry_run_caps_per_model(self, tmp_path: Path) -> None:
        items = _items(n_objects=3)  # 6 items
        config = self._base_config(tmp_path, items, dry_run_n=2)
        provider = _FakeProvider()

        stats = run_sweep(provider, config)

        assert stats.stopped_reason == "dry_run"
        assert stats.by_model["vendor/model-a"].completed == 2

    def test_s3_strategy_uses_dev_derived_prior(self, tmp_path: Path) -> None:
        items = (
            Item(Path("apple001S(1).JPG"), "apple001", "apple", "side", 200.0, "dev", 1),
            Item(Path("apple002S(1).JPG"), "apple002", "apple", "side", 300.0, "dev", 1),
            Item(Path("apple003S(1).JPG"), "apple003", "apple", "side", 250.0, "holdout", 1),
        )
        config = self._base_config(
            tmp_path, items, strategies=(ObservationStrategy.S3,), split_filter="holdout"
        )
        provider = _FakeProvider()

        run_sweep(provider, config)

        assert provider.calls is not None
        (call,) = provider.calls
        text = call["messages"][0]["content"][0]["text"]
        assert "200.0" in text and "300.0" in text

    def test_prompt_and_image_hashes_are_present_and_deterministic(self, tmp_path: Path) -> None:
        items = _items(n_objects=1)
        config = self._base_config(tmp_path, items)

        run_sweep(_FakeProvider(), config)

        lines = [json.loads(line) for line in config.results_path.read_text().splitlines()]
        raw = (config.snapshot_dir / "images" / items[0].image_path).read_bytes()
        expected_image_sha = (
            __import__("hashlib").sha256(normalize(raw, config.image_config)).hexdigest()
        )
        assert lines[0]["image_sent_sha256"] == expected_image_sha
        assert len(lines[0]["prompt_sha256"]) == 64
