"""Run the model x strategy x item sweep, append-only, resumable.

docs/dev/pre-study.md §4.2 (L2/L3): cross items x models x strategies x
repeats; normalize each image, call ``observe()``, append one JSON line
per call to ``results.jsonl``. Resume = skip keys already present.
``--dry-run N`` caps each model at N real calls and extrapolates; the
caller (``__main__.py``) is responsible for actually running only a
small number and asking for confirmation first — this module makes no
distinction between a "real" and "live" call; every call here is live.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from food_vision.imaging.preprocess import ImageConfig, normalize
from food_vision.observation.adapter import (
    ObservationConfig,
    Routing,
    render_prompt,
)
from food_vision.observation.adapter import observe as observe_fn
from food_vision.observation.schema import ObservationStrategy
from food_vision.prestudy.ecustfd import Item
from food_vision.proxy.openrouter import ModelProvider, RoutingMode, RoutingPolicy
from food_vision.utils.log_factory import get_logger

logger = get_logger(__name__)

__all__ = [
    "ModelSpec",
    "ResultRecord",
    "RunConfig",
    "ModelRunStats",
    "RunStats",
    "git_sha",
    "load_models_toml",
    "load_existing_keys",
    "append_result",
    "compute_priors",
    "build_key",
    "run_sweep",
]


@dataclass(frozen=True)
class ModelSpec:
    """One candidate model's fixed routing (pre-study.md §3: "one pinned
    provider per model"). Loaded from ``models.toml`` — deliberately not
    inferred, since a wrong pin would silently change what's measured."""

    model: str
    provider_pin: str
    quantizations: tuple[str, ...] = ()
    #: ``False`` when a live smoke test found that *no* OpenRouter endpoint
    #: for this model lists ``temperature`` in ``supported_parameters``
    #: (discovered for ``anthropic/claude-sonnet-5`` and ``openai/gpt-5``;
    #: with ``require_parameters: true``, sending it filters out every
    #: candidate endpoint regardless of provider pin). ``run_sweep`` then
    #: omits ``temperature`` from the request entirely rather than sending
    #: a value the provider will reject the whole request over.
    temperature_zero_ok: bool = True
    #: Key to send the completion-length cap under. Some GPT-5-family
    #: endpoints reject ``"max_tokens"`` and require
    #: ``"max_completion_tokens"`` instead.
    max_tokens_param: str = "max_tokens"
    notes: str = ""


def load_models_toml(path: Path) -> dict[str, ModelSpec]:
    """Load ``models.toml``'s ``[[models]]`` array.

    Raises:
        FileNotFoundError: ``path`` doesn't exist.
        KeyError: a `[[models]]` entry is missing ``model`` or
            ``provider_pin``.
    """
    import tomllib

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    specs: dict[str, ModelSpec] = {}
    for entry in data.get("models", []):
        spec = ModelSpec(
            model=entry["model"],
            provider_pin=entry["provider_pin"],
            quantizations=tuple(entry.get("quantizations", [])),
            temperature_zero_ok=entry.get("temperature_zero_ok", True),
            max_tokens_param=entry.get("max_tokens_param", "max_tokens"),
            notes=entry.get("notes", ""),
        )
        specs[spec.model] = spec
    return specs


@dataclass(frozen=True)
class ResultRecord:
    """One line of ``results.jsonl`` (pre-study.md §4.2's schema, verbatim field order)."""

    key: str
    model: str
    provider: str | None
    strategy: str
    repeat: int
    object_key: str
    fruit_type: str
    view: str
    split: str
    true_g: float
    pred_g: float | None
    outcome: str
    model_resolved: str | None
    generation_id: str | None
    finish_reason: str | None
    prompt_sha256: str
    image_sent_sha256: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    latency_ms: float
    git_sha: str
    ts: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=False) + "\n"


def git_sha() -> str:
    """Current commit, for ``ResultRecord.git_sha``. Never raises —
    returns ``"unknown"`` outside a git repo or if git isn't installed,
    since that shouldn't block a dry run."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_key(*, model: str, strategy: ObservationStrategy, image_sha256: str, repeat: int) -> str:
    """pre-study.md §4.2: ``key = model|strategy|image_sha256|repeat``."""
    return f"{model}|{strategy.value}|{image_sha256}|{repeat}"


def load_existing_keys(results_path: Path) -> frozenset[str]:
    """Keys already present in ``results.jsonl`` — resume skips these.

    Tolerates a missing file (nothing run yet) and tolerates/skips
    malformed trailing lines (e.g. from a process killed mid-write)
    rather than refusing to resume.
    """
    if not results_path.exists():
        return frozenset()

    keys: set[str] = set()
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed results.jsonl line", extra={"line": line_number})
                continue
            key = record.get("key")
            if isinstance(key, str):
                keys.add(key)
    return frozenset(keys)


def append_result(results_path: Path, record: ResultRecord) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json_line())


def compute_priors(items: tuple[Item, ...]) -> dict[str, tuple[float, float]]:
    """Per-type ``(prior_low_g, prior_high_g)`` for strategy S3, from
    **dev-split weights only** (pre-study.md §2: "Dev is used for ... the
    S3 prior. The decision uses hold-out only.").

    Raises:
        ValueError: a fruit type in ``items`` has no dev-split items.
    """
    weights_by_type: dict[str, list[float]] = defaultdict(list)
    for item in items:
        if item.split == "dev":
            weights_by_type[item.fruit_type].append(item.weight_g)

    priors: dict[str, tuple[float, float]] = {}
    for fruit_type in {item.fruit_type for item in items}:
        weights = weights_by_type.get(fruit_type, [])
        if not weights:
            raise ValueError(
                f"No dev-split items for fruit_type={fruit_type!r}; can't form a prior."
            )
        priors[fruit_type] = (min(weights), max(weights))
    return priors


@dataclass
class ModelRunStats:
    attempted: int = 0
    skipped_existing: int = 0
    completed: int = 0
    cost_usd: float = 0.0


@dataclass
class RunStats:
    by_model: dict[str, ModelRunStats] = field(default_factory=dict)
    #: ``None`` if the sweep ran to completion; otherwise why it stopped.
    stopped_reason: str | None = None

    def total_cost_usd(self) -> float:
        return sum(stats.cost_usd for stats in self.by_model.values())

    def total_completed(self) -> int:
        return sum(stats.completed for stats in self.by_model.values())


@dataclass(frozen=True)
class RunConfig:
    items: tuple[Item, ...]
    models: tuple[ModelSpec, ...]
    strategies: tuple[ObservationStrategy, ...]
    results_path: Path
    #: Directory containing the ECUSTFD snapshot's ``images/`` subdir
    #: (``ecustfd.download()``'s return value).
    snapshot_dir: Path
    repeats: int = 1
    split_filter: str = "holdout"  # "dev" | "holdout" | "all"
    image_config: ImageConfig = field(
        default_factory=lambda: ImageConfig(max_decoded_pixels=40_000_000, long_edge_px=1024)
    )
    max_tokens: int = 2048
    #: Cap on real calls *per model* before moving to the next model.
    #: ``None`` runs the full planned sweep.
    dry_run_n: int | None = None
    #: Hard stop once this invocation's cumulative cost reaches the
    #: budget. Checked before each call, so the budget is never exceeded
    #: mid-call, only possibly exceeded by at most one call's cost.
    budget_usd: float | None = None


def run_sweep(provider: ModelProvider, config: RunConfig) -> RunStats:
    """Run (a bounded or full) sweep, appending to ``config.results_path``.

    Raises:
        ValueError: a strategy's prompt_set/priors can't be built (e.g.
            S3 with a fruit type missing from the dev split).
        OpenRouterError: propagated from a transport failure after the
            proxy's own retries — not caught here. Already-appended
            results stay in ``results.jsonl``, so re-running resumes.
    """
    existing_keys = set(load_existing_keys(config.results_path))
    items = tuple(
        item
        for item in config.items
        if config.split_filter == "all" or item.split == config.split_filter
    )
    priors = compute_priors(config.items) if ObservationStrategy.S3 in config.strategies else {}

    stats = RunStats(by_model={spec.model: ModelRunStats() for spec in config.models})
    current_git_sha = git_sha()

    for model_spec in config.models:
        model_stats = stats.by_model[model_spec.model]
        for strategy in config.strategies:
            for item in items:
                for repeat in range(config.repeats):
                    if (
                        config.budget_usd is not None
                        and stats.total_cost_usd() >= config.budget_usd
                    ):
                        stats.stopped_reason = "budget"
                        return stats
                    if config.dry_run_n is not None and model_stats.completed >= config.dry_run_n:
                        break  # this model's dry-run cap reached; move to next strategy/model

                    record = _run_one(
                        provider=provider,
                        model_spec=model_spec,
                        strategy=strategy,
                        item=item,
                        repeat=repeat,
                        config=config,
                        priors=priors,
                        existing_keys=existing_keys,
                        model_stats=model_stats,
                        git_sha_value=current_git_sha,
                    )
                    if record is not None:
                        append_result(config.results_path, record)

    if config.dry_run_n is not None:
        stats.stopped_reason = "dry_run"
    return stats


def _run_one(
    *,
    provider: ModelProvider,
    model_spec: ModelSpec,
    strategy: ObservationStrategy,
    item: Item,
    repeat: int,
    config: RunConfig,
    priors: dict[str, tuple[float, float]],
    existing_keys: set[str],
    model_stats: ModelRunStats,
    git_sha_value: str,
) -> ResultRecord | None:
    """One (model, strategy, item, repeat) call, or ``None`` if its key
    already exists (resume)."""
    raw = (config.snapshot_dir / "images" / item.image_path).read_bytes()
    normalized = normalize(raw, config.image_config)
    image_sha256 = hashlib.sha256(normalized).hexdigest()

    key = build_key(
        model=model_spec.model, strategy=strategy, image_sha256=image_sha256, repeat=repeat
    )
    model_stats.attempted += 1
    if key in existing_keys:
        model_stats.skipped_existing += 1
        return None

    prior_low_g: float | None = None
    prior_high_g: float | None = None
    if strategy is ObservationStrategy.S3:
        prior_low_g, prior_high_g = priors[item.fruit_type]

    obs_config = ObservationConfig(
        strategy=strategy,
        prompt_set="ecustfd",
        views=("single",),
        max_tokens=config.max_tokens,
        max_tokens_param=model_spec.max_tokens_param,
        prior_low_g=prior_low_g,
        prior_high_g=prior_high_g,
        send_temperature=model_spec.temperature_zero_ok,
    )
    prompt_text = render_prompt(obs_config)
    routing = Routing(
        model=model_spec.model,
        mode=RoutingMode.BENCHMARK,
        policy=RoutingPolicy(
            provider_pin=model_spec.provider_pin,
            quantizations=model_spec.quantizations or None,
        ),
    )

    observation = observe_fn(provider, images=[normalized], config=obs_config, routing=routing)
    completion = observation.completion
    pred_g = observation.values["mass_g"] if observation.values is not None else None

    record = ResultRecord(
        key=key,
        model=model_spec.model,
        provider=completion.provider,
        strategy=strategy.value,
        repeat=repeat,
        object_key=item.object_key,
        fruit_type=item.fruit_type,
        view=item.view,
        split=item.split,
        true_g=item.weight_g,
        pred_g=pred_g,
        outcome=observation.outcome.value,
        model_resolved=completion.model_resolved,
        generation_id=completion.generation_id,
        finish_reason=completion.finish_reason,
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        image_sent_sha256=image_sha256,
        prompt_tokens=completion.usage.input_tokens,
        completion_tokens=completion.usage.output_tokens,
        reasoning_tokens=completion.reasoning_tokens,
        cost_usd=completion.usage.cost_usd,
        latency_ms=completion.latency_ms,
        git_sha=git_sha_value,
        ts=datetime.now(UTC).isoformat(),
    )

    existing_keys.add(key)
    model_stats.completed += 1
    if record.cost_usd is not None:
        model_stats.cost_usd += record.cost_usd
    return record
