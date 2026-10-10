"""Hold-out metrics and the decision rule (docs/dev/pre-study.md §5).

numpy only — no pandas, no scipy. Reads ``results.jsonl`` and the
committed ``split.csv``; everything here is pure/deterministic given
those two inputs (plus the bootstrap's fixed seed).

**Unit of analysis is the object**, not the image or the attempt: per
(model, strategy, object), the prediction is a two-level median —
first over each (view, variant) slot's repeats (so one "re-ask the
same photo" burst collapses to one number before it can outweigh a
slot with fewer repeats), then over that object's slot-level medians.
With exactly one variant and one repeat per view (the prior, pre-E3
behaviour — see ``ecustfd.py``'s "Variants" note), this reduces to the
same flat median across views as before: each slot has exactly one
value, so the outer median sees the same numbers either way. An object
with zero ``ok`` attempts gets the B0 (dev-mean-weight-per-type)
prediction — this is what "penalises unreliable models" (§5) means in
code: a model that mostly fails still gets scored, just badly, rather
than being excluded from the comparison.

**Implementation note not fully specified in §5:** validity (share of
calls with outcome ``ok``) is computed over *attempted* calls only. An
object with zero attempts (e.g. a partial/interrupted run) contributes a
B0-imputed prediction to MAPE/β/bias as above, but doesn't change the
validity ratio — there's no attempt to call "ok" or not. Re-running
``prestudy run`` to complete missing attempts is the intended fix, not a
validity penalty for incompleteness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from food_vision.prestudy.ecustfd import Item

__all__ = [
    "AttemptRecord",
    "ObjectPrediction",
    "ModelStrategyMetrics",
    "Decision",
    "load_results",
    "load_kfit",
    "compute_b0",
    "predictions_per_object",
    "bbox_predictions_per_object",
    "mape",
    "geometric_bias",
    "size_slope_beta",
    "validity",
    "repeat_cv",
    "cost_per_1000",
    "latency_percentiles",
    "bootstrap_mape_diff_ci",
    "compute_metrics",
    "passes_gates",
    "choose_strategy_per_model",
    "decide",
]

#: §5's gates, as constants so the decision rule reads as the spec's prose.
_GAIN_VS_B0_MIN = 0.30
_BETA_RANGE = (0.7, 1.3)
_GEOMETRIC_BIAS_MAX = 0.10
_VALIDITY_MIN = 0.98
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 20261010  # fixed, per §5: "95% intervals: ... fixed seed"


@dataclass(frozen=True)
class AttemptRecord:
    """The subset of a ``results.jsonl`` line analysis needs."""

    model: str
    strategy: str
    repeat: int
    object_key: str
    fruit_type: str
    view: str
    split: str
    true_g: float
    pred_g: float | None
    outcome: str
    cost_usd: float | None
    latency_ms: float
    #: Defaults to ``1`` for rows written before this field existed (see
    #: ``run.ResultRecord.variant``'s docstring) — those runs only ever
    #: used variant 1, so the default is exact, not a guess.
    variant: int = 1
    #: The strategy's full validated observation (``None`` if ``outcome``
    #: wasn't ``ok``, or for rows written before this field existed).
    #: Needed for ``strategy=="BBOX"``, whose mass isn't a single number
    #: on its own (review.md §5 "E1") — ``None`` defaults here are never
    #: silently wrong for BBOX specifically, since BBOX didn't exist
    #: before this field did.
    parsed: dict[str, Any] | None = None
    #: The sent image's pixel dimensions, needed to convert BBOX's
    #: normalised ``[0,1]`` coordinates to mm. ``None`` for older rows.
    image_width_px: int | None = None
    image_height_px: int | None = None


def load_results(path: Path) -> tuple[AttemptRecord, ...]:
    """Read ``results.jsonl``, tolerating malformed trailing lines the
    same way ``run.load_existing_keys`` does."""
    if not path.exists():
        return ()
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                AttemptRecord(
                    model=raw["model"],
                    strategy=raw["strategy"],
                    repeat=raw["repeat"],
                    object_key=raw["object_key"],
                    fruit_type=raw["fruit_type"],
                    view=raw["view"],
                    split=raw["split"],
                    true_g=raw["true_g"],
                    pred_g=raw["pred_g"],
                    outcome=raw["outcome"],
                    cost_usd=raw["cost_usd"],
                    latency_ms=raw["latency_ms"],
                    variant=raw.get("variant", 1),
                    parsed=raw.get("parsed"),
                    image_width_px=raw.get("image_width_px"),
                    image_height_px=raw.get("image_height_px"),
                )
            )
    return tuple(records)


def load_kfit(path: Path) -> dict[str, float]:
    """The ground-truth-box-fitted per-fruit-type mass constant (§BBOX
    strategy; ``bench/scripts/fit_k.py``). "Conservative and fine" per
    review.md §5 as the reference k; per-model dev-fitted k is a
    documented follow-up, not implemented here.

    Raises:
        OSError: ``path`` doesn't exist or isn't readable.
        ValueError: ``path``'s contents aren't a JSON object of
            ``{fruit_type: float}``.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in raw.items()
    ):
        raise ValueError(f"{path} must contain a JSON object of {{fruit_type: float}}.")
    return {str(k): float(v) for k, v in raw.items()}


def compute_b0(items: tuple[Item, ...]) -> dict[str, float]:
    """B0: dev-split mean weight per fruit type (§5)."""
    weights_by_type: dict[str, list[float]] = defaultdict(list)
    for item in items:
        if item.split == "dev":
            weights_by_type[item.fruit_type].append(item.weight_g)
    return {
        fruit_type: sum(weights) / len(weights)
        for fruit_type, weights in weights_by_type.items()
        if weights
    }


@dataclass(frozen=True)
class ObjectPrediction:
    object_key: str
    fruit_type: str
    true_g: float
    pred_g: float
    #: ``False`` means this is a B0 fallback, not a real model prediction.
    had_ok_result: bool


def predictions_per_object(
    attempts: tuple[AttemptRecord, ...],
    *,
    model: str,
    strategy: str,
    holdout_objects: dict[str, tuple[str, float]],  # object_key -> (fruit_type, true_g)
    b0_by_type: dict[str, float],
) -> tuple[ObjectPrediction, ...]:
    """One row per hold-out object: a two-level median of this (model,
    strategy)'s ``ok`` predictions — first within each (view, variant)
    slot (collapsing repeats), then across the object's slots — or the
    B0 fallback if it has none at all. See the module docstring.

    Raises:
        KeyError: an object has ``ok`` predictions but its fruit type
            has no B0 value (no dev-split items of that type at all).
    """
    ok_by_slot: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for attempt in attempts:
        if (
            attempt.model == model
            and attempt.strategy == strategy
            and attempt.split == "holdout"
            and attempt.outcome == "ok"
            and attempt.pred_g is not None
            and attempt.object_key in holdout_objects
        ):
            ok_by_slot[(attempt.object_key, attempt.view, attempt.variant)].append(attempt.pred_g)

    slot_medians_by_object: dict[str, list[float]] = defaultdict(list)
    for (object_key, _view, _variant), preds in ok_by_slot.items():
        slot_medians_by_object[object_key].append(float(np.median(preds)))

    results = []
    for object_key, (fruit_type, true_g) in holdout_objects.items():
        slot_medians = slot_medians_by_object.get(object_key)
        if slot_medians:
            pred_g = float(np.median(slot_medians))
            had_ok = True
        else:
            pred_g = b0_by_type[fruit_type]
            had_ok = False
        results.append(ObjectPrediction(object_key, fruit_type, true_g, pred_g, had_ok))
    return tuple(sorted(results, key=lambda p: p.object_key))


def _box_wh_mm(
    parsed: dict[str, Any], image_width_px: int, image_height_px: int
) -> tuple[float, float] | None:
    """(fruit_w_mm, fruit_h_mm) from a "BBOX" attempt's normalised
    ``[0,1]`` boxes, scaled by *this attempt's own* predicted coin box
    (never ground truth — a real pipeline has no access to that either;
    same method as ``bench/scripts/bbox_vlm_eval.py``). ``None`` if the
    predicted coin box collapses to ~0px (degenerate scale)."""
    coin_w_px = (parsed["coin_xmax"] - parsed["coin_xmin"]) * image_width_px
    coin_h_px = (parsed["coin_ymax"] - parsed["coin_ymin"]) * image_height_px
    if coin_w_px + coin_h_px <= 0:
        return None
    scale = 25.0 / ((coin_w_px + coin_h_px) / 2)  # mm per px
    fruit_w_px = (parsed["fruit_xmax"] - parsed["fruit_xmin"]) * image_width_px
    fruit_h_px = (parsed["fruit_ymax"] - parsed["fruit_ymin"]) * image_height_px
    return fruit_w_px * scale, fruit_h_px * scale


def bbox_predictions_per_object(
    attempts: tuple[AttemptRecord, ...],
    *,
    model: str,
    holdout_objects: dict[str, tuple[str, float]],
    b0_by_type: dict[str, float],
    kfit: dict[str, float],
) -> tuple[ObjectPrediction, ...]:
    """``predictions_per_object``'s BBOX-strategy counterpart
    (review.md §5 "E1"). Unlike S1/S2/S3, a single (view, variant)
    BBOX attempt's boxes don't yield a mass estimate on their own — mass
    needs the top view's (width, height) *and* the side view's height
    together (the same coin-scaled geometry formula as
    ``bbox_oracle.py``/``bbox_vlm_eval.py``: ``k * a * b * h`` where
    ``a=median(max(top w,h))``, ``b=median(min(top w,h))``,
    ``h=median(side h)``). So this groups by ``(object_key, view)``
    directly — pooling every variant/repeat within a view, like
    ``bbox_vlm_eval.py``'s ``vlm_proxy_both`` — rather than reusing
    ``predictions_per_object``'s per-attempt-pred_g aggregation, which
    assumes each attempt already carries a complete mass estimate.

    An object missing a usable top+side pair, or whose fruit type has no
    ``kfit`` entry, gets the B0 fallback, same semantics as
    ``predictions_per_object``.
    """
    wh_mm_by_slot: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for attempt in attempts:
        if not (
            attempt.model == model
            and attempt.strategy == "BBOX"
            and attempt.split == "holdout"
            and attempt.outcome == "ok"
            and attempt.parsed is not None
            and attempt.image_width_px is not None
            and attempt.image_height_px is not None
            and attempt.object_key in holdout_objects
        ):
            continue
        wh_mm = _box_wh_mm(attempt.parsed, attempt.image_width_px, attempt.image_height_px)
        if wh_mm is not None:
            wh_mm_by_slot[(attempt.object_key, attempt.view)].append(wh_mm)

    results = []
    for object_key, (fruit_type, true_g) in holdout_objects.items():
        top = wh_mm_by_slot.get((object_key, "top"))
        side = wh_mm_by_slot.get((object_key, "side"))
        k = kfit.get(fruit_type)
        if top and side and k is not None:
            a = float(np.median([max(w, h) for w, h in top]))
            b = float(np.median([min(w, h) for w, h in top]))
            h = float(np.median([wh[1] for wh in side]))
            pred_g = k * a * b * h
            had_ok = True
        else:
            pred_g = b0_by_type[fruit_type]
            had_ok = False
        results.append(ObjectPrediction(object_key, fruit_type, true_g, pred_g, had_ok))
    return tuple(sorted(results, key=lambda p: p.object_key))


def mape(predictions: tuple[ObjectPrediction, ...]) -> float:
    """Mean absolute percentage error over objects."""
    errors = [abs(p.pred_g - p.true_g) / p.true_g for p in predictions]
    return float(np.mean(errors))


def geometric_bias(predictions: tuple[ObjectPrediction, ...]) -> float:
    """``exp(mean(log(pred/true))) - 1``. Predictions <= 0 are excluded
    (undefined log); B0 and schema-valid model predictions are always
    positive, so this only drops genuinely degenerate rows."""
    ratios = [p.pred_g / p.true_g for p in predictions if p.pred_g > 0]
    if not ratios:
        return float("nan")
    return float(np.exp(np.mean(np.log(ratios))) - 1)


def size_slope_beta(predictions: tuple[ObjectPrediction, ...]) -> float:
    """OLS of log(pred) on log(true) with fruit-type fixed effects (one
    dummy column per type, no shared intercept, so each type gets its
    own intercept and they share one slope — that slope is β).

    Rows with ``pred_g <= 0`` are excluded (undefined log). Returns
    ``nan`` if fewer than 2 usable rows remain (not enough to fit a
    slope at all).
    """
    usable = [p for p in predictions if p.pred_g > 0]
    if len(usable) < 2:
        return float("nan")

    fruit_types = sorted({p.fruit_type for p in usable})
    type_index = {fruit_type: i for i, fruit_type in enumerate(fruit_types)}
    x = np.zeros((len(usable), len(fruit_types) + 1))
    y = np.zeros(len(usable))
    for row, prediction in enumerate(usable):
        x[row, type_index[prediction.fruit_type]] = 1.0
        log_true = np.log(prediction.true_g)
        x[row, -1] = log_true
        y[row] = np.log(prediction.pred_g)

    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(x, y, rcond=None)
    return float(coefficients[-1])


def validity(attempts: tuple[AttemptRecord, ...], *, model: str, strategy: str) -> float:
    """Share of **attempted** hold-out calls with ``outcome == "ok"``.
    ``nan`` if there were no attempts at all (not 0%  — there's nothing
    to be valid or invalid about yet)."""
    relevant = [
        a for a in attempts if a.model == model and a.strategy == strategy and a.split == "holdout"
    ]
    if not relevant:
        return float("nan")
    ok_count = sum(1 for a in relevant if a.outcome == "ok")
    return ok_count / len(relevant)


def repeat_cv(attempts: tuple[AttemptRecord, ...], *, model: str, strategy: str) -> float | None:
    """Median within-image coefficient of variation across *repeats*,
    for whichever (object, view, variant) groups have >= 2 ``ok``
    repeats (the stability subset, §3 — only run for the top-2 models).
    Grouping includes ``variant`` deliberately: this measures model-
    sampling noise on one fixed image, not photo-to-photo variation —
    conflating the two would mislabel ordinary variant noise as model
    instability. Returns ``None`` if no such group exists (this
    model/strategy has no repeat-subset data), distinct from a computed
    ``0.0``."""
    preds_by_group: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for attempt in attempts:
        if (
            attempt.model == model
            and attempt.strategy == strategy
            and attempt.split == "holdout"
            and attempt.outcome == "ok"
            and attempt.pred_g is not None
        ):
            preds_by_group[(attempt.object_key, attempt.view, attempt.variant)].append(
                attempt.pred_g
            )

    cvs = [
        float(np.std(preds) / np.mean(preds))
        for preds in preds_by_group.values()
        if len(preds) >= 2 and np.mean(preds) > 0
    ]
    if not cvs:
        return None
    return float(np.median(cvs))


def cost_per_1000(
    attempts: tuple[AttemptRecord, ...], *, model: str, strategy: str
) -> float | None:
    costs = [
        a.cost_usd
        for a in attempts
        if a.model == model
        and a.strategy == strategy
        and a.split == "holdout"
        and a.cost_usd is not None
    ]
    if not costs:
        return None
    return float(np.mean(costs) * 1000)


def latency_percentiles(
    attempts: tuple[AttemptRecord, ...], *, model: str, strategy: str
) -> tuple[float, float] | None:
    latencies = [
        a.latency_ms
        for a in attempts
        if a.model == model and a.strategy == strategy and a.split == "holdout"
    ]
    if not latencies:
        return None
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))


def bootstrap_mape_diff_ci(
    predictions_a: tuple[ObjectPrediction, ...],
    predictions_b: tuple[ObjectPrediction, ...],
    *,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI for ``MAPE(a) - MAPE(b)``, paired bootstrap over objects.

    Both prediction tuples must cover the same object set (same
    hold-out), in any order — they're aligned by ``object_key``.

    Raises:
        ValueError: the two tuples don't cover the same object set.
    """
    by_key_a = {p.object_key: p for p in predictions_a}
    by_key_b = {p.object_key: p for p in predictions_b}
    if set(by_key_a) != set(by_key_b):
        raise ValueError("predictions_a and predictions_b must cover the same object set.")

    object_keys = sorted(by_key_a)
    errors_a = np.array(
        [abs(by_key_a[k].pred_g - by_key_a[k].true_g) / by_key_a[k].true_g for k in object_keys]
    )
    errors_b = np.array(
        [abs(by_key_b[k].pred_g - by_key_b[k].true_g) / by_key_b[k].true_g for k in object_keys]
    )

    rng = np.random.default_rng(seed)
    n = len(object_keys)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        diffs[i] = errors_a[sample_idx].mean() - errors_b[sample_idx].mean()

    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True)
class ModelStrategyMetrics:
    model: str
    strategy: str
    n_objects: int
    mape: float
    gain_vs_b0: float
    beta: float
    geometric_bias: float
    validity: float
    repeat_cv: float | None
    cost_per_1000: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    predictions: tuple[ObjectPrediction, ...]


def compute_metrics(
    attempts: tuple[AttemptRecord, ...],
    *,
    model: str,
    strategy: str,
    holdout_objects: dict[str, tuple[str, float]],
    b0_by_type: dict[str, float],
    kfit: dict[str, float] | None = None,
) -> ModelStrategyMetrics:
    """``kfit`` is required (and only used) for ``strategy == "BBOX"``
    (review.md §5 "E1") — the per-object mass combines both views'
    boxes via ``bbox_predictions_per_object``, unlike every other
    strategy's per-attempt ``pred_g``.

    Raises:
        ValueError: ``strategy == "BBOX"`` without ``kfit``.
    """
    if strategy == "BBOX":
        if kfit is None:
            raise ValueError('compute_metrics(strategy="BBOX", ...) requires kfit.')
        predictions = bbox_predictions_per_object(
            attempts,
            model=model,
            holdout_objects=holdout_objects,
            b0_by_type=b0_by_type,
            kfit=kfit,
        )
    else:
        predictions = predictions_per_object(
            attempts,
            model=model,
            strategy=strategy,
            holdout_objects=holdout_objects,
            b0_by_type=b0_by_type,
        )
    b0_predictions = tuple(
        ObjectPrediction(p.object_key, p.fruit_type, p.true_g, b0_by_type[p.fruit_type], False)
        for p in predictions
    )
    model_mape = mape(predictions)
    b0_mape = mape(b0_predictions)
    gain = 1 - model_mape / b0_mape if b0_mape > 0 else float("nan")

    return ModelStrategyMetrics(
        model=model,
        strategy=strategy,
        n_objects=len(predictions),
        mape=model_mape,
        gain_vs_b0=gain,
        beta=size_slope_beta(predictions),
        geometric_bias=geometric_bias(predictions),
        validity=validity(attempts, model=model, strategy=strategy),
        repeat_cv=repeat_cv(attempts, model=model, strategy=strategy),
        cost_per_1000=cost_per_1000(attempts, model=model, strategy=strategy),
        latency_p50_ms=(lat := latency_percentiles(attempts, model=model, strategy=strategy))
        and lat[0],
        latency_p95_ms=lat and lat[1],
        predictions=predictions,
    )


def passes_gates(metrics: ModelStrategyMetrics) -> bool:
    """§5's gates for the *chosen* strategy's metrics. ``repeat_cv``'s
    gate (<=5%) only applies when the stability subset exists for this
    model (§3: top-2 only) — a ``None`` doesn't fail the gate."""
    beta_low, beta_high = _BETA_RANGE
    return (
        metrics.gain_vs_b0 >= _GAIN_VS_B0_MIN
        and beta_low <= metrics.beta <= beta_high
        and abs(metrics.geometric_bias) <= _GEOMETRIC_BIAS_MAX
        and metrics.validity >= _VALIDITY_MIN
        and (metrics.repeat_cv is None or metrics.repeat_cv <= 0.05)
    )


def choose_strategy_per_model(
    metrics_s1: ModelStrategyMetrics,
    metrics_s3: ModelStrategyMetrics | None,
    metrics_bbox: ModelStrategyMetrics | None = None,
) -> ModelStrategyMetrics:
    """§5: "use S3 instead of S1 only if its MAPE is lower AND β stays in
    range; otherwise S1." Generalised to a 3rd candidate, BBOX (not in
    §5's original rule — review.md §5 "E1"): S1 is always the starting
    default; each additional candidate, in the order given, replaces the
    *current* winner only if it has a strictly lower MAPE **and** its
    own β is in range — never on MAPE alone. This is the exact
    2-candidate rule applied twice, not a new rule: for any call that
    only passes ``metrics_s3`` (``metrics_bbox=None``), this is
    bit-for-bit identical to the original.
    """
    beta_low, beta_high = _BETA_RANGE
    winner = metrics_s1
    for candidate in (metrics_s3, metrics_bbox):
        if (
            candidate is not None
            and candidate.mape < winner.mape
            and beta_low <= candidate.beta <= beta_high
        ):
            winner = candidate
    return winner


def _vendor(model: str) -> str:
    return model.split("/", 1)[0]


@dataclass(frozen=True)
class Decision:
    stop: bool
    reason: str
    default_model: str | None
    default_strategy: str | None
    fallback_model: str | None
    fallback_strategy: str | None
    #: Every model's chosen-strategy metrics, ranked by MAPE ascending —
    #: for the report, not just the winner.
    ranked: tuple[ModelStrategyMetrics, ...]
    tie_breaks: tuple[str, ...]


def decide(chosen_by_model: dict[str, ModelStrategyMetrics]) -> Decision:
    """§5's decision rule, given each model's already-strategy-chosen
    metrics (see :func:`choose_strategy_per_model`)."""
    passing = {model: m for model, m in chosen_by_model.items() if passes_gates(m)}
    if not passing:
        return Decision(
            stop=True,
            reason="No model passes all gates (gain>=30%, beta in [0.7,1.3], "
            "|bias|<=10%, validity>=98%). Stop and rethink before Phase 0.",
            default_model=None,
            default_strategy=None,
            fallback_model=None,
            fallback_strategy=None,
            ranked=tuple(sorted(chosen_by_model.values(), key=lambda m: m.mape)),
            tie_breaks=(),
        )

    ranked = sorted(passing.values(), key=lambda m: m.mape)
    best = ranked[0]
    tie_breaks: list[str] = []

    default = best
    if len(ranked) > 1:
        runner_up = ranked[1]
        ci_low, ci_high = bootstrap_mape_diff_ci(best.predictions, runner_up.predictions)
        if ci_low <= 0 <= ci_high:
            tie_breaks.append(
                f"{best.model} and {runner_up.model} are tied (bootstrap MAPE-diff CI "
                f"[{ci_low:.4f}, {ci_high:.4f}] includes 0); picking the cheaper one."
            )
            cost_best = best.cost_per_1000 if best.cost_per_1000 is not None else float("inf")
            cost_runner_up = (
                runner_up.cost_per_1000 if runner_up.cost_per_1000 is not None else float("inf")
            )
            default = best if cost_best <= cost_runner_up else runner_up

    fallback_candidates = [m for m in ranked if _vendor(m.model) != _vendor(default.model)]
    fallback = fallback_candidates[0] if fallback_candidates else None

    return Decision(
        stop=False,
        reason="Default/fallback chosen from models passing all gates, ranked by MAPE.",
        default_model=default.model,
        default_strategy=default.strategy,
        fallback_model=fallback.model if fallback else None,
        fallback_strategy=fallback.strategy if fallback else None,
        ranked=tuple(sorted(chosen_by_model.values(), key=lambda m: m.mape)),
        tie_breaks=tuple(tie_breaks),
    )
