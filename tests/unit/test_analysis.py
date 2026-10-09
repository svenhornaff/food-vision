"""prestudy.analysis: metrics against hand-computed fixtures; decision
rule pass/tie/stop. numpy only, no network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from food_vision.prestudy.analysis import (
    AttemptRecord,
    ModelStrategyMetrics,
    ObjectPrediction,
    bootstrap_mape_diff_ci,
    choose_strategy_per_model,
    compute_b0,
    compute_metrics,
    cost_per_1000,
    decide,
    geometric_bias,
    latency_percentiles,
    load_results,
    mape,
    passes_gates,
    predictions_per_object,
    repeat_cv,
    size_slope_beta,
    validity,
)
from food_vision.prestudy.ecustfd import Item


def _attempt(
    *,
    model: str = "vendor/model-a",
    strategy: str = "S1",
    repeat: int = 0,
    object_key: str = "apple001",
    fruit_type: str = "apple",
    view: str = "top",
    split: str = "holdout",
    true_g: float = 200.0,
    pred_g: float | None = 190.0,
    outcome: str = "ok",
    cost_usd: float | None = 0.001,
    latency_ms: float = 100.0,
) -> AttemptRecord:
    return AttemptRecord(
        model=model,
        strategy=strategy,
        repeat=repeat,
        object_key=object_key,
        fruit_type=fruit_type,
        view=view,
        split=split,
        true_g=true_g,
        pred_g=pred_g,
        outcome=outcome,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def _prediction(true_g: float, pred_g: float, fruit_type: str = "apple") -> ObjectPrediction:
    return ObjectPrediction("obj", fruit_type, true_g, pred_g, True)


class TestLoadResults:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_results(tmp_path / "nope.jsonl") == ()

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        good = {
            "model": "m",
            "strategy": "S1",
            "repeat": 0,
            "object_key": "o",
            "fruit_type": "apple",
            "view": "top",
            "split": "holdout",
            "true_g": 100.0,
            "pred_g": 90.0,
            "outcome": "ok",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
        }
        path.write_text(json.dumps(good) + "\n\n\n")

        assert len(load_results(path)) == 1

    def test_loads_valid_lines_and_skips_malformed(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        good = {
            "model": "m",
            "strategy": "S1",
            "repeat": 0,
            "object_key": "o",
            "fruit_type": "apple",
            "view": "top",
            "split": "holdout",
            "true_g": 100.0,
            "pred_g": 90.0,
            "outcome": "ok",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
        }
        path.write_text(json.dumps(good) + "\nnot json\n")

        records = load_results(path)

        assert len(records) == 1
        assert records[0].model == "m"


class TestComputeB0:
    def test_mean_dev_weight_per_type(self) -> None:
        items = (
            Item(Path("a.JPG"), "a", "apple", "top", 100.0, "dev", 1),
            Item(Path("b.JPG"), "b", "apple", "top", 200.0, "dev", 1),
            Item(Path("c.JPG"), "c", "apple", "top", 9999.0, "holdout", 1),
        )
        assert compute_b0(items) == {"apple": 150.0}


class TestPredictionsPerObject:
    def test_median_of_ok_attempts(self) -> None:
        attempts = (
            _attempt(object_key="a", pred_g=100.0),
            _attempt(object_key="a", pred_g=200.0),
            _attempt(object_key="a", pred_g=300.0),
        )
        holdout_objects = {"a": ("apple", 200.0)}

        predictions = predictions_per_object(
            attempts,
            model="vendor/model-a",
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type={},
        )

        assert predictions[0].pred_g == 200.0
        assert predictions[0].had_ok_result is True

    def test_b0_fallback_when_no_ok_attempts(self) -> None:
        attempts = (_attempt(object_key="a", outcome="invalid", pred_g=None),)
        holdout_objects = {"a": ("apple", 200.0)}

        predictions = predictions_per_object(
            attempts,
            model="vendor/model-a",
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type={"apple": 150.0},
        )

        assert predictions[0].pred_g == 150.0
        assert predictions[0].had_ok_result is False

    def test_object_with_no_attempts_at_all_gets_b0(self) -> None:
        holdout_objects = {"a": ("apple", 200.0)}

        predictions = predictions_per_object(
            (),
            model="m",
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type={"apple": 150.0},
        )

        assert predictions[0].pred_g == 150.0

    def test_attempts_for_objects_outside_holdout_set_are_ignored(self) -> None:
        attempts = (_attempt(object_key="not-in-holdout", pred_g=999.0),)
        holdout_objects = {"a": ("apple", 200.0)}

        predictions = predictions_per_object(
            attempts,
            model="m",
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type={"apple": 150.0},
        )

        assert len(predictions) == 1
        assert predictions[0].pred_g == 150.0  # B0, not 999.0


class TestMape:
    def test_known_value(self) -> None:
        predictions = (_prediction(100.0, 110.0), _prediction(200.0, 180.0))
        assert mape(predictions) == pytest.approx(0.1)


class TestGeometricBias:
    def test_known_value(self) -> None:
        predictions = (_prediction(100.0, 110.0), _prediction(100.0, 90.0))
        expected = math.exp((math.log(1.1) + math.log(0.9)) / 2) - 1
        assert geometric_bias(predictions) == pytest.approx(expected)

    def test_nan_when_no_positive_predictions(self) -> None:
        predictions = (_prediction(100.0, 0.0),)
        assert math.isnan(geometric_bias(predictions))


class TestSizeSlopeBeta:
    def test_perfect_prediction_gives_beta_one(self) -> None:
        predictions = (
            _prediction(100.0, 100.0, "apple"),
            _prediction(200.0, 200.0, "apple"),
            _prediction(50.0, 50.0, "banana"),
            _prediction(150.0, 150.0, "banana"),
        )
        assert size_slope_beta(predictions) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_scaling_still_gives_beta_one(self) -> None:
        """A constant multiplicative bias (pred = 2x true) shouldn't move
        beta -- that's what geometric_bias is for."""
        predictions = (
            _prediction(100.0, 200.0, "apple"),
            _prediction(200.0, 400.0, "apple"),
            _prediction(50.0, 100.0, "banana"),
            _prediction(150.0, 300.0, "banana"),
        )
        assert size_slope_beta(predictions) == pytest.approx(1.0, abs=1e-6)

    def test_matches_independent_ols_for_a_single_type(self) -> None:
        true_values = [50.0, 100.0, 150.0, 200.0]
        pred_values = [90.0, 100.0, 110.0, 120.0]  # compressed towards the mean
        predictions = tuple(
            _prediction(t, p) for t, p in zip(true_values, pred_values, strict=True)
        )

        expected_beta, _intercept = np.polyfit(np.log(true_values), np.log(pred_values), 1)

        assert size_slope_beta(predictions) == pytest.approx(float(expected_beta), abs=1e-6)
        assert 0 < size_slope_beta(predictions) < 1  # compressed guesses: beta < 1

    def test_nan_with_fewer_than_two_usable_rows(self) -> None:
        assert math.isnan(size_slope_beta((_prediction(100.0, 100.0),)))


class TestValidity:
    def test_known_ratio(self) -> None:
        attempts = (
            _attempt(outcome="ok"),
            _attempt(outcome="ok"),
            _attempt(outcome="invalid"),
        )
        assert validity(attempts, model="vendor/model-a", strategy="S1") == pytest.approx(2 / 3)

    def test_nan_when_no_attempts(self) -> None:
        assert math.isnan(validity((), model="m", strategy="S1"))


class TestRepeatCv:
    def test_median_cv_across_qualifying_groups(self) -> None:
        attempts = (
            _attempt(object_key="a", view="top", repeat=0, pred_g=100.0),
            _attempt(object_key="a", view="top", repeat=1, pred_g=110.0),
            _attempt(object_key="a", view="top", repeat=2, pred_g=90.0),
            _attempt(object_key="b", view="top", repeat=0, pred_g=200.0),  # single repeat: excluded
        )
        expected = float(np.std([100.0, 110.0, 90.0]) / np.mean([100.0, 110.0, 90.0]))

        assert repeat_cv(attempts, model="vendor/model-a", strategy="S1") == pytest.approx(expected)

    def test_none_when_no_qualifying_group(self) -> None:
        attempts = (_attempt(object_key="a", repeat=0, pred_g=100.0),)
        assert repeat_cv(attempts, model="vendor/model-a", strategy="S1") is None


class TestCostAndLatency:
    def test_cost_per_1000(self) -> None:
        attempts = (_attempt(cost_usd=0.001), _attempt(cost_usd=0.002))
        assert cost_per_1000(attempts, model="vendor/model-a", strategy="S1") == pytest.approx(1.5)

    def test_cost_per_1000_none_when_no_cost_data(self) -> None:
        attempts = (_attempt(cost_usd=None),)
        assert cost_per_1000(attempts, model="vendor/model-a", strategy="S1") is None

    def test_latency_percentiles(self) -> None:
        attempts = tuple(_attempt(latency_ms=float(v)) for v in range(1, 101))
        p50, p95 = latency_percentiles(attempts, model="vendor/model-a", strategy="S1") or (0, 0)
        assert p50 == pytest.approx(50.5, abs=1.0)
        assert p95 == pytest.approx(95.05, abs=1.0)

    def test_latency_percentiles_none_when_no_data(self) -> None:
        assert latency_percentiles((), model="m", strategy="S1") is None


class TestBootstrapMapeDiffCi:
    def test_identical_predictions_give_zero_width_ci_at_zero(self) -> None:
        predictions = (_prediction(100.0, 110.0), _prediction(200.0, 180.0))

        low, high = bootstrap_mape_diff_ci(predictions, predictions, n_resamples=100)

        assert low == pytest.approx(0.0)
        assert high == pytest.approx(0.0)

    def test_clearly_different_predictions_give_ci_excluding_zero(self) -> None:
        good = tuple(_prediction(100.0 * i, 100.0 * i) for i in range(1, 6))
        bad = tuple(_prediction(100.0 * i, 300.0 * i) for i in range(1, 6))

        low, high = bootstrap_mape_diff_ci(good, bad, n_resamples=500)

        assert high < 0  # good's MAPE is reliably lower than bad's

    def test_mismatched_object_sets_raise(self) -> None:
        a = (ObjectPrediction("x", "apple", 100.0, 100.0, True),)
        b = (ObjectPrediction("y", "apple", 100.0, 100.0, True),)

        with pytest.raises(ValueError, match="same object set"):
            bootstrap_mape_diff_ci(a, b)


class TestComputeMetrics:
    def test_integrates_predictions_and_gates_inputs(self) -> None:
        attempts = (
            _attempt(
                object_key="a", fruit_type="apple", pred_g=100.0, true_g=100.0, cost_usd=0.002
            ),
            _attempt(
                object_key="b", fruit_type="apple", outcome="invalid", pred_g=None, true_g=200.0
            ),
        )
        holdout_objects = {"a": ("apple", 100.0), "b": ("apple", 200.0)}

        metrics = compute_metrics(
            attempts,
            model="vendor/model-a",
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type={"apple": 150.0},
        )

        assert metrics.n_objects == 2
        assert metrics.validity == pytest.approx(0.5)
        # object b falls back to B0 (150.0); its error vs B0's own B0-based
        # MAPE is 0 for that object, so gain_vs_b0 reflects only object a.
        assert metrics.mape == pytest.approx(
            mape(
                predictions_per_object(
                    attempts,
                    model="vendor/model-a",
                    strategy="S1",
                    holdout_objects=holdout_objects,
                    b0_by_type={"apple": 150.0},
                )
            )
        )


def _metrics(
    *,
    model: str = "vendor/model-a",
    strategy: str = "S1",
    mape_value: float = 0.1,
    gain_vs_b0: float = 0.5,
    beta: float = 1.0,
    geometric_bias_value: float = 0.0,
    validity_value: float = 1.0,
    repeat_cv_value: float | None = None,
    cost: float | None = 1.0,
    predictions: tuple[ObjectPrediction, ...] = (),
) -> ModelStrategyMetrics:
    return ModelStrategyMetrics(
        model=model,
        strategy=strategy,
        n_objects=len(predictions),
        mape=mape_value,
        gain_vs_b0=gain_vs_b0,
        beta=beta,
        geometric_bias=geometric_bias_value,
        validity=validity_value,
        repeat_cv=repeat_cv_value,
        cost_per_1000=cost,
        latency_p50_ms=None,
        latency_p95_ms=None,
        predictions=predictions,
    )


class TestPassesGates:
    def test_all_gates_pass(self) -> None:
        assert passes_gates(_metrics()) is True

    def test_fails_on_low_gain(self) -> None:
        assert passes_gates(_metrics(gain_vs_b0=0.1)) is False

    def test_fails_on_beta_out_of_range(self) -> None:
        assert passes_gates(_metrics(beta=1.5)) is False

    def test_fails_on_bias_out_of_range(self) -> None:
        assert passes_gates(_metrics(geometric_bias_value=0.2)) is False

    def test_fails_on_low_validity(self) -> None:
        assert passes_gates(_metrics(validity_value=0.9)) is False

    def test_none_repeat_cv_does_not_fail(self) -> None:
        assert passes_gates(_metrics(repeat_cv_value=None)) is True

    def test_high_repeat_cv_fails(self) -> None:
        assert passes_gates(_metrics(repeat_cv_value=0.10)) is False


class TestChooseStrategyPerModel:
    def test_s3_chosen_when_better_mape_and_beta_in_range(self) -> None:
        s1 = _metrics(strategy="S1", mape_value=0.2, beta=1.0)
        s3 = _metrics(strategy="S3", mape_value=0.1, beta=1.0)

        assert choose_strategy_per_model(s1, s3) is s3

    def test_s1_chosen_when_s3_beta_out_of_range(self) -> None:
        s1 = _metrics(strategy="S1", mape_value=0.2, beta=1.0)
        s3 = _metrics(strategy="S3", mape_value=0.1, beta=2.0)

        assert choose_strategy_per_model(s1, s3) is s1

    def test_s1_chosen_when_s3_mape_worse(self) -> None:
        s1 = _metrics(strategy="S1", mape_value=0.1, beta=1.0)
        s3 = _metrics(strategy="S3", mape_value=0.2, beta=1.0)

        assert choose_strategy_per_model(s1, s3) is s1

    def test_s1_chosen_when_s3_absent(self) -> None:
        s1 = _metrics(strategy="S1")
        assert choose_strategy_per_model(s1, None) is s1


class TestDecide:
    def test_stop_when_no_model_passes(self) -> None:
        chosen = {"vendor-a/x": _metrics(model="vendor-a/x", validity_value=0.5)}

        decision = decide(chosen)

        assert decision.stop is True
        assert decision.default_model is None

    def test_default_and_fallback_from_different_vendors(self) -> None:
        preds_a = (_prediction(100.0, 101.0),)
        preds_b = (_prediction(100.0, 150.0),)
        chosen = {
            "vendor-a/x": _metrics(model="vendor-a/x", mape_value=0.01, predictions=preds_a),
            "vendor-b/y": _metrics(model="vendor-b/y", mape_value=0.5, predictions=preds_b),
        }

        decision = decide(chosen)

        assert decision.stop is False
        assert decision.default_model == "vendor-a/x"
        assert decision.fallback_model == "vendor-b/y"

    def test_tied_models_pick_cheaper(self) -> None:
        shared_predictions = (_prediction(100.0, 110.0), _prediction(200.0, 180.0))
        chosen = {
            "vendor-a/x": _metrics(
                model="vendor-a/x", mape_value=0.1, cost=5.0, predictions=shared_predictions
            ),
            "vendor-b/y": _metrics(
                model="vendor-b/y", mape_value=0.1, cost=1.0, predictions=shared_predictions
            ),
        }

        decision = decide(chosen)

        assert decision.default_model == "vendor-b/y"  # cheaper, tied on MAPE
        assert decision.tie_breaks != ()

    def test_no_fallback_when_only_one_vendor_passes(self) -> None:
        chosen = {
            "vendor-a/x": _metrics(model="vendor-a/x", predictions=(_prediction(100.0, 101.0),))
        }

        decision = decide(chosen)

        assert decision.default_model == "vendor-a/x"
        assert decision.fallback_model is None
