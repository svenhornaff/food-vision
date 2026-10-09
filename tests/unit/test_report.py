"""prestudy.report: smoke tests only (no pixel-level diffing, per
docs/dev/pre-study.md §9)."""

from __future__ import annotations

from pathlib import Path

from food_vision.prestudy.analysis import Decision, ModelStrategyMetrics, ObjectPrediction
from food_vision.prestudy.report import write_decision_md, write_report_md, write_scatter_png

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _metrics(model: str, strategy: str, *, mape_value: float = 0.1) -> ModelStrategyMetrics:
    predictions = (
        ObjectPrediction("a", "apple", 100.0, 105.0, True),
        ObjectPrediction("b", "apple", 200.0, 190.0, True),
    )
    return ModelStrategyMetrics(
        model=model,
        strategy=strategy,
        n_objects=2,
        mape=mape_value,
        gain_vs_b0=0.4,
        beta=1.0,
        geometric_bias=0.02,
        validity=0.99,
        repeat_cv=0.03,
        cost_per_1000=1.5,
        latency_p50_ms=1000.0,
        latency_p95_ms=2000.0,
        predictions=predictions,
    )


def _passing_decision() -> Decision:
    ranked = (_metrics("vendor-a/x", "S1"), _metrics("vendor-b/y", "S1", mape_value=0.2))
    return Decision(
        stop=False,
        reason="Default/fallback chosen from models passing all gates, ranked by MAPE.",
        default_model="vendor-a/x",
        default_strategy="S1",
        fallback_model="vendor-b/y",
        fallback_strategy="S1",
        ranked=ranked,
        tie_breaks=(),
    )


def _stop_decision() -> Decision:
    return Decision(
        stop=True,
        reason="No model passes all gates.",
        default_model=None,
        default_strategy=None,
        fallback_model=None,
        fallback_strategy=None,
        ranked=(_metrics("vendor-a/x", "S1"),),
        tie_breaks=(),
    )


class TestWriteReportMd:
    def test_contains_model_rows_and_decision(self, tmp_path: Path) -> None:
        decision = _passing_decision()
        path = tmp_path / "report.md"

        write_report_md(path, decision.ranked, decision)

        text = path.read_text()
        assert "vendor-a/x" in text
        assert "vendor-b/y" in text
        assert "## Decision" in text
        assert "**Default:**" in text
        assert "**Fallback:**" in text

    def test_stop_decision_renders_stop(self, tmp_path: Path) -> None:
        decision = _stop_decision()
        path = tmp_path / "report.md"

        write_report_md(path, decision.ranked, decision)

        assert "**Result: stop.**" in path.read_text()

    def test_tie_break_notes_included(self, tmp_path: Path) -> None:
        decision = _passing_decision()
        decision_with_note = Decision(
            **{**decision.__dict__, "tie_breaks": ("models x and y are tied",)}
        )
        path = tmp_path / "report.md"

        write_report_md(path, decision_with_note.ranked, decision_with_note)

        assert "models x and y are tied" in path.read_text()

    def test_no_fallback_case(self, tmp_path: Path) -> None:
        decision = Decision(
            stop=False,
            reason="r",
            default_model="vendor-a/x",
            default_strategy="S1",
            fallback_model=None,
            fallback_strategy=None,
            ranked=(),
            tie_breaks=(),
        )
        path = tmp_path / "report.md"

        write_report_md(path, (), decision)

        assert "**Fallback:** none" in path.read_text()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "report.md"
        write_report_md(path, (), _stop_decision())
        assert path.exists()


class TestWriteDecisionMd:
    def test_passing_decision_has_default_and_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "decision.md"
        write_decision_md(path, _passing_decision())

        text = path.read_text()
        assert "`vendor-a/x`" in text
        assert "`vendor-b/y`" in text

    def test_stop_decision_has_no_default(self, tmp_path: Path) -> None:
        path = tmp_path / "decision.md"
        write_decision_md(path, _stop_decision())

        text = path.read_text()
        assert "**Result: stop.**" in text
        assert "Default model" not in text

    def test_no_fallback_case(self, tmp_path: Path) -> None:
        decision = Decision(
            stop=False,
            reason="r",
            default_model="vendor-a/x",
            default_strategy="S1",
            fallback_model=None,
            fallback_strategy=None,
            ranked=(),
            tie_breaks=(),
        )
        path = tmp_path / "decision.md"

        write_decision_md(path, decision)

        assert "none" in path.read_text()

    def test_tie_break_notes_included(self, tmp_path: Path) -> None:
        decision = Decision(**{**_passing_decision().__dict__, "tie_breaks": ("x and y are tied",)})
        path = tmp_path / "decision.md"

        write_decision_md(path, decision)

        assert "x and y are tied" in path.read_text()


class TestWriteScatterPng:
    def test_writes_a_valid_png_with_one_panel_per_model(self, tmp_path: Path) -> None:
        ranked = (_metrics("vendor-a/x", "S1"), _metrics("vendor-b/y", "S1"))
        path = tmp_path / "scatter.png"

        write_scatter_png(path, ranked)

        assert path.exists()
        assert path.read_bytes()[:8] == _PNG_MAGIC
        assert path.stat().st_size > 0

    def test_empty_ranking_still_writes_a_valid_png(self, tmp_path: Path) -> None:
        path = tmp_path / "scatter.png"

        write_scatter_png(path, ())

        assert path.read_bytes()[:8] == _PNG_MAGIC

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "scatter.png"
        write_scatter_png(path, ())
        assert path.exists()
