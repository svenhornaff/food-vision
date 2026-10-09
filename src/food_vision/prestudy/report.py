"""Write ``report.md``, ``scatter.png`` and ``decision.md`` (L4 exit).

docs/dev/pre-study.md §4.2/§7. ``matplotlib`` with the ``Agg`` backend
(headless, no display needed) — this module is the only place in the
pre-study that imports it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from food_vision.prestudy.analysis import Decision, ModelStrategyMetrics

__all__ = ["write_report_md", "write_scatter_png", "write_decision_md"]


def write_report_md(
    path: Path, ranked: tuple[ModelStrategyMetrics, ...], decision: Decision
) -> None:
    """A metrics table (every model's chosen strategy, ranked by MAPE)
    plus the decision, as markdown."""
    lines = [
        "# Pre-study report (ECUSTFD, hold-out)",
        "",
        "| Model | Strategy | n | MAPE | Gain vs B0 | β | Bias | Validity | Repeat CV | "
        "$/1000 | P50 ms | P95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ranked:
        lines.append(
            "| {model} | {strategy} | {n} | {mape:.3f} | {gain:.1%} | {beta:.2f} | {bias:+.1%} | "
            "{validity:.1%} | {cv} | {cost} | {p50} | {p95} |".format(
                model=m.model,
                strategy=m.strategy,
                n=m.n_objects,
                mape=m.mape,
                gain=m.gain_vs_b0,
                beta=m.beta,
                bias=m.geometric_bias,
                validity=m.validity,
                cv=f"{m.repeat_cv:.1%}" if m.repeat_cv is not None else "—",
                cost=f"${m.cost_per_1000:.2f}" if m.cost_per_1000 is not None else "—",
                p50=f"{m.latency_p50_ms:.0f}" if m.latency_p50_ms is not None else "—",
                p95=f"{m.latency_p95_ms:.0f}" if m.latency_p95_ms is not None else "—",
            )
        )

    lines += ["", "## Decision", "", decision.reason, ""]
    if decision.stop:
        lines.append("**Result: stop.**")
    else:
        lines.append(f"**Default:** `{decision.default_model}` ({decision.default_strategy})")
        if decision.fallback_model:
            lines.append(
                f"**Fallback:** `{decision.fallback_model}` ({decision.fallback_strategy})"
            )
        else:
            lines.append("**Fallback:** none (no passing model from a different vendor).")
    for note in decision.tie_breaks:
        lines.append(f"\n_{note}_")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scatter_png(path: Path, ranked: tuple[ModelStrategyMetrics, ...]) -> None:
    """Predicted vs. true, log-log, one panel per model (§4.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ranked:
        # An empty sweep still needs a valid (empty) figure, not a crash —
        # the CLI's exit criterion is "scatter.png is written", full stop.
        fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.savefig(path)
        plt.close(fig)
        return

    n_models = len(ranked)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    for ax, metrics in zip(axes[0], ranked, strict=True):
        true_values = [p.true_g for p in metrics.predictions]
        pred_values = [p.pred_g for p in metrics.predictions]
        ax.scatter(true_values, pred_values, s=20)
        if true_values:
            lo, hi = min(true_values + pred_values), max(true_values + pred_values)
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("true_g")
        ax.set_ylabel("pred_g")
        ax.set_title(f"{metrics.model}\n{metrics.strategy}")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_decision_md(path: Path, decision: Decision) -> None:
    """The decision alone, as its own file (§7's three named outputs)."""
    lines = ["# Pre-study decision", "", decision.reason, ""]
    if decision.stop:
        lines.append("**Result: stop.**")
    else:
        lines.append(f"**Default model:** `{decision.default_model}`")
        lines.append(f"**Default strategy:** {decision.default_strategy}")
        if decision.fallback_model:
            lines.append(f"**Fallback model:** `{decision.fallback_model}`")
            lines.append(f"**Fallback strategy:** {decision.fallback_strategy}")
        else:
            lines.append("**Fallback:** none (no passing model from a different vendor).")
    for note in decision.tie_breaks:
        lines.append(f"\n_{note}_")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
