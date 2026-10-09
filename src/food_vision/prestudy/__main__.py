"""``python -m food_vision.prestudy {prepare,run,analyze}``.

docs/dev/pre-study.md §4.2/§7.7: argparse, no ``typer``, no web UI, no
database. This module wires the library functions together; it holds no
logic of its own worth unit-testing beyond argument parsing (covered by
``tests/unit/test_prestudy_cli.py`` using ``argv`` + a monkeypatched
entry point, not a live run).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from food_vision.observation.schema import ObservationStrategy
from food_vision.prestudy.analysis import (
    choose_strategy_per_model,
    compute_b0,
    compute_metrics,
    decide,
    load_results,
)
from food_vision.prestudy.ecustfd import DEFAULT_TYPES, download, prepare, read_split_csv
from food_vision.prestudy.report import write_decision_md, write_report_md, write_scatter_png
from food_vision.prestudy.run import RunConfig, load_models_toml, run_sweep
from food_vision.proxy.model_provider import get_default_provider
from food_vision.utils.log_factory import configure_logging, get_logger

logger = get_logger(__name__)

_DEFAULT_SPLIT_CSV = Path("bench/runs/prestudy-lean/split.csv")
_DEFAULT_RESULTS = Path("bench/runs/prestudy-lean/results.jsonl")
_DEFAULT_MODELS_TOML = Path("bench/runs/prestudy-lean/models.toml")
_DEFAULT_REPORT_DIR = Path("bench/reports/prestudy-lean")


def _cmd_prepare(args: argparse.Namespace) -> int:
    report = prepare(types=DEFAULT_TYPES, split_csv_path=args.split_csv)

    print(f"grouping key used: {report.grouping.key_used}")
    if report.grouping.fallback_triggered:
        print(f"  fallback triggered for: {report.grouping.inconsistent_object_ids}")
    print(f"excluded (multi-object) images: {report.excluded_multi_object_images}")
    print(f"unparsed filenames: {len(report.unparsed_filenames)}")

    by_type = Counter(item.fruit_type for item in report.items)
    by_split = Counter(item.split for item in report.items)
    by_view = Counter(item.view for item in report.items)
    print(f"items: {len(report.items)}")
    print(f"  by type:  {dict(sorted(by_type.items()))}")
    print(f"  by split: {dict(sorted(by_split.items()))}")
    print(f"  by view:  {dict(sorted(by_view.items()))}")
    print(f"wrote {args.split_csv}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    items = read_split_csv(args.split_csv)
    models_by_id = load_models_toml(args.models_toml)
    requested_models = [m.strip() for m in args.models.split(",")]
    missing = [m for m in requested_models if m not in models_by_id]
    if missing:
        print(
            f"Missing models.toml entries for: {missing}. Add a [[models]] block with at "
            f"least model/provider_pin before running.",
            file=sys.stderr,
        )
        return 1
    models = tuple(models_by_id[m] for m in requested_models)
    strategies = tuple(ObservationStrategy(s.strip()) for s in args.strategies.split(","))

    snapshot_dir = download()
    config = RunConfig(
        items=items,
        models=models,
        strategies=strategies,
        results_path=args.results,
        snapshot_dir=snapshot_dir,
        repeats=args.repeats,
        split_filter=args.split,
        dry_run_n=args.dry_run,
        budget_usd=args.budget,
    )
    provider = get_default_provider()
    stats = run_sweep(provider, config)

    for model, model_stats in stats.by_model.items():
        print(
            f"{model}: attempted={model_stats.attempted} completed={model_stats.completed} "
            f"skipped={model_stats.skipped_existing} cost_usd={model_stats.cost_usd:.4f}"
        )
        if args.dry_run is not None and model_stats.completed > 0:
            planned_per_model = (
                len([i for i in items if args.split == "all" or i.split == args.split])
                * len(strategies)
                * args.repeats
            )
            extrapolated = model_stats.cost_usd / model_stats.completed * planned_per_model
            print(f"  extrapolated full-sweep cost for this model: ${extrapolated:.2f}")
    print(f"stopped_reason={stats.stopped_reason}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    attempts = load_results(args.results)
    items = read_split_csv(args.split_csv)
    b0_by_type = compute_b0(items)
    holdout_objects = {
        item.object_key: (item.fruit_type, item.weight_g)
        for item in items
        if item.split == "holdout"
    }

    attempted_models = sorted({a.model for a in attempts})
    chosen_by_model = {}
    for model in attempted_models:
        s1 = compute_metrics(
            attempts,
            model=model,
            strategy="S1",
            holdout_objects=holdout_objects,
            b0_by_type=b0_by_type,
        )
        s3 = None
        if any(a.strategy == "S3" and a.model == model for a in attempts):
            s3 = compute_metrics(
                attempts,
                model=model,
                strategy="S3",
                holdout_objects=holdout_objects,
                b0_by_type=b0_by_type,
            )
        chosen_by_model[model] = choose_strategy_per_model(s1, s3)

    decision = decide(chosen_by_model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_report_md(args.out_dir / "report.md", decision.ranked, decision)
    write_scatter_png(args.out_dir / "scatter.png", decision.ranked)
    write_decision_md(args.out_dir / "decision.md", decision)

    print(f"wrote {args.out_dir / 'report.md'}")
    print(f"wrote {args.out_dir / 'scatter.png'}")
    print(f"wrote {args.out_dir / 'decision.md'}")
    print(decision.reason)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m food_vision.prestudy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="download ECUSTFD, write split.csv")
    prepare_parser.add_argument("--split-csv", type=Path, default=_DEFAULT_SPLIT_CSV)
    prepare_parser.set_defaults(handler=_cmd_prepare)

    run_parser = subparsers.add_parser("run", help="run the model sweep (live OpenRouter calls)")
    run_parser.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    run_parser.add_argument("--strategies", default="S1,S3")
    run_parser.add_argument("--split", default="holdout", choices=("dev", "holdout", "all"))
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--dry-run", type=int, default=None, metavar="N")
    run_parser.add_argument("--budget", type=float, default=None, metavar="USD")
    run_parser.add_argument("--split-csv", type=Path, default=_DEFAULT_SPLIT_CSV)
    run_parser.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    run_parser.add_argument("--models-toml", type=Path, default=_DEFAULT_MODELS_TOML)
    run_parser.set_defaults(handler=_cmd_run)

    analyze_parser = subparsers.add_parser("analyze", help="compute metrics, write report/decision")
    analyze_parser.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    analyze_parser.add_argument("--split-csv", type=Path, default=_DEFAULT_SPLIT_CSV)
    analyze_parser.add_argument("--out-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    analyze_parser.set_defaults(handler=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
