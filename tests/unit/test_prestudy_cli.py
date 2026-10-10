"""prestudy.__main__: argument parsing and command wiring.

No network and no live OpenRouter calls — ``prepare``/``download``/
``get_default_provider``/``run_sweep`` are monkeypatched for ``run`` and
``prepare`` tests; ``analyze`` is exercised for real against tiny
synthetic fixtures (no network involved in analysis at all).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import food_vision.prestudy.__main__ as cli
from food_vision.prestudy.ecustfd import GroupingResult, Item, PrepareReport
from food_vision.prestudy.run import ModelRunStats, RunStats


class TestBuildParser:
    def test_prepare_defaults(self) -> None:
        args = cli.build_parser().parse_args(["prepare"])
        assert args.command == "prepare"
        assert args.split_csv == cli._DEFAULT_SPLIT_CSV

    def test_run_requires_models(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run"])

    def test_run_defaults(self) -> None:
        args = cli.build_parser().parse_args(["run", "--models", "vendor/a"])
        assert args.strategies == "S1,S3"
        assert args.split == "holdout"
        assert args.repeats == 1
        assert args.dry_run is None
        assert args.budget is None
        assert args.max_variants_per_object == 1

    def test_run_max_variants_per_object_accepts_all(self) -> None:
        args = cli.build_parser().parse_args(
            ["run", "--models", "vendor/a", "--max-variants-per-object", "all"]
        )
        assert args.max_variants_per_object is None

    def test_run_max_variants_per_object_accepts_int(self) -> None:
        args = cli.build_parser().parse_args(
            ["run", "--models", "vendor/a", "--max-variants-per-object", "3"]
        )
        assert args.max_variants_per_object == 3

    def test_run_max_variants_per_object_rejects_zero(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["run", "--models", "vendor/a", "--max-variants-per-object", "0"]
            )

    def test_run_dry_run_and_budget_parsed(self) -> None:
        args = cli.build_parser().parse_args(
            ["run", "--models", "vendor/a", "--dry-run", "5", "--budget", "2.5"]
        )
        assert args.dry_run == 5
        assert args.budget == 2.5

    def test_analyze_defaults(self) -> None:
        args = cli.build_parser().parse_args(["analyze"])
        assert args.out_dir == cli._DEFAULT_REPORT_DIR


class TestCmdPrepare:
    def test_prints_summary(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        items = (
            Item(Path("a.JPG"), "a", "apple", "top", 100.0, "dev", 1),
            Item(Path("b.JPG"), "b", "banana", "side", 150.0, "holdout", 1),
        )
        report = PrepareReport(
            items=items,
            grouping=GroupingResult("portions_id", False, 72, ()),
            excluded_multi_object_images=0,
            unparsed_filenames=(),
        )
        monkeypatch.setattr(cli, "prepare", lambda **kwargs: report)

        exit_code = cli.main(["prepare", "--split-csv", str(tmp_path / "split.csv")])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "grouping key used: portions_id" in out
        assert "items: 2" in out


class TestCmdRun:
    def test_missing_models_toml_entry_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        split_csv = tmp_path / "split.csv"
        split_csv.write_text("object_key,fruit_type,view,image_file,weight_g,split,variant\n")
        models_toml = tmp_path / "models.toml"
        models_toml.write_text("")  # no [[models]] entries at all

        exit_code = cli.main(
            [
                "run",
                "--models",
                "vendor/missing",
                "--split-csv",
                str(split_csv),
                "--models-toml",
                str(models_toml),
            ]
        )

        assert exit_code == 1
        assert "vendor/missing" in capsys.readouterr().err

    def test_dry_run_prints_extrapolated_cost(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        split_csv = tmp_path / "split.csv"
        split_csv.write_text(
            "object_key,fruit_type,view,image_file,weight_g,split,variant\n"
            "a,apple,top,a.JPG,100.0,holdout,1\n"
            "b,apple,top,b.JPG,120.0,holdout,1\n"
        )
        models_toml = tmp_path / "models.toml"
        models_toml.write_text('[[models]]\nmodel = "vendor/a"\nprovider_pin = "vendor-provider"\n')

        stats = RunStats(
            by_model={"vendor/a": ModelRunStats(attempted=1, completed=1, cost_usd=0.01)}
        )
        monkeypatch.setattr(cli, "download", lambda: tmp_path)
        monkeypatch.setattr(cli, "get_default_provider", lambda: object())
        monkeypatch.setattr(cli, "run_sweep", lambda provider, config: stats)

        exit_code = cli.main(
            [
                "run",
                "--models",
                "vendor/a",
                "--dry-run",
                "1",
                "--split-csv",
                str(split_csv),
                "--models-toml",
                str(models_toml),
                "--results",
                str(tmp_path / "results.jsonl"),
            ]
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "extrapolated full-sweep cost" in out

    def test_max_variants_per_object_default_selects_lowest_variant_only(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """split.csv has 2 variants for the same (object, view); the
        default ``--max-variants-per-object 1`` must select only one."""
        split_csv = tmp_path / "split.csv"
        split_csv.write_text(
            "object_key,fruit_type,view,image_file,weight_g,split,variant\n"
            "a,apple,top,a1.JPG,100.0,holdout,1\n"
            "a,apple,top,a2.JPG,100.0,holdout,2\n"
        )
        models_toml = tmp_path / "models.toml"
        models_toml.write_text('[[models]]\nmodel = "vendor/a"\nprovider_pin = "vendor-provider"\n')
        monkeypatch.setattr(cli, "download", lambda: tmp_path)
        monkeypatch.setattr(cli, "get_default_provider", lambda: object())
        monkeypatch.setattr(cli, "run_sweep", lambda provider, config: RunStats(by_model={}))

        exit_code = cli.main(
            [
                "run",
                "--models",
                "vendor/a",
                "--split-csv",
                str(split_csv),
                "--models-toml",
                str(models_toml),
                "--results",
                str(tmp_path / "results.jsonl"),
            ]
        )

        assert exit_code == 0
        assert "selected items: 1 (max_variants_per_object=1)" in capsys.readouterr().out

    def test_max_variants_per_object_all_selects_both(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        split_csv = tmp_path / "split.csv"
        split_csv.write_text(
            "object_key,fruit_type,view,image_file,weight_g,split,variant\n"
            "a,apple,top,a1.JPG,100.0,holdout,1\n"
            "a,apple,top,a2.JPG,100.0,holdout,2\n"
        )
        models_toml = tmp_path / "models.toml"
        models_toml.write_text('[[models]]\nmodel = "vendor/a"\nprovider_pin = "vendor-provider"\n')
        monkeypatch.setattr(cli, "download", lambda: tmp_path)
        monkeypatch.setattr(cli, "get_default_provider", lambda: object())
        monkeypatch.setattr(cli, "run_sweep", lambda provider, config: RunStats(by_model={}))

        exit_code = cli.main(
            [
                "run",
                "--models",
                "vendor/a",
                "--max-variants-per-object",
                "all",
                "--split-csv",
                str(split_csv),
                "--models-toml",
                str(models_toml),
                "--results",
                str(tmp_path / "results.jsonl"),
            ]
        )

        assert exit_code == 0
        assert "selected items: 2 (max_variants_per_object=all)" in capsys.readouterr().out


class TestCmdAnalyze:
    def test_writes_report_scatter_and_decision(self, tmp_path: Path) -> None:
        split_csv = tmp_path / "split.csv"
        split_csv.write_text(
            "object_key,fruit_type,view,image_file,weight_g,split,variant\n"
            "a,apple,top,a.JPG,100.0,dev,1\n"
            "b,apple,top,b.JPG,100.0,holdout,1\n"
            "c,apple,top,c.JPG,200.0,holdout,1\n"
        )
        results = tmp_path / "results.jsonl"
        rows: list[dict[str, Any]] = [
            {
                "key": f"vendor/a|S1|hash{i}|0",
                "model": "vendor/a",
                "provider": "p",
                "strategy": "S1",
                "repeat": 0,
                "object_key": object_key,
                "fruit_type": "apple",
                "view": "top",
                "split": "holdout",
                "true_g": true_g,
                "pred_g": pred_g,
                "outcome": "ok",
                "cost_usd": 0.001,
                "latency_ms": 100.0,
            }
            for i, (object_key, true_g, pred_g) in enumerate(
                [("b", 100.0, 102.0), ("c", 200.0, 198.0)]
            )
        ]
        results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        out_dir = tmp_path / "report"
        exit_code = cli.main(
            [
                "analyze",
                "--results",
                str(results),
                "--split-csv",
                str(split_csv),
                "--out-dir",
                str(out_dir),
            ]
        )

        assert exit_code == 0
        assert (out_dir / "report.md").exists()
        assert (out_dir / "scatter.png").exists()
        assert (out_dir / "decision.md").exists()
