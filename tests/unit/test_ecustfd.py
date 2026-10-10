"""prestudy.ecustfd: object grouping verification, split determinism,
variant selection, CSV round-trip. No network (no snapshot_download call)."""

from __future__ import annotations

from pathlib import Path

import pytest

from food_vision.prestudy.ecustfd import (
    Item,
    load_items,
    read_split_csv,
    select_variants,
    verify_object_grouping,
    write_split_csv,
)

_PORTIONS = [
    {"sheet": "apple", "id": "apple001", "type": "apple", "volume_mm3": "310", "weight_g": "244.5"},
    {"sheet": "apple", "id": "apple002", "type": "apple", "volume_mm3": "290", "weight_g": "232.5"},
    {
        "sheet": "banana",
        "id": "banana001",
        "type": "banana",
        "volume_mm3": "150",
        "weight_g": "120.0",
    },
    {"sheet": "egg", "id": "egg001", "type": "egg", "volume_mm3": "40", "weight_g": "55.0"},
]


def _annotation_row(file: str, object_id: str, name: str = "apple") -> dict[str, str]:
    return {
        "file": file,
        "object_id": object_id,
        "name": name,
        "xmin": "0",
        "ymin": "0",
        "xmax": "10",
        "ymax": "10",
    }


_CLEAN_ANNOTATIONS = [
    _annotation_row("apple001S(1).JPG", "apple001"),
    _annotation_row("apple001S(1).JPG", "apple001", name="coin"),
    _annotation_row("apple001S(2).JPG", "apple001"),
    _annotation_row("apple001T(1).JPG", "apple001"),
    _annotation_row("apple002S(1).JPG", "apple002"),
    _annotation_row("apple002T(1).JPG", "apple002"),
    _annotation_row("banana001S(1).JPG", "banana001", name="banana"),
    _annotation_row("banana001T(1).JPG", "banana001", name="banana"),
]


class TestVerifyObjectGrouping:
    def test_stable_grouping_uses_portions_id(self) -> None:
        result = verify_object_grouping(_PORTIONS, _CLEAN_ANNOTATIONS)

        assert result.key_used == "portions_id"
        assert result.fallback_triggered is False
        assert result.inconsistent_object_ids == ()
        assert result.total_objects == len(_PORTIONS)

    def test_object_id_missing_from_portions_triggers_fallback(self) -> None:
        annotations = [*_CLEAN_ANNOTATIONS, _annotation_row("ghost001S(1).JPG", "ghost001")]

        result = verify_object_grouping(_PORTIONS, annotations)

        assert result.key_used == "type_weight_volume"
        assert result.fallback_triggered is True
        assert "ghost001" in result.inconsistent_object_ids

    def test_filename_not_prefixed_by_object_id_triggers_fallback(self) -> None:
        annotations = [*_CLEAN_ANNOTATIONS, _annotation_row("mislabeled.JPG", "apple001")]

        result = verify_object_grouping(_PORTIONS, annotations)

        assert result.fallback_triggered is True
        assert "apple001" in result.inconsistent_object_ids

    def test_coin_rows_never_affect_grouping(self) -> None:
        """A coin row with a bogus object_id must not trigger the fallback —
        only non-coin (food) rows are checked."""
        annotations = [*_CLEAN_ANNOTATIONS, _annotation_row("x.JPG", "no-such-id", name="coin")]

        result = verify_object_grouping(_PORTIONS, annotations)

        assert result.fallback_triggered is False


class TestLoadItems:
    def _write_csvs(self, tmp_path: Path) -> Path:
        import csv

        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        with (snapshot_dir / "portions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sheet", "id", "type", "volume_mm3", "weight_g"]
            )
            writer.writeheader()
            writer.writerows(_PORTIONS)
        with (snapshot_dir / "annotations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file", "object_id", "name", "xmin", "ymin", "xmax", "ymax"]
            )
            writer.writeheader()
            writer.writerows(_CLEAN_ANNOTATIONS)
        return snapshot_dir

    def test_filters_to_requested_types(self, tmp_path: Path) -> None:
        snapshot_dir = self._write_csvs(tmp_path)

        report = load_items(snapshot_dir, types=frozenset({"apple"}))

        assert all(item.fruit_type == "apple" for item in report.items)
        assert {item.object_key for item in report.items} == {"apple001", "apple002"}

    def test_keeps_all_variants_per_object_view(self, tmp_path: Path) -> None:
        """apple001/side has variants (1) and (2) in the fixture data;
        both must survive load_items (E3: 'use all variants')— narrowing
        to one is select_variants' job, not load_items'."""
        snapshot_dir = self._write_csvs(tmp_path)

        report = load_items(snapshot_dir, types=frozenset({"apple"}))

        side_items = [
            item for item in report.items if item.object_key == "apple001" and item.view == "side"
        ]
        assert {item.variant for item in side_items} == {1, 2}
        assert {item.image_path for item in side_items} == {
            Path("apple001S(1).JPG"),
            Path("apple001S(2).JPG"),
        }

    def test_every_object_view_slot_is_present(self, tmp_path: Path) -> None:
        snapshot_dir = self._write_csvs(tmp_path)

        report = load_items(snapshot_dir, types=frozenset({"apple", "banana"}))

        # (object_key, view, variant) triples are unique; (object_key, view)
        # slots need not be (apple001/side has 2 variants).
        triples = [(item.object_key, item.view, item.variant) for item in report.items]
        assert len(triples) == len(set(triples))
        slots = {(item.object_key, item.view) for item in report.items}
        assert ("apple001", "side") in slots
        assert ("apple001", "top") in slots
        assert ("apple002", "side") in slots
        assert ("banana001", "side") in slots

    def test_split_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        snapshot_dir = self._write_csvs(tmp_path)

        first = load_items(snapshot_dir, types=frozenset({"apple", "banana"}))
        second = load_items(snapshot_dir, types=frozenset({"apple", "banana"}))

        assert {item.object_key: item.split for item in first.items} == {
            item.object_key: item.split for item in second.items
        }
        assert all(item.split in ("dev", "holdout") for item in first.items)

    def test_excludes_multi_object_images(self, tmp_path: Path) -> None:
        annotations = [
            *_CLEAN_ANNOTATIONS,
            _annotation_row("weird001S(1).JPG", "apple001"),
            {**_annotation_row("weird001S(1).JPG", "apple002"), "name": "apple"},
        ]
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        import csv

        with (snapshot_dir / "portions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sheet", "id", "type", "volume_mm3", "weight_g"]
            )
            writer.writeheader()
            writer.writerows(_PORTIONS)
        with (snapshot_dir / "annotations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file", "object_id", "name", "xmin", "ymin", "xmax", "ymax"]
            )
            writer.writeheader()
            writer.writerows(annotations)

        report = load_items(snapshot_dir, types=frozenset({"apple"}))

        assert report.excluded_multi_object_images == 1
        assert not any(item.image_path.name == "weird001S(1).JPG" for item in report.items)

    def test_unparsed_filename_is_reported_not_crashed_on(self, tmp_path: Path) -> None:
        annotations = [*_CLEAN_ANNOTATIONS, _annotation_row("apple001-weird-name.JPG", "apple001")]
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        import csv

        with (snapshot_dir / "portions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sheet", "id", "type", "volume_mm3", "weight_g"]
            )
            writer.writeheader()
            writer.writerows(_PORTIONS)
        with (snapshot_dir / "annotations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file", "object_id", "name", "xmin", "ymin", "xmax", "ymax"]
            )
            writer.writeheader()
            writer.writerows(annotations)

        report = load_items(snapshot_dir, types=frozenset({"apple"}))

        assert "apple001-weird-name.JPG" in report.unparsed_filenames


class TestSelectVariants:
    def _items(self) -> tuple[Item, ...]:
        return (
            Item(Path("aS1.JPG"), "a", "apple", "side", 100.0, "holdout", 1),
            Item(Path("aS2.JPG"), "a", "apple", "side", 100.0, "holdout", 2),
            Item(Path("aS3.JPG"), "a", "apple", "side", 100.0, "holdout", 3),
            Item(Path("aT1.JPG"), "a", "apple", "top", 100.0, "holdout", 1),
        )

    def test_default_keeps_only_lowest_variant_per_slot(self) -> None:
        """max_variants_per_object=1 (the CLI default) must reproduce the
        prior load_items' 'lowest variant only' behaviour exactly — same
        images, so already-paid results.jsonl rows stay resumable."""
        selected = select_variants(self._items(), max_variants_per_object=1)

        assert {(item.object_key, item.view, item.variant) for item in selected} == {
            ("a", "side", 1),
            ("a", "top", 1),
        }

    def test_max_two_keeps_two_lowest_per_slot(self) -> None:
        selected = select_variants(self._items(), max_variants_per_object=2)

        side_variants = {item.variant for item in selected if item.view == "side"}
        assert side_variants == {1, 2}

    def test_none_keeps_everything(self) -> None:
        items = self._items()
        selected = select_variants(items, max_variants_per_object=None)

        assert selected == items

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            select_variants(self._items(), max_variants_per_object=0)


class TestSplitCsvRoundTrip:
    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        items = (
            Item(
                image_path=Path("apple001S(1).JPG"),
                object_key="apple001",
                fruit_type="apple",
                view="side",
                weight_g=244.5,
                split="dev",
                variant=1,
            ),
            Item(
                image_path=Path("apple001T(1).JPG"),
                object_key="apple001",
                fruit_type="apple",
                view="top",
                weight_g=244.5,
                split="dev",
                variant=1,
            ),
        )
        path = tmp_path / "split.csv"

        write_split_csv(items, path)
        reloaded = read_split_csv(path)

        assert reloaded == items

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "split.csv"

        write_split_csv((), path)

        assert path.exists()
