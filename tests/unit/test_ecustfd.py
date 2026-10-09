"""prestudy.ecustfd: object grouping verification, split determinism,
variant selection, CSV round-trip. No network (no snapshot_download call)."""

from __future__ import annotations

from pathlib import Path

from food_vision.prestudy.ecustfd import (
    Item,
    load_items,
    read_split_csv,
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

    def test_picks_lowest_variant_per_object_view(self, tmp_path: Path) -> None:
        snapshot_dir = self._write_csvs(tmp_path)

        report = load_items(snapshot_dir, types=frozenset({"apple"}))

        side_items = [
            item for item in report.items if item.object_key == "apple001" and item.view == "side"
        ]
        assert len(side_items) == 1
        assert side_items[0].variant == 1
        assert side_items[0].image_path == Path("apple001S(1).JPG")

    def test_one_item_per_object_view(self, tmp_path: Path) -> None:
        snapshot_dir = self._write_csvs(tmp_path)

        report = load_items(snapshot_dir, types=frozenset({"apple", "banana"}))

        keys = [(item.object_key, item.view) for item in report.items]
        assert len(keys) == len(set(keys))
        assert ("apple001", "side") in keys
        assert ("apple001", "top") in keys
        assert ("apple002", "side") in keys
        assert ("banana001", "side") in keys

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
