"""ECUSTFD dataset loading, object grouping, and the dev/hold-out split.

docs/dev/pre-study.md §2/§4.2 (L1). Downloads
``ai5labsOfficial/ecustfd`` (a public mirror of Liang & Li 2017) via
``huggingface_hub``, reads its CSVs with stdlib ``csv``, filters to the
fruit types this pre-study uses, and builds one :class:`Item` per
(object, view).

**Verified against the real dataset (2026-10-10, snapshot
``2000fa65170163e91e7043afbc25b708eaff2236``):** the ``portions.csv``
``id`` column (e.g. ``"apple001"``) is a 100%-stable join key — every
``annotations.csv`` row's ``object_id`` exists in ``portions.csv``, every
image filename starts with its own ``object_id``, and no image has more
than one non-coin annotated object. :func:`verify_object_grouping` still
checks this on every run rather than assuming it, per §2: "Verify which
holds in L1 before anything else." The ``(type, weight_g, volume_mm3)``
fallback (§2) is implemented and exercised by tests, but has not been
needed against the real data.

**Variants (2026-10-10, revised per external review):** each (object,
view) may have 2-34 repeated photos in the raw dataset ("S" and "T"
each have several numbered variants, e.g. ``apple001S(1).JPG`` through
``apple001S(8).JPG``). :func:`load_items` now keeps *every* variant as
its own :class:`Item` — ``bench/runs/prestudy-lean/split.csv`` is the
full dataset manifest, not a pre-selected sweep plan. A *prior* version
of this module kept only the lowest-numbered variant per (object, view)
(144 of 1,014 fruit images); an external pre-study review
(``bench/reports/prestudy-lean/review.md`` §2/§5, "E3") flagged this as
under-using the dataset and recommended using all variants, median per
object. :func:`select_variants` is the dataset-enumeration/experiment-
selection split the review asked for: callers that want the old
one-variant-per-slot behaviour (e.g. to keep reusing already-paid
``results.jsonl`` rows unchanged) pass ``max_variants_per_object=1``
(the CLI's default); a full re-analysis can pass ``None`` for all of
them.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from food_vision.utils.log_factory import get_logger

logger = get_logger(__name__)

__all__ = [
    "ECUSTFD_REPO_ID",
    "DEFAULT_TYPES",
    "Item",
    "GroupingResult",
    "PrepareReport",
    "download",
    "verify_object_grouping",
    "load_items",
    "select_variants",
    "write_split_csv",
    "read_split_csv",
    "prepare",
]

#: Hugging Face dataset repo (pre-study.md §2).
ECUSTFD_REPO_ID = "ai5labsOfficial/ecustfd"

#: Fruit subset used by the lean pre-study (pre-study.md §2). ECUSTFD's
#: "kiwi" type is spelled "qiwi" in its own CSVs — kept verbatim so it
#: round-trips through portions.csv without a silent rename.
DEFAULT_TYPES: frozenset[str] = frozenset({"apple", "banana", "orange", "pear", "qiwi"})

#: Filenames are ``{object_id}{S|T}({variant}).JPG``, e.g. ``apple001S(1).JPG``.
_FILENAME_SUFFIX_RE = re.compile(r"^([ST])\((\d+)\)\.JPG$", re.IGNORECASE)

_VIEW_BY_LETTER = {"S": "side", "T": "top"}


@dataclass(frozen=True)
class Item:
    """One (object, view) the harness will send to a model.

    ``image_path`` is a path *relative to the dataset snapshot's*
    ``images/`` *directory* (just the filename) — ``split.csv`` is
    committed to the repo and must not bake in a machine-specific HF
    cache path. Resolve with ``snapshot_dir / "images" / item.image_path``.
    """

    image_path: Path
    object_key: str
    fruit_type: str
    view: str  # "top" | "side"
    weight_g: float
    split: str  # "dev" | "holdout"
    #: The variant number actually picked (lowest available); kept for
    #: traceability, not part of the spec's minimal Item signature.
    variant: int


@dataclass(frozen=True)
class GroupingResult:
    """Which object-grouping key held, and what :func:`verify_object_grouping`
    found (pre-study.md §2's required verification)."""

    key_used: str  # "portions_id" | "type_weight_volume"
    fallback_triggered: bool
    total_objects: int
    #: ``object_id``s where the primary key failed to hold (filename
    #: didn't match, or the id was missing from ``portions.csv``). Empty
    #: when ``fallback_triggered`` is ``False``.
    inconsistent_object_ids: tuple[str, ...]


@dataclass(frozen=True)
class PrepareReport:
    """Everything ``prepare`` needs to print and write."""

    items: tuple[Item, ...]
    grouping: GroupingResult
    #: Images excluded for having zero or >1 distinct non-coin annotated
    #: objects (defensive — verified 0 on the real dataset, but not
    #: assumed).
    excluded_multi_object_images: int
    #: Filenames that didn't match the expected ``{id}{S|T}(n).JPG`` shape.
    unparsed_filenames: tuple[str, ...]


def download(*, cache_dir: Path | None = None) -> Path:
    """Download (or reuse the cached) ECUSTFD snapshot.

    Uses ``huggingface_hub``'s own cache (``~/.cache/huggingface`` by
    default) — deliberately *not* a path under the repo, so the ~3GB
    dataset is never at risk of being committed or `git add -A`-ed.

    Returns:
        The local snapshot directory (contains ``images/``,
        ``portions.csv``, ``annotations.csv``, ``food_types.csv``).
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=ECUSTFD_REPO_ID, repo_type="dataset", cache_dir=cache_dir)
    return Path(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_object_grouping(
    portions_rows: list[dict[str, str]], annotations_rows: list[dict[str, str]]
) -> GroupingResult:
    """Check whether ``portions.csv``'s ``id`` is a stable grouping key.

    An object is "stable" under ``id`` when every non-coin annotation row
    for it has an ``object_id`` that (a) exists in ``portions.csv`` and
    (b) is a prefix of its own filename. If *any* object fails this, the
    whole dataset falls back to ``(type, weight_g, volume_mm3)`` for
    every object — a per-object mix of keys would make joins ambiguous.
    """
    portions_by_id = {row["id"]: row for row in portions_rows}
    inconsistent: set[str] = set()

    for row in annotations_rows:
        if row["name"] == "coin":
            continue
        object_id = row["object_id"]
        file_name = row["file"]
        if object_id not in portions_by_id or not file_name.startswith(object_id):
            inconsistent.add(object_id)

    total_objects = len(portions_by_id)
    if inconsistent:
        return GroupingResult(
            key_used="type_weight_volume",
            fallback_triggered=True,
            total_objects=total_objects,
            inconsistent_object_ids=tuple(sorted(inconsistent)),
        )
    return GroupingResult(
        key_used="portions_id",
        fallback_triggered=False,
        total_objects=total_objects,
        inconsistent_object_ids=(),
    )


def _object_key(row: dict[str, str], grouping: GroupingResult) -> str:
    if grouping.key_used == "portions_id":
        return row["id"]
    return f"{row['type']}_{row['weight_g']}_{row['volume_mm3']}"


def _is_holdout(object_key: str) -> bool:
    """pre-study.md §2: ``holdout = int(sha256(object_key), 16) % 10 < 3``."""
    digest = hashlib.sha256(object_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 < 3


def load_items(snapshot_dir: Path, *, types: frozenset[str] = DEFAULT_TYPES) -> PrepareReport:
    """Build the full (object, view, variant) item list from a downloaded
    snapshot — every usable fruit image, not a pre-selected sweep plan
    (see the module docstring's "Variants" note). Use
    :func:`select_variants` to cut this down for a specific paid run.

    Raises:
        FileNotFoundError: ``portions.csv`` or ``annotations.csv`` is
            missing from ``snapshot_dir``.
    """
    portions_rows = _read_csv(snapshot_dir / "portions.csv")
    annotations_rows = _read_csv(snapshot_dir / "annotations.csv")
    portions_by_id = {row["id"]: row for row in portions_rows}

    grouping = verify_object_grouping(portions_rows, annotations_rows)

    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotations_rows:
        by_file[row["file"]].append(row)

    excluded_multi_object = 0
    unparsed: list[str] = []
    items: list[Item] = []

    for file_name, rows in by_file.items():
        non_coin = [row for row in rows if row["name"] != "coin"]
        distinct_object_ids = {row["object_id"] for row in non_coin}
        if len(distinct_object_ids) != 1:
            excluded_multi_object += 1
            continue
        object_id = next(iter(distinct_object_ids))

        portion = portions_by_id.get(object_id)
        if portion is None or portion["type"] not in types:
            continue

        suffix_match = _FILENAME_SUFFIX_RE.match(file_name[len(object_id) :])
        if suffix_match is None:
            unparsed.append(file_name)
            continue
        view_letter, variant_text = suffix_match.groups()
        view = _VIEW_BY_LETTER[view_letter.upper()]
        variant = int(variant_text)
        object_key = _object_key(portion, grouping)

        items.append(
            Item(
                image_path=Path(file_name),
                object_key=object_key,
                fruit_type=portion["type"],
                view=view,
                weight_g=float(portion["weight_g"]),
                split="holdout" if _is_holdout(object_key) else "dev",
                variant=variant,
            )
        )
    items.sort(key=lambda item: (item.object_key, item.view, item.variant))

    return PrepareReport(
        items=tuple(items),
        grouping=grouping,
        excluded_multi_object_images=excluded_multi_object,
        unparsed_filenames=tuple(sorted(unparsed)),
    )


def select_variants(
    items: tuple[Item, ...], *, max_variants_per_object: int | None = 1
) -> tuple[Item, ...]:
    """Cut a full (all-variants) item list down to at most
    ``max_variants_per_object`` variants per (object_key, view) slot —
    the lowest-numbered variants first.

    This is the dataset-enumeration/experiment-selection split the
    external review asked for (§5, "E3"): :func:`load_items` always
    returns everything; *this* function decides what a specific paid
    run actually sends. ``max_variants_per_object=1`` (the CLI default)
    reproduces the prior version's "keep only the lowest variant"
    behaviour exactly — same images, same ``image_sha256`` values, same
    resume keys as any already-paid ``results.jsonl`` rows.

    Args:
        max_variants_per_object: ``None`` keeps every variant.

    Raises:
        ValueError: ``max_variants_per_object`` is not ``None`` and not
            a positive integer.
    """
    if max_variants_per_object is None:
        return items
    if max_variants_per_object < 1:
        raise ValueError(
            f"max_variants_per_object must be a positive integer or None, "
            f"got {max_variants_per_object!r}."
        )

    variants_by_slot: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in items:
        variants_by_slot[(item.object_key, item.view)].append(item.variant)
    allowed_by_slot = {
        slot: set(sorted(variants)[:max_variants_per_object])
        for slot, variants in variants_by_slot.items()
    }
    return tuple(
        item for item in items if item.variant in allowed_by_slot[(item.object_key, item.view)]
    )


_SPLIT_CSV_FIELDS = (
    "object_key",
    "fruit_type",
    "view",
    "image_file",
    "weight_g",
    "split",
    "variant",
)


def write_split_csv(items: tuple[Item, ...], path: Path) -> None:
    """Write the committed, portable split file (pre-study.md §9: kept,
    unlike ``results.jsonl``, which is gitignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SPLIT_CSV_FIELDS)
        for item in items:
            writer.writerow(
                (
                    item.object_key,
                    item.fruit_type,
                    item.view,
                    item.image_path.as_posix(),
                    item.weight_g,
                    item.split,
                    item.variant,
                )
            )


def read_split_csv(path: Path) -> tuple[Item, ...]:
    """Reload a previously written ``split.csv`` without re-downloading
    or re-parsing the ECUSTFD CSVs — used by ``run.py``/``analysis.py``."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        Item(
            image_path=Path(row["image_file"]),
            object_key=row["object_key"],
            fruit_type=row["fruit_type"],
            view=row["view"],
            weight_g=float(row["weight_g"]),
            split=row["split"],
            variant=int(row["variant"]),
        )
        for row in rows
    )


def prepare(
    *, types: frozenset[str] = DEFAULT_TYPES, split_csv_path: Path, cache_dir: Path | None = None
) -> PrepareReport:
    """L1 entry point: download, verify grouping, load items, write
    ``split.csv``. Returns the report for the CLI to print."""
    snapshot_dir = download(cache_dir=cache_dir)
    report = load_items(snapshot_dir, types=types)
    write_split_csv(report.items, split_csv_path)
    return report
