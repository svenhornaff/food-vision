"""One-off pre-study script (not part of the package); see
bench/reports/prestudy-lean/review.md §5 "the size constant k is fitted
on ground-truth dev boxes. That's conservative and fine here."

Fits the per-fruit-type mass constant k (true_g = k * a * b * h, the
same coin-scaled top+side geometry formula as bbox_oracle.py) from
GROUND TRUTH annotation boxes on dev-split objects only, and writes it
as a small committed JSON artifact so analysis.py's BBOX-strategy path
(bbox_predictions_per_object) doesn't need a filesystem dependency on
the raw ECUSTFD Annotations dir at analysis time -- only at the one time
this script is run.

This is the dataset-level, ground-truth-fit k (an upper-bound reference,
not model-specific). Per-model dev-fitted k ("fit k on each model's own
dev boxes, which corrects systematic box bias per model" -- review.md
§5) is a documented follow-up: it needs dev-split BBOX attempts, which
the current hold-out-only sweep doesn't have.

Usage:
    python bench/scripts/fit_k.py \
        <ECUSTFD Annotations dir> \
        bench/runs/prestudy-lean/split.csv \
        bench/runs/prestudy-lean/kfit.json
"""

# ruff: noqa: E402
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


def _elem_text(elem: ET.Element | None, tag: str) -> str:
    if elem is None:
        raise ValueError(f"missing <{tag}> element in annotation XML")
    child = elem.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"<{tag}> has no text in annotation XML")
    return child.text


@dataclass(frozen=True)
class _ObjMeta:
    fruit_type: str
    weight_g: float
    split: str


def main() -> None:
    ann_dir, split_csv, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    objs: dict[str, _ObjMeta] = {}
    with split_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            objs[row["object_key"]] = _ObjMeta(
                fruit_type=row["fruit_type"],
                weight_g=float(row["weight_g"]),
                split=row["split"],
            )

    pat = re.compile(r"^([a-z]+\d+)([ST])\((\d+)\)\.xml$")
    gt: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for f in sorted(ann_dir.iterdir()):
        m = pat.match(f.name)
        if not m or m.group(1) not in objs:
            continue
        root = ET.parse(f).getroot()
        boxes = [
            (
                _elem_text(o, "name"),
                [int(_elem_text(o.find("bndbox"), k)) for k in ("xmin", "ymin", "xmax", "ymax")],
            )
            for o in root.iter("object")
        ]
        coins = [b for n, b in boxes if n == "coin"]
        fruit = [b for n, b in boxes if n != "coin"]
        if len(coins) != 1 or len(fruit) != 1:
            continue
        c = coins[0]
        s = 25.0 / (((c[2] - c[0]) + (c[3] - c[1])) / 2)
        fw, fh = (fruit[0][2] - fruit[0][0]) * s, (fruit[0][3] - fruit[0][1]) * s
        key = m.group(1)
        gt.setdefault(key, {"S": [], "T": []})[m.group(2)].append((fw, fh))

    kt: dict[str, list[float]] = {}
    for key, obj in objs.items():
        if obj.split != "dev":
            continue
        views = gt.get(key)
        if not views or not views["T"] or not views["S"]:
            continue
        a = st.median(max(x) for x in views["T"])
        b = st.median(min(x) for x in views["T"])
        h = st.median(y for _, y in views["S"])
        proxy = a * b * h
        kt.setdefault(obj.fruit_type, []).append(math.log(obj.weight_g / proxy))

    kfit = {fruit_type: math.exp(st.mean(logs)) for fruit_type, logs in kt.items()}
    out_json.write_text(json.dumps(kfit, indent=1, sort_keys=True) + "\n")
    for fruit_type, k in sorted(kfit.items()):
        print(f"{fruit_type:10s} k={k:.4f}  (n_dev_objects={len(kt[fruit_type])})")


if __name__ == "__main__":
    main()
