"""One-off pre-study analysis script (not part of the package); see
bench/reports/prestudy-lean/review.md §5, "E1": does VLM-predicted
bounding-box geometry get close to the ground-truth-box oracle
(bbox_oracle.py: 8.1% MAPE) once real localisation error is included?

Reuses bbox_oracle.py's exact geometry/metrics/bootstrap methodology
(same formulas, same per-type dev-fitted k constants from *ground-truth*
boxes — so the only new source of error, versus the oracle, is the VLM's
own box predictions, not a second fitting step) applied to VLM-predicted
(coin, fruit) boxes from results.jsonl's "BBOX" strategy rows instead of
annotation-XML boxes.

Usage:
    python bench/scripts/bbox_vlm_eval.py \
        <ECUSTFD Annotations dir> <ECUSTFD images dir> \
        bench/runs/prestudy-lean/split.csv \
        bench/runs/prestudy-lean/results.jsonl \
        bench/reports/prestudy-lean/bbox_vlm_eval.json
"""

# ruff: noqa: N806, N816, E741, E402, B905, B007, E501
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image

from food_vision.imaging.preprocess import ImageConfig, normalize

# Must match run.RunConfig's default exactly: used to re-derive the
# sent image's dimensions and to sha256-verify it against results.jsonl
# (image_sent_sha256) — any mismatch means the eval's assumptions about
# what was actually sent have drifted from the real run.
_IMAGE_CONFIG = ImageConfig(max_decoded_pixels=40_000_000, long_edge_px=1024)

ann_dir = Path(sys.argv[1])
images_dir = Path(sys.argv[2])
split_csv = Path(sys.argv[3])
results_jsonl = Path(sys.argv[4])
out_json = Path(sys.argv[5])


def _read_split_csv(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, int], str]]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(f"could not read split csv {path}: {exc}") from exc
    objs: dict[str, dict[str, Any]] = {}
    image_file_by_slot: dict[tuple[str, str, int], str] = {}
    for r in csv.DictReader(io.StringIO(text)):
        objs[r["object_key"]] = {
            "type": r["fruit_type"],
            "w": float(r["weight_g"]),
            "split": r["split"],
        }
        image_file_by_slot[(r["object_key"], r["view"], int(r["variant"]))] = r["image_file"]
    return objs, image_file_by_slot


objs, image_file_by_slot = _read_split_csv(split_csv)


def _elem_text(elem: ET.Element | None, tag: str) -> str:
    """``elem.find(tag).text``, but explicit about the two ways
    ElementTree says "missing": the child tag absent (``find`` returns
    ``None``) vs. present-but-empty (``.text`` is ``None``). Annotation
    XML this malformed would mean the dataset itself is broken, so this
    raises rather than silently treating it as 0/""."""
    if elem is None:
        raise ValueError(f"missing <{tag}> element in annotation XML")
    child = elem.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"<{tag}> has no text in annotation XML")
    return child.text


# ---- dev-fitted k, from GROUND TRUTH boxes (identical to bbox_oracle.py) ----
pat = re.compile(r"^([a-z]+\d+)([ST])\((\d+)\)\.xml$")
gt: dict[str, dict[str, list[tuple[float, float]]]] = collections.defaultdict(
    lambda: {"S": [], "T": []}
)
for f in sorted(ann_dir.iterdir()):
    m = pat.match(f.name)
    if not m or m.group(1) not in objs:
        continue
    try:
        root = ET.parse(f).getroot()
    except ET.ParseError as exc:
        raise SystemExit(f"malformed annotation XML {f}: {exc}") from exc
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
    gt[m.group(1)][m.group(2)].append((fw, fh))


def gt_proxy_both(key: str) -> float | None:
    T, S = gt[key]["T"], gt[key]["S"]
    if not T or not S:
        return None
    a = float(st.median([max(x) for x in T]))
    b = float(st.median([min(x) for x in T]))
    h = float(st.median([y for _, y in S]))
    return a * b * h


kt: dict[str, list[float]] = collections.defaultdict(list)
for k, o in objs.items():
    if o["split"] == "dev":
        p = gt_proxy_both(k)
        if p:
            kt[o["type"]].append(math.log(o["w"] / p))
kfit = {t: math.exp(st.mean(v)) for t, v in kt.items()}

# ---- VLM-predicted boxes, from results.jsonl's BBOX rows ----
_image_dims_cache: dict[str, tuple[int, int]] = {}


def _normalized_bytes(image_file: str) -> bytes:
    try:
        raw = (images_dir / image_file).read_bytes()
    except OSError as exc:
        raise SystemExit(f"could not read raw image {image_file}: {exc}") from exc
    return normalize(raw, _IMAGE_CONFIG)


def image_dims(image_file: str) -> tuple[int, int]:
    """Re-run the exact same normalize() the paid call used (same
    ImageConfig defaults run.py uses) to get the width/height the model
    actually saw — results.jsonl stores only a sha256 of the sent bytes,
    not the bytes or their dimensions, so this is reconstructed, not
    stored; sha256-verified below so a silent config drift would be
    caught, not silently mismeasured."""
    if image_file not in _image_dims_cache:
        with Image.open(io.BytesIO(_normalized_bytes(image_file))) as im:
            _image_dims_cache[image_file] = im.size
    return _image_dims_cache[image_file]


def verify_sent_image(image_file: str, expected_sha256: str) -> bool:
    return hashlib.sha256(_normalized_bytes(image_file)).hexdigest() == expected_sha256


def box_wh_mm(parsed: dict[str, float], image_w: int, image_h: int) -> tuple[float, float] | None:
    """(fruit_w_mm, fruit_h_mm) from normalised-fraction boxes, scaled by
    the VLM's *own* predicted coin box (not the ground-truth one — a real
    pipeline has no access to that either). ``None`` if the predicted
    coin box collapses to ~0 px (degenerate scale)."""
    coin_w_px = (parsed["coin_xmax"] - parsed["coin_xmin"]) * image_w
    coin_h_px = (parsed["coin_ymax"] - parsed["coin_ymin"]) * image_h
    if coin_w_px + coin_h_px <= 0:
        return None
    s = 25.0 / ((coin_w_px + coin_h_px) / 2)  # mm per px, same averaging as the oracle
    fruit_w_px = (parsed["fruit_xmax"] - parsed["fruit_xmin"]) * image_w
    fruit_h_px = (parsed["fruit_ymax"] - parsed["fruit_ymin"]) * image_h
    return fruit_w_px * s, fruit_h_px * s


rows_by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
skipped_sha_mismatch = 0
try:
    lines = results_jsonl.read_text().splitlines()
except OSError as exc:
    raise SystemExit(f"could not read results jsonl {results_jsonl}: {exc}") from exc
for line in lines:
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed results.jsonl line: {exc}") from exc
    if row.get("strategy") != "BBOX" or row.get("split") != "holdout":
        continue
    if row.get("outcome") != "ok" or row.get("parsed") is None:
        continue
    image_file = image_file_by_slot.get((row["object_key"], row["view"], row["variant"]))
    if image_file is None:
        continue
    if not verify_sent_image(image_file, row["image_sent_sha256"]):
        skipped_sha_mismatch += 1
        continue
    rows_by_model[row["model"]].append({**row, "image_file": image_file})


def vlm_proxy_both(rows: list[dict[str, Any]], object_key: str) -> float | None:
    """Same top+side combination as bbox_oracle.py's evaluate('both'),
    applied to this model's VLM-predicted (fruit_w_mm, fruit_h_mm) per
    view, medianed across repeats/variants within a view first."""
    top_wh: list[tuple[float, float]] = []
    side_wh: list[tuple[float, float]] = []
    for row in rows:
        if row["object_key"] != object_key:
            continue
        w, h = image_dims(row["image_file"])
        wh_mm = box_wh_mm(row["parsed"], w, h)
        if wh_mm is None:
            continue
        (top_wh if row["view"] == "top" else side_wh).append(wh_mm)
    if not top_wh or not side_wh:
        return None
    a = float(st.median([max(x) for x in top_wh]))
    b = float(st.median([min(x) for x in top_wh]))
    h2 = float(st.median([y for _, y in side_wh]))
    return a * b * h2


def metrics(preds: list[tuple[str, float, float]]) -> dict[str, Any]:
    ape = [abs(p - t) / t for _, t, p in preds]
    lb = st.mean([math.log(p / t) for _, t, p in preds])
    by: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for ty, t, p in preds:
        by[ty].append((math.log(t), math.log(p)))
    sxx = sxy = 0.0
    for v in by.values():
        mx = st.mean(x for x, _ in v)
        my = st.mean(y for _, y in v)
        sxx += sum((x - mx) ** 2 for x, _ in v)
        sxy += sum((x - mx) * (y - my) for x, y in v)
    return dict(
        n=len(preds),
        mape=st.mean(ape),
        medape=st.median(ape),
        bias=math.exp(lb) - 1,
        beta=sxy / sxx if sxx else float("nan"),
    )


def boot(
    preds: list[tuple[str, float, float]],
    f: Any,
    n: int = 2000,
    seed: int = 7,
) -> tuple[float, float]:
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rnd.choice(preds) for _ in preds]
        try:
            vals.append(f(s))
        except ZeroDivisionError:
            pass
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


out: dict[str, Any] = {
    "skipped_sha_mismatch": skipped_sha_mismatch,
    "kfit_from_ground_truth_dev": kfit,
}
rows_out: dict[str, dict[str, Any]] = {}
for model, model_rows in sorted(rows_by_model.items()):
    object_keys = sorted({row["object_key"] for row in model_rows})
    preds = []
    for key in object_keys:
        proxy = vlm_proxy_both(model_rows, key)
        if proxy is None or objs[key]["type"] not in kfit:
            continue
        pred_g = kfit[objs[key]["type"]] * proxy
        preds.append((objs[key]["type"], objs[key]["w"], pred_g))
    if not preds:
        rows_out[model] = {"n": 0, "note": "no object had both a top and a side ok BBOX attempt"}
        continue
    model_metrics = metrics(preds)
    model_metrics["mape_ci"] = boot(preds, lambda s: metrics(s)["mape"])
    model_metrics["beta_ci"] = boot(preds, lambda s: metrics(s)["beta"])
    rows_out[model] = model_metrics

out["rows"] = rows_out
try:
    out_json.write_text(json.dumps(out, indent=1))
except OSError as exc:
    raise SystemExit(f"could not write {out_json}: {exc}") from exc

for model, model_metrics in rows_out.items():
    if model_metrics.get("n", 0) == 0:
        print(f"{model:35s} {model_metrics.get('note', 'n=0')}")
        continue
    print(
        f"{model:35s} n={model_metrics['n']:2d} MAPE={model_metrics['mape']:.3f} "
        f"[{model_metrics['mape_ci'][0]:.3f},{model_metrics['mape_ci'][1]:.3f}] "
        f"MedAPE={model_metrics['medape']:.3f} "
        f"bias={model_metrics['bias']:+.3f} beta={model_metrics['beta']:.2f} "
        f"[{model_metrics['beta_ci'][0]:.2f},{model_metrics['beta_ci'][1]:.2f}]"
    )
print("skipped (sha256 mismatch, config drift)", skipped_sha_mismatch)
