"""One-off, one-time migration (not part of the package): the 100 real
"BBOX" rows run before ``run.ResultRecord`` gained ``image_width_px``/
``image_height_px`` predate that field. Backfills them in place so the
formal pipeline (``analysis.bbox_predictions_per_object``, review.md §5
"E1" integration) can use the data that's already been paid for, instead
of only the one-off ``bbox_vlm_eval.py`` script being able to.

Re-derives dimensions the same way ``bbox_vlm_eval.py`` already does:
re-run ``normalize()`` with the exact ``ImageConfig`` ``run.py`` uses by
default, sha256-verify the result against the row's own
``image_sent_sha256`` before trusting it (refuses to backfill a row it
can't verify -- never silently guesses).

Usage:
    python bench/scripts/backfill_image_dims.py \
        bench/runs/prestudy-lean/split.csv \
        bench/runs/prestudy-lean/results.jsonl \
        <HF snapshot images dir>
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image  # noqa: E402

from food_vision.imaging.preprocess import ImageConfig, normalize  # noqa: E402

# Must match run.RunConfig's default exactly (same constant bbox_vlm_eval.py uses).
_IMAGE_CONFIG = ImageConfig(max_decoded_pixels=40_000_000, long_edge_px=1024)


def main() -> None:
    split_csv, results_jsonl, images_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    image_file_by_slot: dict[tuple[str, str, int], str] = {}
    with split_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_file_by_slot[(row["object_key"], row["view"], int(row["variant"]))] = row[
                "image_file"
            ]

    lines = results_jsonl.read_text().splitlines()
    backfilled = 0
    skipped_sha_mismatch = 0
    skipped_no_image = 0
    out_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"malformed results.jsonl line: {exc}") from exc
        if row.get("image_width_px") is not None or row.get("strategy") != "BBOX":
            out_lines.append(line)
            continue

        image_file = image_file_by_slot.get(
            (row["object_key"], row["view"], row.get("variant", 1))
        )
        if image_file is None:
            skipped_no_image += 1
            out_lines.append(line)
            continue

        try:
            raw_bytes = (images_dir / image_file).read_bytes()
        except OSError:
            skipped_no_image += 1
            out_lines.append(line)
            continue

        normalized = normalize(raw_bytes, _IMAGE_CONFIG)
        if hashlib.sha256(normalized).hexdigest() != row["image_sent_sha256"]:
            skipped_sha_mismatch += 1
            out_lines.append(line)
            continue

        with Image.open(io.BytesIO(normalized)) as decoded:
            row["image_width_px"], row["image_height_px"] = decoded.size
        out_lines.append(json.dumps(row, sort_keys=False))
        backfilled += 1

    results_jsonl.write_text("\n".join(out_lines) + "\n")
    print(f"backfilled={backfilled} skipped_sha_mismatch={skipped_sha_mismatch} "
          f"skipped_no_image={skipped_no_image}")


if __name__ == "__main__":
    main()
