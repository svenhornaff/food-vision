# Pre-study (Phase P) — lean version on ECUSTFD

**Status:** current implementation spec for concept Appendix A. Replaces the
full hybrid CLI + web UI spec (archived, see §9).
**Question:** which vision model should the MVP use, and does it perceive
fruit size at all?
**Principle:** an engineering decision, not a paper. Use a public weighed
dataset, a single script-sized harness on top of M1, and a small own-photo
transfer check.

## 1. Decision this produces

`bench/reports/prestudy-lean/decision.md` names:

- **default model** and **fallback model** (different vendor),
- **strategy**: S1 (estimate grams) or S3 (estimate grams with a stated
  prior),
- or **stop**: no model perceives size; rethink before Phase 0.

## 2. Dataset: ECUSTFD (Hugging Face mirror)

Source: `ai5labsOfficial/ecustfd` (mirror of Liang & Li 2017). 2,978 phone
photos, 19 foods, top and side views, 25 mm coin as scale reference,
`weight_g` and `volume_mm3` per object (`portions.csv`), bounding boxes per
object (`annotations.csv`). Mirror licence CC-BY-4.0; cite the original
paper (arXiv 1705.07632) in anything shared.

**Subset used:** apple, banana, orange, pear, qiwi (kiwi). Images with
exactly one annotated object only.

**Target:** whole-fruit `weight_g`. ECUSTFD has no edible weight; for model
selection whole weight is equivalent (edible = whole × a fixed per-type
ratio downstream).

**Object grouping.** The same physical object appears in several images
(top + side, sometimes repeated). All statistics and the split use the
**object**, not the image. Key: the object id from `portions.csv` if it is
stable across that object's images; otherwise
`(type, weight_g, volume_mm3)`. Verify which holds in L1 before anything
else.

**Split (fixed before the first model call):**
`holdout = int(sha256(object_key).hexdigest(), 16) % 10 < 3` → ~30%
hold-out, ~70% dev. Dev is used for prompt checks and the S3 prior. The
decision uses hold-out only. Prompts are never changed after looking at
hold-out results.

**Known limits** (stated in the report, not fixed):

- public since 2017, so contamination can't be excluded,
- coin instead of card, iPhone 4s/7, Chinese market produce,
- whole weight, not edible weight.

The transfer check (§6) covers the practical impact.

## 3. What is varied

| Factor | Levels |
|---|---|
| Model | 6 candidates: 2 frontier, 2 economical, 2 open-weight (incl. Qwen3-VL). Each with image input + strict JSON schema on a ZDR route; one pinned provider per model. |
| Strategy | S1, S3. S2 is dropped: ECUSTFD has no length/diameter ground truth. |
| View | top, side (run as separate images, analysed together per object) |

Fixed: temperature 0 (provider default where 0 is rejected, recorded),
`max_tokens` 2,048, provider-default reasoning, `json_schema` strict (or
`json_object` for models without it), no output repair, JPEG 1024 px, one
repeat. A stability subset (20 hold-out images × 3 repeats) is run for the
top-2 models only.

## 4. Harness

### 4.1 Changes to M1 code

| File | Change |
|---|---|
| `observation/schema.py` | Rename output field `edible_g` → `mass_g`. The prompt defines which mass is meant. |
| `observation/prompts/` | Add `ecustfd_s1_v1_en.md`, `ecustfd_s3_v1_en.md`: whole fruit including peel, 25 mm coin as scale, top or side view. Existing `fruit_*` prompts stay for the transfer check. |
| `observation/adapter.py` | `ObservationConfig` gains `prompt_set: str = "fruit"` (selects the file prefix) and `max_tokens: int | None`, passed to `provider.complete()`. |
| `config/settings.py` | No `PRESTUDY_*` additions. |
| `pyproject.toml` | Add `huggingface_hub`, `matplotlib`. Nothing else. |

### 4.2 New code: `src/food_vision/prestudy/`

| Module | Responsibility |
|---|---|
| `ecustfd.py` | `snapshot_download` the dataset to the HF cache; read the CSVs with stdlib `csv`; filter types and single-object images; build `Item(image_path, object_key, fruit_type, view, weight_g, split)`. |
| `run.py` | Cross items × models × strategies; call `normalize()` + `observe()`; append one JSON line per call to `bench/runs/prestudy-lean/results.jsonl`. **Resume = skip keys already in the file.** `--dry-run N` runs N calls per model and prints the extrapolated cost. `--budget` stops cleanly when reached. |
| `analysis.py` | numpy only. Reads `results.jsonl`, computes §5 metrics on hold-out, applies the decision rule. |
| `report.py` | Writes `report.md` (tables + decision) and `scatter.png` (predicted vs true, log-log, one panel per model). |
| `__main__.py` | `python -m food_vision.prestudy {prepare,run,analyze}` via `argparse`. No typer, no web UI, no database. |

**`results.jsonl` line** (key = `model|strategy|image_sha256|repeat`):

```json
{"key": "...", "model": "...", "provider": "...", "strategy": "S1", "repeat": 0,
 "object_key": "...", "fruit_type": "apple", "view": "top", "split": "holdout",
 "true_g": 182.0, "pred_g": 170.0, "outcome": "ok",
 "model_resolved": "...", "generation_id": "...", "finish_reason": "stop",
 "prompt_sha256": "...", "image_sent_sha256": "...",
 "prompt_tokens": 1240, "completion_tokens": 160, "reasoning_tokens": 0,
 "cost_usd": 0.0011, "latency_ms": 2140, "git_sha": "...", "ts": "2026-10-10T09:12:03Z"}
```

That line is the whole provenance model: the prompt hash, image hash,
resolved model and git SHA are enough to reproduce or audit any number.

### 4.3 Size

~5 types × ~25–35 hold-out objects × ~2–4 images ≈ 300–400 hold-out images.
6 models × 2 strategies × ~350 images ≈ 4,200 calls, plus ~1,000 on dev.
Expected cost is in the single-to-low-double-digit USD range. The dry run
gives the real figure before the full run.

## 5. Analysis and decision rule

**Unit:** object. Per (model, strategy, object): prediction = median over
that object's images and repeats with outcome `ok`. An object with no `ok`
result gets the B0 prediction (penalises unreliable models).

| Metric | Definition | Gate |
|---|---|---|
| MAPE | mean over objects of \|pred − true\| / true | ranking |
| Gain vs B0 | 1 − MAPE / MAPE(B0); B0 = dev mean weight per type | ≥ 30% |
| Size slope β | OLS of log(pred) on log(true) with fruit-type dummies | 0.7 ≤ β ≤ 1.3 |
| Geometric bias | exp(mean log(pred/true)) − 1 | within ±10% |
| Validity | share of calls with outcome `ok` | ≥ 98% |
| Repeat CV | median within-image CV, stability subset | ≤ 5% (top-2 only) |
| Cost / 1,000 images, latency P50/P95 | from `results.jsonl` | tie-breakers |

95% intervals: bootstrap over objects, 2,000 resamples, fixed seed.

**Rule:**

- Per model, use S3 instead of S1 only if its MAPE is lower **and** β stays
  in range; otherwise S1.
- Among models passing all gates, pick the lowest MAPE. If the
  bootstrapped MAPE difference to the best includes 0, treat them as tied
  and pick the cheaper one.
- Fallback: best passing model from a different vendor.
- No model passes → **stop**.

## 6. Transfer check (own photos, ~30 minutes)

12 own fruits (2–3 each of apple, banana, orange, pear, kiwi, small to
large). For each: one 45° photo with an ID-1 card, `whole_g`, `edible_g`.
Run **only** the chosen default and fallback with the existing `fruit_*`
prompts (card, edible mass). Store the photos and a `transfer.csv` under
`bench/sets/transfer-v1/` (images untracked).

Pass if, for the default model:

- MAPE on own photos ≤ ECUSTFD hold-out MAPE + 10 pp, and
- predictions rise with true weight (β > 0.5 on these 12; n is small, so
  this is a sanity check, not a test).

If it fails, the ECUSTFD result doesn't transfer to your setup. Run the
full own-photo study (archived spec) before Phase 0.

## 7. Milestones

| ID | Content | Done when |
|---|---|---|
| M1 | proxy extension, imaging, observation, calculator | **done** (`f3ee28f`) |
| L1 | §4.1 changes; `ecustfd.py`; `prepare` command | `prepare` prints counts per type/split/view; object grouping verified; split file `bench/runs/prestudy-lean/split.csv` committed |
| L2 | `run.py`, dry run, smoke on 5 dev images per model | every candidate returns `ok` on dev; capability notes (schema mode, temperature) recorded in `bench/runs/prestudy-lean/models.toml` |
| L3 | full hold-out run (+ stability subset) | `results.jsonl` complete; cost within budget |
| L4 | `analysis.py`, `report.py`, transfer check | `report.md`, `scatter.png`, `decision.md` written |

**Tests** (no network, fake provider): split determinism and object
grouping on a tiny CSV fixture; resume skips existing keys; dry-run and
budget stop; metrics against hand-computed fixtures; decision rule on
synthetic results (pass, tie, stop). Keep the 85% coverage floor.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Object ids in `portions.csv` not stable across images | fallback key `(type, weight_g, volume_mm3)`; verified in L1 |
| ECUSTFD seen in training | transfer check on own photos; report states the limit |
| Coin vs card, old phones | transfer check |
| Reasoning models ignore temperature or exhaust tokens | `max_tokens` 2,048; `truncated` outcome counted against validity |
| Provider swaps model mid-run | `model_resolved` per line; analysis flags models with >1 value |
| Cost overrun | dry run, `--budget` hard stop |

## 9. Repository changes

**Remove / archive (docs):**

- `docs/dev/pre-study-implementation.md` → delete (superseded twice).
- Full hybrid spec (previous content of this file) →
  `docs/dev/archive/pre-study-full-spec.md`. Revive it only if the transfer
  check fails or the pre-study becomes a publication.
- Rename this file: `git mv docs/dev/pre-study-web-ui.md docs/dev/pre-study.md`
  and update links in `README.md`, `boilerplate.md` and the concept doc.

**Update (docs):**

- `food-vision-concept.md` Appendix A: keep A.1 (rationale), replace
  A.2–A.7 with a three-line summary pointing to `docs/dev/pre-study.md`
  (ECUSTFD hold-out, S1/S3, decision rule §5, transfer check §6). Phase P
  in §14: "ECUSTFD lean pre-study + own-photo transfer check".
- `boilerplate.md`: checklist items for M2–M6 replaced by L1–L4.

**Do not build (code):** `prestudy/db.py`, SQLite schema, `sweeps`,
`protocols`, `executor` with leases, `fits`, `configs` with content hashing,
`web/` (FastAPI, templates), `charts` with provenance tables, `typer` CLI.
Do not add `fastapi`, `uvicorn`, `python-multipart`, `jinja2`, `typer`.

**Keep (code), already in M1:** everything. `imaging/covariates.py` and
`domain/calculator.py` are unused by the lean path but tested and needed
again in Phase 0 / the full spec. Leave them.

**Next step:** L1. Start with `ecustfd.py` and `prepare`: the object
grouping check decides whether the split is sound, and everything else
builds on it.