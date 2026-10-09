# food-vision — Technical Concept

**Version:** 0.2
**Date:** 8 October 2026
**Status:** Proposed architecture / prototype scope

## 1. Executive Summary

Build **`food-vision`** (distribution `food-vision`, import package `food_vision`), a small Python service that turns a meal photo plus cheap context into a **reviewable, database-grounded nutrition estimate**. A multimodal model accessed via **OpenRouter** *observes*: foods, preparation, bounding boxes, density class, portion cues, uncertainty. Deterministic code *calculates*: it matches observations to versioned reference records (BLS 4.0, USDA FDC, Open Food Facts) and computes kcal and macros as ranges.

The first milestone is an **evidence-driven proof of concept**, not a diary app. It answers one question: *after a 10-second review, is photo logging accurate, cheap and fast enough to replace manual logging for me?*

**Positioning.** The state of the art in October 2026 is clear on four points. Recognition is largely solved; portion estimation is not. Frontier MLLMs are mediocre direct portion estimators (24–36 g per-food MAE on Nutrition5k) and a small geometry head on top of them cuts that by a third. Cheap context (meal time, venue type, a known plate, a weight) reduces error more than model choice does. Consumer apps under-count energy and fat by roughly a third because they hide all of this behind a single number. `food-vision` is designed around those four facts.

**Key architectural decisions**

1. **Observe with the model, calculate deterministically.** Nutrient values come only from reference records. Frontier MLLMs abstain or hallucinate on nutrients beyond the big four; they are never a nutrient source.
2. **Context is a first-class input.** Meal slot, venue type, note, barcode, label photo, plate reference, known weight. Each is optional; each overrides a visual guess.
3. **Ranges, not points.** Every portion and total carries a low/mid/high band, asymmetric for large portions.
4. **Bias-aware by design.** Hidden-fat defaults, targeted follow-up questions, signed-error tracking, optional personal calibration.
5. **Observation contract built for a future portion head.** Bbox and density range are in the schema from day one, so a geometry head can be added later without an API change.
6. **Backend-owned credentials, zero-retention routing, pinned providers for benchmarks.**
7. **Human correction is the product.** Corrections are cheap, recalculate without re-inference, and feed evaluation.

## 2. Goals and Non-goals

### Goals

- Accept a photo (optionally a second angle, or an "after" photo for leftovers) from a browser/PWA with optional note, barcode, label photo, plate ID, meal slot, venue type and known weight.
- Recognize multiple foods, preparation methods and likely hidden contributors (fat, sauce, dressing, sugar in drinks).
- Estimate portions as ranges; mark low-confidence items.
- Ask at most 1–3 targeted follow-up questions when the answer materially moves the total.
- Match to personal library → BLS 4.0 → USDA FDC → OFF; preserve provenance.
- Compute kcal, protein, carbs, fat, fibre per item and per meal, with ranges.
- Recalculate deterministically after edits.
- Record model, prompt version, provider, latency, tokens, cost and correction deltas.

### Non-goals for MVP

- Diary, accounts, subscriptions, coaching, recommendations, social.
- Micronutrients from vision; micronutrients only ever from reference records, and not in the MVP UI.
- Medical-grade measurement; diabetes dosing support.
- Direct OpenRouter calls from the client.
- Training or fine-tuning before the baseline is measured.
- LiDAR/depth capture (native only; Phase 4).
- Persisting original photos by default.

## 3. What others do and what the research says

### 3.1 Product landscape (Oct 2026)

| Product | Notable approach | Take-away |
|---|---|---|
| **MacroFactor** | Photo, photo+text, upload; results built from database entries; editable "plate"; label scanner; front + label photo → custom food; "put your fist in the photo" hint | Adopt: photo+text, DB-grounded inspectable items, label-based food creation, reference-object hint |
| **SnapCalorie** | Nutrition5k authors; LiDAR volume on iPhone Pro, visual estimate otherwise; USDA lookup and sum; voice notes | Adopt: identify → measure → look up → sum. Defer: LiDAR. iOS-only is the cost of a hard sensor dependency |
| **Cal AI** | Speed-first, multi-angle capture, large third-party DB | Adopt: optional second angle. Avoid: opaque single number |
| **Foodvisor** | Early EU pioneer; weak on mixed dishes and hidden oil; crowd DB quality issues | Avoid: unverified crowd data as primary nutrient source |
| **Yazio** (DE) | Big German base; photo recognition added 2025/26, early | German-dish coverage is open; BLS 4.0 is the lever |

### 3.2 Research signals (2021–2026) and what each changes in the design

| Finding | Source | Design consequence |
|---|---|---|
| Portion, not recognition, is the bottleneck: direct kcal MAE ~3× that of kcal-per-gram; depth-assisted mass error 18.7% → 13.7% | Nutrition5k, CVPR 2021 | Treat grams as the primary quantity; everything else is a lookup |
| Frontier MLLMs as direct portion estimators on 100 multi-food Nutrition5k dishes: Gemini-3.5-Flash 24.1 g, Gemini-3.1-Pro 25.8 g, GPT-5.5 26.7 g, Claude Fable 5 / Opus 4.8 ~36 g per-food MAE. Claude under-estimates large piles less but errs more on ordinary portions | Liao & Li, Jul 2026 | Compare signed error and size sensitivity, not only MAE; model families differ in *bias shape* |
| A small geometry head (frozen DINOv2 + softmax-ownership volume) fed only the MLLM's per-food **name, bounding box and density range** cuts per-food MAE 33–41% (Nutrition5k 28.1 → 16.7 g; NV-Real 41.0 → 27.4 g). Removing the MLLM context doubles the head's error. Heavy-portion tail stays under-predicted. No code released | Liao & Li, Jul 2026 | Put `bbox` and `density_range` in the observation contract now; asymmetric ranges (wider high bound) for large portions; Phase 4 reimplements the head |
| Cheap context cuts error: converting timestamp → meal type and GPS → venue type lowered calorie MAE by ~76 kcal on average across 8 models (up to 246 kcal), portion MAE by ~53 g; an "expert persona" prompt added ~75 kcal. Gains larger for open-weight models | ACETADA benchmark, Jul 2025 | Add `meal_slot` and `venue_type` as inputs (user-set or derived on device; never send GPS). Persona line in the prompt |
| Known weight dominates: giving Gemini 2.5 Flash the true weight dropped carb MAPE from 39.5% to 20.2% | ODU / ACM BCB 2025 | `known_weight_g` collapses the range to a point and skips portion estimation |
| Two-step decomposition (ingredients, portions, cooking → then nutrition) beats one-shot | CVPRW 2025 | Step two is deterministic code here |
| Class-level volume prior stated in text ("approximate volume is X ml") materially improves a volume regressor; single-view MAPE ~35% vs ~23% stereo | Vinod et al., Apr 2026 | Give the model a per-food **typical-portion prior table** as input; the prior is also the B0 baseline |
| Before/after photo pairs estimate consumed weight at ~14% PMAE vs ~39% for VLMs on separate images; leftovers of amorphous food handled | DietDelta, Apr 2026 | Optional "after" photo; consumed = before − after. Phase 3 |
| Frontier MLLMs are unreliable for micronutrients: they abstain or return implausible values; fine-tuned small VLMs on 1.1 M synthetic triplets match them | NutriMLLM, Jun 2026 | Never take any nutrient from the model; micronutrients only from BLS/FDC records |
| A "visual estimation" prompt (do not read text or numbers in the image) improves meal estimates with capable models; mid-tier models hit the best cost/accuracy; salt is consistently poor (MedAPE 34–64%); packaged foods are easier | Nakagawa & Yamamoto, Nutrients 2026 | Two prompt modes: **meal** (ignore printed numbers) and **label** (read them). Don't promise salt |
| Photo-based consumer apps under-estimate energy by 250–345 kcal and fat by ~30 g per meal; better on carbs than fat/protein | NIH / ASN, Jul 2026 | Fat-aware defaults, signed-error metric, range display |
| CVPR 2026 MetaFood workshop: depth-free radiance-field volume (VolETA++), agentic interactive estimation, egocentric real-time tracking, cooking-video dish estimation | CVPRW 2026 | Interactive follow-ups are now mainstream research; depth-free volume is a Phase 4 candidate alongside monocular metric depth (Depth Anything v3 metric, Depth Pro) |

### 3.3 Where `food-vision` does better than consumer apps

- **Show the range.** "520–780 kcal" is honest and points the user at the item worth correcting.
- **Fat-aware defaults.** Editable "cooking fat" line for fried/sautéed/roasted items. Deleting a wrong default is one tap; noticing a missing one is impossible.
- **Ask only when it matters.** Follow-up questions only above a kcal-impact threshold; max three.
- **Cheap context.** Meal slot, venue type, plate reference. Proven gains, no sensor.
- **Personal library first.** Own recipes and recent meals before generic DBs.
- **Measure bias, then correct it.** Signed error per category from corrections; visible, reversible calibration factor.
- **Provenance for every number.** Each value links to its source record and, for BLS, its data-origin category.
- **Leftovers.** Optional after-photo; most apps assume the plate was finished.

## 4. System Context and Architecture

```text
 ┌──────────────────────────────┐
 │ PWA / browser (thin)         │ photo(s) · note · barcode · label · weight
 │                              │ plate · meal slot · venue type · answers
 └──────────────┬───────────────┘
                │ HTTPS
 ┌──────────────▼──────────────────────────────────────────────────────────┐
 │ FastAPI  /api/v1                                                        │
 │                                                                         │
 │  preprocess ─► observe (vision adapter) ─► validate (Pydantic)          │
 │  EXIF strip,    OpenRouter, ZDR, pinned      strict schema,             │
 │  resize, hash   json_schema strict           one repair retry           │
 │                         │                                               │
 │                         ▼                                               │
 │  enrich ─► match ───────────────► calculate ─► review state             │
 │  priors,    personal → BLS 4.0     ranges,      pending items,          │
 │  fat rules, → FDC → OFF (local)    totals       follow-up questions     │
 │  questions  provenance kept                                             │
 │                                                                         │
 │  answers / PATCH ─► recalculate (no re-inference)                       │
 └───────┬────────────────────────────┬────────────────────────────────────┘
         │                            │
   ┌─────▼──────┐              ┌──────▼──────────────────────────────┐
   │ telemetry  │              │ reference store (SQLite/DuckDB)     │
   │ (no images)│              │ BLS 4.0 · FDC · OFF parquet ·       │
   └────────────┘              │ density + portion priors ·          │
                               │ personal foods / plates / meals     │
                               └─────────────────────────────────────┘
```

**Deployment boundary.** The client captures and corrects. Inference orchestration, matching, calculation, credentials and storage are server-side. Native apps later reuse the same versioned API.

## 5. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Runtime | Python 3.13 | 3.14 acceptable |
| Packaging | `uv`, `pyproject.toml`, `src` layout, `hatchling` | Lockfile committed |
| Lint / format / types | `ruff` (pycodestyle, pyflakes, isort, pep8-naming, bugbear), `mypy --strict` | PEP 8 enforced in CI, not by convention |
| API | FastAPI + Uvicorn | Multipart uploads, OpenAPI |
| Validation | Pydantic v2 | One schema source → JSON Schema for the model and the API |
| AI gateway | OpenRouter via `openai` client with `base_url` override | Thin adapter; no agent framework |
| Images | Pillow + `pillow-heif` | Orient, resize, re-encode, strip metadata |
| Reference data | Local SQLite/DuckDB built from BLS 4.0 xlsx, FDC bulk CSV, OFF parquet, density/prior tables | No live API on the hot path; reproducible |
| Fuzzy matching | `rapidfuzz` + optional embedding index | Embeddings for candidate retrieval only |
| HTTP | `httpx` | Barcode misses only |
| CLI | `typer` via `[project.scripts]` | `food-vision analyze`, `food-vision bench`, `food-vision build-reference` |
| Tests | pytest, pytest-asyncio, respx, hypothesis | Property tests for the calculator |
| Persistence (later) | PostgreSQL + SQLAlchemy + Alembic | Not needed for stateless PoC |
| Client | Responsive web / PWA, `<input type="file" capture>` | Barcode via `BarcodeDetector`, ZXing fallback |

### Data licensing (verified Oct 2026)

| Source | Terms | Implication |
|---|---|---|
| **BLS 4.0** (Max Rubner-Institut) | Free, no licence fee; app/software development explicitly permitted; cite DOI 10.25826/Data20251217-134202-0 | Primary generic source; ship attribution |
| **USDA FoodData Central** | US public domain (verify current statement) | Secondary generic source |
| **Open Food Facts** | ODbL (attribution + share-alike), contents DbCL, images CC BY-SA; live API ~15 product reads/min/IP, custom User-Agent | Bulk parquet locally; keep OFF-derived data in a separate table to contain share-alike |
| **FAO/INFOODS density database** | FAO open-knowledge publication; check reuse terms | Seed for the density table |
| **Model terms** | Per model/provider | OpenRouter access doesn't change a model's licence |

## 6. End-to-end Processing Flow

1. **Capture** — `POST /api/v1/meals/analyze` with `image` and optional `image_2`, `image_after`, `note`, `barcode`, `known_weight_g`, `plate_id`, `meal_slot`, `venue_type`, `locale`.
2. **Validate** — MIME sniffing, decoded format, size and dimension limits, rate limit.
3. **Normalize** — orientation, resize long edge (start 1024 px; benchmark 768/1280), re-encode, strip EXIF/GPS, perceptual hash for duplicate detection.
4. **Short-circuit** — barcode + known weight resolves without vision.
5. **Observe** — image(s), note, plate dimensions, meal slot, venue type, the typical-portion prior table for likely foods, and the observation schema go to the model with `response_format: json_schema, strict: true`, in **meal mode** (ignore printed numbers) or **label mode**.
6. **Validate output** — strict Pydantic parse; one repair retry carrying the validation error; then `status: observation_failed`.
7. **Enrich** — fat/sauce defaults from the preparation table; known weight applied; asymmetric range widening for large portions; follow-up questions where kcal sensitivity exceeds threshold.
8. **Match** — personal library → BLS 4.0 → FDC → OFF; top-k with scores; raw vs cooked preferred by preparation.
9. **Calculate** — `nutrient = per_100g × grams / 100` for low/mid/high; sum accepted items only; pending items excluded, `partial: true`.
10. **Review** — items, ranges, questions, pending items.
11. **Recalculate** — answers and edits update state; no second inference unless a new image arrives.
12. **Leftovers (optional)** — `image_after` triggers a second observation; consumed = before − after per item, floored at 0.
13. **Persist (optional)** — corrected meal, reference IDs, calc version, deltas vs. observation. Photos discarded unless the user opts into the benchmark set.

**Limitation.** A single RGB photo can't reveal hidden ingredients, absorbed fat, density or exact mass. The system presents ranges and invites correction; it never implies measurement-grade accuracy.

## 7. API Design

### `POST /api/v1/meals/analyze`

`multipart/form-data`: `image` (required unless `barcode`), `image_2`, `image_after`, `note`, `barcode`, `known_weight_g`, `plate_id`, `meal_slot` (`breakfast|lunch|dinner|snack`), `venue_type` (`home|restaurant|canteen|takeaway|other`), `locale`, `model` (server allowlist).

**Response (illustrative, not a real inference):**

```json
{
  "analysis_id": "ana_01J...",
  "status": "needs_review",
  "partial": true,
  "model": {"id": "allowlisted-model", "provider": "pinned-provider", "prompt_version": "obs-v1"},
  "context": {"meal_slot": "dinner", "venue_type": "home", "plate_id": "plate_27cm"},
  "items": [
    {
      "id": "item_1",
      "label": "Bratkartoffeln",
      "label_de": "Bratkartoffeln",
      "preparation": "pan_fried",
      "bbox": [0.12, 0.30, 0.58, 0.78],
      "density_class": "starchy_cooked",
      "grams": {"low": 150, "mid": 200, "high": 280},
      "portion_confidence": "medium",
      "source": "vision",
      "match": {
        "status": "auto_accepted",
        "provider": "bls",
        "food_id": "BLS-CODE",
        "name": "Bratkartoffeln",
        "score": 0.93,
        "alternatives": [{"provider": "bls", "food_id": "BLS-CODE-2", "name": "Kartoffeln, gekocht"}]
      },
      "nutrition": {
        "kcal": {"low": 180, "mid": 240, "high": 336},
        "protein_g": {"low": 3.5, "mid": 4.6, "high": 6.4},
        "carbs_g": {"low": 24.0, "mid": 32.0, "high": 44.8},
        "fat_g": {"low": 7.5, "mid": 10.0, "high": 14.0}
      }
    },
    {
      "id": "item_2",
      "label": "cooking fat (default)",
      "preparation": null,
      "bbox": null,
      "density_class": "fat",
      "grams": {"low": 5, "mid": 10, "high": 15},
      "portion_confidence": "low",
      "source": "default_rule:pan_fried",
      "match": {"status": "pending", "candidates": ["butter", "sunflower oil", "rapeseed oil"]},
      "nutrition": null
    }
  ],
  "totals": {
    "kcal": {"low": 180, "mid": 240, "high": 336},
    "protein_g": {"low": 3.5, "mid": 4.6, "high": 6.4},
    "carbs_g": {"low": 24.0, "mid": 32.0, "high": 44.8},
    "fat_g": {"low": 7.5, "mid": 10.0, "high": 14.0},
    "excludes_pending_items": ["item_2"]
  },
  "questions": [
    {"id": "q1", "item_id": "item_2", "text": "Fried in butter or oil?", "options": ["butter", "oil", "none"], "max_kcal_impact": 135}
  ],
  "uncertainties": ["Portion estimated from one image", "Absorbed fat not visible"],
  "telemetry": {"latency_ms": 4120, "input_tokens": 1650, "output_tokens": 410, "cost_usd": 0.0021, "cost_is_estimate": false}
}
```

**Contract rules**

- `nutrition` non-null only for `match.status ∈ {confirmed, auto_accepted}` and known grams; otherwise `null`, listed in `excludes_pending_items`, `partial: true`.
- `auto_accepted` requires score ≥ threshold and no close runner-up.
- `source` ∈ `vision | user | barcode | label | default_rule:<rule>`.
- Ranges propagate linearly; user-entered weight collapses the range to a point.
- `bbox` normalized `[x0, y0, x1, y1]`; `density_class` from a fixed enum mapped to a g/ml range in the reference store.

### Later endpoints

- `POST /api/v1/meals/{id}/answers` — answer follow-ups; recalculate.
- `PATCH /api/v1/meals/{id}/items/{item_id}` — change identity, match, grams, preparation.
- `POST /api/v1/meals/{id}/items` — add a missing item.
- `POST /api/v1/meals/{id}/after` — upload after-photo; compute consumed amounts.
- `POST /api/v1/meals/{id}/confirm` — finalize; store correction deltas.
- `POST /api/v1/foods/from-label` — front + label photo → custom food, label mode, `source: label`.
- `GET/POST /api/v1/plates` — register plate/bowl/glass dimensions.
- `GET /health`, `GET /ready`.

## 8. Domain Model

```text
MealAnalysis
 ├── analysis_id, created_at, calc_version
 ├── inputs: image_hashes[], note, barcode, known_weight_g, plate_id,
 │           meal_slot, venue_type
 ├── Observation                 (immutable, as returned by the model)
 │    ├── model_id, provider, prompt_version, prompt_mode (meal | label)
 │    └── ObservedItem[]: label, label_de, preparation, bbox,
 │                        density_class, grams{low,mid,high},
 │                        confidence, notes
 ├── Item[]                      (working state after enrich / match / edit)
 │    ├── source
 │    ├── grams{low,mid,high}, consumed_grams{...} | null
 │    ├── Match (status, provider, food_id, score, alternatives, retrieved_at)
 │    └── Nutrition{low,mid,high} | null
 ├── questions[], answers[]
 ├── totals, partial
 └── telemetry (latency, tokens, cost, retries)
```

Three layers stay separate — **observation**, **match**, **calculation** — so models can be swapped on identical downstream logic and corrections can be measured against the untouched observation.

`bbox` and `density_class` are required in the observation even though Phase 1 only uses grams. They are exactly the inputs the Jul 2026 geometry head consumes; collecting them from day one means the Phase 2 benchmark set doubles as training/eval data for Phase 4.

## 9. OpenRouter Integration

- `OPENROUTER_API_KEY` server-side only; per-environment keys with spend limits.
- `FOOD_VISION_MODELS` allowlist; default + fallback configured.
- Request settings:
  - `response_format: {type: json_schema, json_schema: {strict: true, schema: ...}}`
  - `provider.require_parameters: true`
  - `provider.data_collection: "deny"` and zero-data-retention routing where available; also set account-wide. Verify field names at implementation time.
  - `temperature: 0`; fixed `seed` where supported.
  - **Benchmarks:** `provider.order: [<one provider>]`, `allow_fallbacks: false`, explicit `quantizations`. Default routing load-balances across providers and quantizations and is not reproducible.
  - **Production:** fallbacks among an approved provider list.
- Timeouts connect 5 s / total 30 s; retry only 429/5xx with jitter, max 2.
- Provider-reported usage and cost logged; estimates marked.
- Prompts versioned as package data (`food_vision/observation/prompts/observe_meal_v1.md`, `observe_label_v1.md`) and referenced by hash in telemetry.

### Prompt design (from the 2025–2026 evidence)

- **Decomposition first:** list components, preparation, bbox, density class, grams range; no nutrition.
- **Persona line:** "You are a registered dietitian estimating portions from a photo." Measurable gain in ACETADA.
- **Context block:** meal slot, venue type, plate dimensions, note, typical-portion priors for the likely foods.
- **Meal mode:** "Do not read or use any printed numbers or text in the image."
- **Label mode:** read the nutrition table; output per-100 g values with `source: label`.
- **Density class enum** instead of free-text g/ml, mapped server-side.

### Model selection

Pre-study (Appendix A) decides. Expect the Gemini Flash tier and the Claude Sonnet tier to be the price/accuracy candidates and Claude's bias to differ in shape (less under-estimation of large piles, more error on ordinary portions). Compare signed error and size sensitivity, not only MAE.

### Optional: self-consistency

For low-confidence meals, 2–3 samples (or two models); spread widens/narrows the range. Only if Phase 2 shows the calibration gain pays for the cost.

## 10. Nutrition Reference and Calculation

### Source priority

1. **Personal library** — recipes, custom foods from label scans, recent confirmed meals.
2. **BLS 4.0** — 7,140 foods incl. composite German dishes, 138 nutrients, per 100 g edible portion; missing ≠ zero; data origin per value; recipe values account for cooking losses.
3. **USDA FDC** — Foundation / SR Legacy for gaps.
4. **Open Food Facts** — branded and barcodes; Atwater sanity check `|kcal − (4P + 4C + 9F + 7Alc)|` before auto-accept.

### Supporting tables (versioned data, not prompt text)

| Table | Content | Use |
|---|---|---|
| `density_classes` | enum → g/ml range (e.g. leafy 0.2–0.4, starchy_cooked 0.6–0.9, meat 0.9–1.1, liquid 1.0–1.05, fat 0.9) seeded from FAO/INFOODS density tables | Observation contract; Phase 4 head input |
| `portion_priors` | per food class: typical serving g (p10/p50/p90) by meal slot and venue type, seeded from BLS/NVS II portion data and own corrections | Prompt context; B0 baseline; range fallback |
| `preparation_defaults` | added fat/sauce per 100 g by preparation (pan_fried 5–10, deep_fried 8–15 unless cooked entry exists, roasted veg 3–6, dressed salad 10–20 g per serving, boiled/raw 0) | Enrich step; separate editable item |
| `yield_factors` | raw → cooked weight ratios | Raw/cooked matching |

### Matching

- Normalize labels (lowercase, lemmatize, DE/EN synonyms). Model returns `label` and `label_de`.
- Retrieval: exact/synonym → `rapidfuzz` → embedding nearest-neighbour. Ranking prefers matching preparation state.
- Margin rule for `auto_accepted`; else `candidate` with top-3 alternatives.

### Calculation

```text
for bound in (low, mid, high):
    item[bound] = per_100g × grams[bound] / 100
meal[bound] = Σ item[bound]          # accepted items only
```

- Missing nutrient ≠ 0: propagate `null`, flag.
- Energy from the source record; don't recompute unless missing.
- Asymmetric widening: for `mid > p90` of the class prior, `high = mid × 1.4` instead of `1.3` (single-view under-predicts the heavy tail).
- Pure function, property-tested (monotonic in grams, additive, unit-safe).

### Optional personal calibration (Phase 2+)

Signed error per category from confirmed corrections; visible, reversible correction factor; never silent.

## 11. Privacy and Security

- HTTPS only; no provider credentials on the client.
- EXIF/GPS stripped before inference; photos in memory, discarded by default.
- `venue_type` is chosen by the user or derived on the device; raw GPS never leaves the phone and is never sent to a model.
- Inference restricted to no-training / zero-retention providers; provider chain in the privacy notice.
- No photos, prompts, raw payloads or meal contents in logs; telemetry holds IDs, timings, tokens, cost.
- Benchmark opt-in stores photos encrypted and separately.
- Upload limits, content validation, rate limits; auth before multi-user use.
- Diet logs can reveal health information: GDPR assessment (lawful basis, Art. 9 exposure, processor agreements, DPIA if warranted) before onboarding anyone else.

## 12. Quality Evaluation and Benchmark

### Datasets

| Set | Size | Ground truth | Purpose |
|---|---|---|---|
| **fruit-v1** (own) | 90 images | weighed, measured | Pre-study: model selection (Appendix A) |
| **Own weighed meals** | ≥50 → 100+ | weighed per component, oil recorded, plate ID, meal slot, venue | Target distribution: German home and restaurant food |
| **Nutrition5k** 100-dish test subset | 100 (overhead + 30°) | weighed per ingredient | External anchor; comparable to Liao & Li 2026 flagship numbers |
| **NutritionVerse-Real** | 889 images / 251 dishes | weighed | Freehand phone captures |
| **ACETADA** (when released) | 806 before/after pairs | weighed, dietitian-verified, fiducial marker | Context metadata and leftover evaluation |
| **SNAPMe** | 3,311 phone photos | self-reported ASA24 records | Robustness to messy real-world photos only |
| **NutriBench** sample (text) | — | verified macros | Isolates matching + calculation layer (CC BY-NC-SA) |

Re-run external datasets' ingredient weights through the own calculator to get a "your-database truth" so vision error is separated from USDA-vs-BLS differences.

### Metrics

| Metric | Definition | Why |
|---|---|---|
| Recognition P/R | item-level vs reference | recognition quality |
| Portion MAE (g), PMAE | per item | dominant error; PMAE matches 2026 literature |
| **Signed kcal error** | mean(estimate − truth) | systematic under-counting |
| **Size sensitivity** | Spearman ρ and slope, predicted vs true grams within class | detects prior-only answers |
| kcal / protein / fat MAE | per meal | user-facing |
| **Range coverage** | share of meals with truth ∈ [low, high] | honest ranges, target ~80% |
| Range width | median (high − low) / mid | not useless |
| Post-review error | after ≤30 s review | the number that matters |
| Context gain | error with vs without meal slot / venue / plate | is cheap context worth the UI |
| Question yield | kcal error reduction per question | friction vs gain |
| Correction rate / time | edits, seconds | usability |
| Latency P50/P95, cost per meal, invalid-output rate | — | operations |

### Protocol

- Fixed prompt version, pinned provider, fixed reference-store version per run.
- Conditions: photo only / + note / + context (slot, venue, plate) / + known weight.
- Raw and post-review numbers with bootstrap 95% CIs; no winner when CIs overlap.
- Results as versioned JSON; `food-vision bench --set own-v1 --model X --provider Y`.

### Provisional acceptance targets

- ≥98% schema-valid responses.
- Median latency ≤8 s, P95 ≤15 s.
- Signed kcal bias within ±10% after review.
- Range coverage 70–90%.
- Accuracy thresholds set after the first run, not before.

## 13. Package Structure

Distribution `food-vision`, import package `food_vision` (PEP 8 allows underscores in package names; the `src` layout is the PyPA default and what `uv init --package` generates). Sub-packages are nouns, not verbs, and avoid soft keywords (`match`) and stdlib names.

```text
food-vision/
├── pyproject.toml                  # [project] name = "food-vision"; scripts; ruff + mypy config
├── uv.lock
├── Dockerfile
├── README.md
├── .env.example
├── src/
│   └── food_vision/
│       ├── __init__.py
│       ├── __main__.py             # python -m food_vision
│       ├── cli.py                  # typer app: analyze, bench, build-reference
│       ├── settings.py             # pydantic-settings
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── meals.py            # routes
│       │   └── foods.py
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py           # MealAnalysis, Item, Observation (Pydantic)
│       │   ├── ranges.py
│       │   └── calculator.py       # pure functions
│       ├── observation/
│       │   ├── __init__.py
│       │   ├── adapter.py          # Protocol + OpenRouter implementation
│       │   ├── schema.py           # JSON schema export for the model
│       │   └── prompts/            # package data
│       │       ├── observe_meal_v1.md
│       │       └── observe_label_v1.md
│       ├── enrichment/
│       │   ├── __init__.py
│       │   ├── preparation.py      # fat/sauce defaults
│       │   ├── priors.py           # portion priors, asymmetric widening
│       │   ├── scale.py            # plate registry
│       │   └── questions.py
│       ├── matching/
│       │   ├── __init__.py
│       │   ├── normalize.py
│       │   ├── retrieve.py
│       │   └── rank.py
│       ├── reference/
│       │   ├── __init__.py
│       │   ├── store.py            # SQLite/DuckDB access
│       │   ├── build.py            # BLS xlsx + FDC CSV + OFF parquet + tables → reference.db
│       │   ├── bls.py
│       │   ├── fdc.py
│       │   ├── off.py
│       │   └── personal.py
│       ├── imaging/
│       │   ├── __init__.py
│       │   └── preprocess.py
│       └── pipeline/
│           ├── __init__.py
│           └── analyze.py          # orchestration
├── data/
│   ├── density_classes.csv
│   ├── portion_priors.csv
│   ├── preparation_defaults.csv
│   └── yield_factors.csv
├── bench/
│   ├── sets/                       # manifests only; images outside git
│   ├── runs/
│   └── reports/
└── tests/
    ├── unit/
    ├── property/
    └── integration/
```

Conventions: `ruff` with `N` (pep8-naming), `E`, `W`, `F`, `I`, `B`, `UP`; line length 100; `from __future__ import annotations`; `Protocol` for the vision adapter; no `services/` catch-all. Benchmark code lives in `food_vision.cli`/`food_vision.bench` so it is tested and typed like everything else; `bench/` holds data and outputs only.

## 14. Delivery Roadmap

### Phase P — Pre-study: model selection (2–3 days incl. photo session)

Appendix A. Weighed single-fruit set, frozen dev/test split, 6–8 candidate models, prior baseline.

**Exit:** default and fallback model chosen on test-set evidence; decision whether "dimensions → formula" beats "direct grams".

### Phase 0 — Feasibility (1–2 days)

- Structured-output + ZDR routing confirmed for winner and fallback.
- `reference.db` built from BLS 4.0 (+ FDC) plus density/prior/preparation tables.
- CLI: one image → validated observation → matched items → ranged totals.
- 10 own meals; failure modes noted.

**Exit:** `food-vision analyze meal.jpg --meal-slot dinner --venue home` returns a validated, DB-grounded result with ranges.

### Phase 1 — MVP Engine (4–6 days)

- FastAPI endpoints (analyze, answers, patch, add item, plates).
- Context inputs, preparation defaults, priors, follow-up questions, known weight.
- Barcode short-circuit via local OFF dump; label mode for custom foods.
- Personal library as first match tier.
- Unit, property and mocked integration tests; ruff + mypy clean.

**Exit:** minimal web page captures, reviews, answers, corrects, confirms.

### Phase 2 — Benchmark & Calibration (3–5 days + meal collection)

- ≥50 weighed own meals, Nutrition5k subset, NutritionVerse-Real sample.
- ≥3 model/provider configurations, 4 input conditions.
- Bias, size sensitivity, range coverage, context gain, post-review error, latency, cost.
- Tune priors and preparation defaults; decide on personal calibration and self-consistency.

**Exit:** written evidence on whether post-review photo logging replaces manual logging.

### Phase 3 — Mobile-first UX (later)

- PWA: camera capture, range visualisation, one-tap fixes, repeat-meal shortcuts, label-based food creation, after-photo for leftovers, offline draft capture.
- Optional accounts + PostgreSQL.

### Phase 4 — Portion accuracy (only if Phase 2 says portions dominate post-review error)

- Reimplement the geometry head: frozen DINOv2 ViT-S/14 patch tokens + per-food name/bbox/density range → softmax-ownership volume → mass. Train on Nutrition5k + NV-Real + own set (bbox and density already collected).
- Monocular metric depth experiment (Depth Anything v3 metric / Depth Pro) and depth-free radiance-field volume as alternatives.
- Native iOS with LiDAR as an optional signal.

## 15. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Single photo can't give exact grams | Ranges, priors, plate reference, context, known weight, second angle |
| Heavy-tail under-estimation | Asymmetric ranges; size-sensitivity metric; Phase 4 head |
| Systematic fat under-estimation | Preparation defaults, targeted questions, signed-error tracking |
| Wrong DB match | Personal library first, margin rule, alternatives, provenance |
| Model invents nutrients | Nutrients only from reference records; micronutrients never from vision |
| Model reads printed numbers instead of estimating | Meal vs label prompt modes |
| Non-reproducible benchmark | Pinned provider/quantization, versioned prompts and reference store |
| Model price/availability change | Allowlist, adapter Protocol, re-runnable benchmark |
| OFF data quality / licence | Atwater check, separate table, local dump |
| Photos or location leaving the device | EXIF strip, no GPS to model, ZDR routing, no storage |
| Too many questions | kcal-impact threshold; max 3; question-yield metric |
| Over-engineering before evidence | Pre-study first, stateless PoC, Phase 4 gated on data |

## 16. Decision Record

| ID | Decision | Status |
|---|---|---|
| ADR-001 | OpenRouter as initial inference gateway | Accepted |
| ADR-002 | Python 3.13, `uv`, FastAPI, Pydantic v2, ruff + mypy in CI | Accepted |
| ADR-003 | Vision observes; deterministic code calculates; no nutrient ever from the model | Accepted |
| ADR-004 | Inference credentials backend-only | Accepted |
| ADR-005 | Versioned HTTP API, PWA first | Proposed |
| ADR-006 | Source photos ephemeral by default; GPS never sent to a model | Accepted |
| ADR-007 | Pre-study on weighed single fruit selects the model before pipeline work | Accepted |
| ADR-008 | Defer accounts and diary | Accepted |
| ADR-009 | BLS 4.0 primary generic reference; local reference store | Accepted |
| ADR-010 | Portions and totals as low/mid/high ranges, asymmetric for large portions | Accepted |
| ADR-011 | Hidden-fat preparation defaults as separate editable items | Accepted |
| ADR-012 | ZDR / no-data-collection routing; pinned providers for benchmarks | Accepted |
| ADR-013 | Observation schema requires bbox + density class for a future geometry head | Accepted |
| ADR-014 | Personal library is the first match tier | Proposed |
| ADR-015 | Meal slot and venue type as first-class context inputs | Accepted |
| ADR-016 | Two prompt modes: meal (ignore printed numbers) and label (read them) | Accepted |
| ADR-017 | `src` layout; import package `food_vision`; distribution `food-vision` | Accepted |
| ADR-018 | Optional after-photo for leftovers (Phase 3) | Proposed |

## 17. Definition of Done (Prototype)

- [ ] One request accepts a phone photo with optional note / weight / barcode / plate / meal slot / venue.
- [ ] Image normalized, metadata stripped, never logged or stored by default.
- [ ] Strict-schema observation incl. bbox and density class from an allowlisted model via ZDR routing.
- [ ] Every nutrient traceable to a reference record with provenance.
- [ ] Ranges on every item and total; pending items excluded and flagged.
- [ ] Fat defaults, priors and follow-up questions working; answers recalculate without re-inference.
- [ ] Credentials never in client or logs.
- [ ] ruff and mypy clean; calculator property tests and mocked integration tests pass.
- [ ] Benchmark report: bias, size sensitivity, MAE/PMAE, range coverage, context gain, post-review error, latency, cost, across ≥3 configurations.
- [ ] README: setup, reference-DB build, attributions (BLS DOI, OFF ODbL, FAO density), limitations, privacy.

---

**Recommendation:** run the pre-study first, then build the vertical slice as a CLI: *photo + context → strict observation → BLS-grounded match → ranged totals*. Collect weighed meals from day one with bbox-capable ground truth, because that set is also the training data for the one upgrade that demonstrably moves portion accuracy.

## Appendix A — Pre-study: vision model selection on a single-fruit set

### A.1 Why start this simple

With a single whole fruit, recognition is trivial and nutrient density is a known constant (BLS, per 100 g edible portion). **kcal error = gram error × constant**, so the study measures one thing: can the model perceive size? That capability dominates real-meal error and needs no matching or calculation layer.

**Trap:** a model scores acceptable MAPE by always answering "a medium banana is ~120 g". Specimens must span small → large per type, and the key metric is whether predictions track true weight.

### A.2 Reference set `fruit-v1`

| Dimension | Values |
|---|---|
| Fruit types | banana, apple, orange, pear, kiwi, mandarin (optional: avocado, cucumber, bell pepper) |
| Specimens per type | 5, from smallest to largest available |
| Capture conditions | C1 top-down with reference card · C2 ~45° with card · C3 ~45° no reference |
| Total | 6 × 5 × 3 = **90 images** (~2 h incl. shopping and weighing) |

Optional later levels: **L2** counted items (3 apples, grapes), **L3** cut items (half apple, sliced banana in a bowl). No mixed plates in the pre-study.

**Capture protocol.** Same phone, same neutral board, daylight, no zoom, 30–40 cm. Reference: any ID-1 card (85.60 × 53.98 mm), flat, fully visible, no personal data facing the camera. No cropping; originals kept.

**Ground truth per specimen.** `whole_g`; `edible_g = whole_g − waste_g` (peel/core weighed after cutting; BLS is per edible portion); `length_cm`, `max_diameter_cm` (banana: outer curve + mid diameter); `kcal_ref = edible_g × BLS kcal/100 g` with BLS code and version.

**Manifest** (`bench/sets/fruit-v1/manifest.jsonl`):

```json
{"image": "banana_03_c2.jpg", "specimen_id": "banana_03", "type": "banana",
 "condition": "c2_45deg_ref", "split": "test",
 "whole_g": 184, "edible_g": 121, "length_cm": 21.5, "max_diameter_cm": 3.8,
 "bls_code": "BLS-CODE", "kcal_ref": 113, "captured": "2026-10-10"}
```

### A.3 Dev / test split

- Split **by specimen**, never by image.
- dev: 2 specimens per type (36 images) for prompt iteration and formula fitting.
- test: 3 specimens per type (54 images), **frozen**, run once per final candidate.
- Assign by size rank (dev = ranks 2 and 4) so both splits span the range.
- Version the set; any change creates `fruit-v2`.

### A.4 Strategies

| ID | Prompt strategy | Calculation |
|---|---|---|
| S1 | "Estimate edible grams" | direct |
| S2 | "Measure length and max diameter in cm using the card as scale" | per-type formula fitted on dev (banana `a × L × D²`; round fruit ellipsoid × density × edible ratio) |
| S3 | S1 plus the class prior in the prompt ("a banana is typically 90–180 g edible") | direct; tests the Vinod et al. finding |
| B0 | no model: dev-set mean weight per type | — |
| B1 | S2 formula on true tape measurements | upper bound for S2 |

S2 tests whether length and thickness determine weight: the model measures, code converts. B1 separates formula error from measuring error. S3 tests whether a stated prior helps or makes the model lazy (slope → 0).

### A.5 Candidates and run protocol

- 6–8 models via OpenRouter: 2–3 frontier (incl. Gemini Flash tier and Claude Sonnet tier), 2 economical, 2 open-weight. Verify image support, structured output and ZDR eligibility on the day.
- Pinned provider, `allow_fallbacks: false`, fixed quantization, temperature 0, strict schema.
- Same normalized image (1024 px; extra 768 px run for the top 2).
- 3 repeats per image.
- Prompts iterated on dev only; freeze; one test run.

### A.6 Metrics and decision rule

| Metric | Definition | Pass bar (provisional) |
|---|---|---|
| Edible-gram MAPE | per image | beats B0 by ≥30% relative |
| **Size sensitivity** | Spearman ρ and regression slope within type | ρ ≥ 0.7, slope 0.7–1.3 |
| Signed bias | mean (pred − true) / true | within ±10% |
| Reference gain | MAPE(C3) − MAPE(C2) | shows scale cues are used |
| Prior effect | MAPE(S3) − MAPE(S1) and slope change | prior helps without flattening slope |
| Repeat stability | mean CV over 3 repeats | ≤5% |
| Schema-valid rate | first try | ≥98% |
| Latency P50/P95, cost per 1,000 images | provider-reported | tie-breakers |

Bootstrap 95% CIs on test. Among models passing size sensitivity and bias, pick lowest MAPE; overlapping CIs → cheaper/faster. Second-best from a different vendor is the fallback. If no model beats B0 meaningfully, stop and rethink before Phase 0.

### A.7 Deliverables

- `bench/sets/fruit-v1/` — manifest, README (protocol, BLS codes, split rule); images outside git.
- `food-vision bench --set fruit-v1 --split dev|test --model X --provider Y --strategy S1|S2|S3`
- `bench/runs/<date>_<model>_<strategy>.json`
- `bench/reports/prestudy-v1.md` — table, per-type scatter, decision.

### A.8 After the pre-study

Sanity-check the winner on ~30 dishes of the Nutrition5k 100-dish test subset to confirm the fruit ranking holds on mixed plates, and compare against the published flagship numbers (24–36 g per-food MAE). Then Phase 0.

## Sources

- Nutrition5k (CVPR 2021): https://arxiv.org/pdf/2103.03375 ; 100-dish test subset: https://www.huggingface.co/datasets/madroid/nutrient5k-test-100
- Liao & Li, Geometry-Enhanced Portion Estimation for Multimodal LLMs (Jul 2026): https://arxiv.org/abs/2607.16514
- ACETADA benchmark, Comprehensive Evaluation of LMMs for Nutrition Analysis with Contextual Metadata (Jul 2025): https://arxiv.org/abs/2507.07048
- DietDelta, before/after dietary assessment (Apr 2026): https://arxiv.org/pdf/2604.06352
- Vinod et al., Not Your Stereo-Typical Estimator (Apr 2026): https://arxiv.org/pdf/2604.09886
- NutriMLLM, micronutrient estimation (Jun 2026): https://arxiv.org/pdf/2606.08948
- Nakagawa & Yamamoto, Prompt Engineering and Model Selection for LLM-Based Nutritional Estimation, Nutrients 2026: https://www.citedrive.com/en/discovery/prompt-engineering-and-model-selection-for-llm-based-nutritional-estimation-from-food-images-a-multi-dataset-investigation/
- Open-KNEAD, knowledge-grounded agentic nutrition estimation (Jul 2026, not reviewed in detail): https://arxiv.org/abs/2607.12911
- Closed-loop multi-agent meal-level nutrition management (Jan 2026): https://arxiv.org/abs/2601.04491
- CVPR 2026 MetaFood Workshop programme: https://openaccess.thecvf.com/CVPR2026_workshops/MTF
- Food Portion Estimation: From Pixels to Calories (survey, Feb 2026): https://arxiv.org/pdf/2602.05078
- LMM nutrition benchmark incl. known-weight effect (ACM BCB 2025): https://digitalcommons.odu.edu/computerscience_fac_pubs/417
- Two-step decomposition prompting (CVPRW 2025): https://repository.li.mahidol.ac.th/handle/123456789/112492
- NutritionVerse-Real: https://arxiv.org/abs/2401.08598v1 ; MetaFood3D: https://arxiv.org/abs/2409.01966v2 ; SNAPMe: https://escholarship.org/uc/item/5fw4h06p ; NutriBench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/ef3a57e4f26b640e6f90d78cbb011feb-Abstract-Conference.html
- NIH / ASN NUTRITION 2026, photo apps underestimate energy and fat: https://www.powershealth.org/about-us/newsroom/health-library/2026/07/27/how-accurate-are-photo-based-calorie-apps-4-are-put-to-the-test
- MacroFactor AI food logging: https://help.macrofactorapp.com/en/articles/258-ai-food-logging ; https://macrofactor.com/version-5-0-0/
- SnapCalorie: https://similarweb.com/app/apple/1574239307
- BLS 4.0: https://blsdb.de/faq ; https://heise.de/-11123877
- FAO/INFOODS density database: https://openknowledge.fao.org/handle/20.500.14283/ap815e
- Open Food Facts API and licence: https://openfoodfacts.github.io/openfoodfacts-server/api/ ; https://support.openfoodfacts.org/help/en-gb/12-api-data-reuse/94-are-there-conditions-to-use-the-api
- OpenRouter structured outputs and provider routing: https://openrouter.ai/docs/features/structured-outputs ; https://openrouter.ai/docs/features/provider-routing
- Depth Anything v3 metric / Depth Pro: https://replicate.com/vufinder/depth-anything-v3-metric ; https://replicate.com/chenxwh/ml-depth-pro
- PEP 8, package and module names: https://peps.python.org/pep-0008/#package-and-module-names ; PyPA src layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/