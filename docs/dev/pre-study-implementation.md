# Pre-study Implementation Concept — Appendix A (Phase P)

> **Superseded (interface layer only):** the CLI-only interface described
> below was replaced by a local FastAPI + SQLite web UI once the real
> requirement surfaced — persisted, queryable runs and publication-quality
> charts, not a one-off batch sweep. See
> `docs/dev/pre-study-web-ui.md` for the current implementation concept.
> The Appendix A **domain** decisions here (dataset, strategies, formulas,
> metrics, decision rule in §4–§7) are unchanged and still the source of
> truth; only §3 (package layout) and §5.8/§9 (`cli.py` / sequencing around
> the CLI command) are superseded.

Status: planning document for implementation. Scope is exactly Appendix A
of `docs/dev/food-vision-concept.md` ("Pre-study: vision model selection on
a single-fruit set") — **CLI only**, no API, no DB, no UI. Anything not
needed to answer Appendix A's decision rule (§A.6) is out of scope here,
even if it appears elsewhere in the concept's package structure (§13).

Cross-references below use `concept.md §N` for the technical concept.

## 1. Goal and exit criterion

Answer, with test-set evidence: which OpenRouter model (and which
prompt/calculation strategy) estimates single-fruit edible grams well
enough to justify building the real pipeline on it — or whether none does
and Phase 0 should be rethought (concept §A.6, last line).

Exit: default + fallback model chosen; decision on "dimensions → formula"
(S2) vs. "direct grams" (S1), recorded in `bench/reports/prestudy-v1.md`.

## 2. What this explicitly does **not** build yet

Per the prior review of §13 against Appendix A, the following are **not**
needed for Phase P and must not be added speculatively:

- `api/`, `pipeline/analyze.py`, `enrichment/`, `matching/` — no matching
  or enrichment layer; single fruit, known BLS code, no mixed plates.
- `reference/build.py`, `reference/fdc.py`, `reference/off.py`,
  `reference/store.py` as a real BLS/FDC/OFF build pipeline — only a
  handful of kcal/100g values for 6–8 fruit types are needed.
- Any database (SQLite/DuckDB/Postgres), FastAPI, Uvicorn, or UI.
- `domain/ranges.py` (low/mid/high widening) — fruit weight is a point
  estimate here, not a ranged meal total.
- Self-consistency / multi-model ensembling beyond the 3-repeat protocol.

These stay reserved for Phase 0+ as already laid out in concept §13/§14.

## 3. Package layout added in this phase

Builds on the existing `src/food_vision/{config,proxy,utils}` (already
implemented: settings, `OpenRouterClient` with `RoutingMode.BENCHMARK`,
`log_factory`). New additions, scoped to Appendix A only:

```text
src/food_vision/
├── cli.py                      # new: typer app, `bench` command only
├── observation/
│   ├── __init__.py
│   ├── schema.py                # JSON schema for S1/S2/S3 output, per strategy
│   ├── prompts/
│   │   ├── observe_fruit_s1.md  # "Estimate edible grams"
│   │   ├── observe_fruit_s2.md  # "Measure length and max diameter in cm..."
│   │   └── observe_fruit_s3.md  # S1 + class prior
│   └── adapter.py               # thin wrapper: build messages+schema, call OpenRouterClient
├── domain/
│   ├── __init__.py
│   └── calculator.py            # pure: S2 formulas, kcal = edible_g * kcal_100g / 100
├── imaging/
│   ├── __init__.py
│   └── preprocess.py            # resize/normalize to 1024px (+768px variant)
└── bench/
    ├── __init__.py
    ├── fruit_reference.py        # tiny in-code/CSV kcal_100g lookup per bls_code
    ├── runner.py                 # orchestration: load manifest, run strategy x model x repeats
    └── metrics.py                # MAPE, Spearman rho, slope, CV, schema-valid rate, bootstrap CI

data/
└── fruit_kcal_100g.csv           # bls_code,label,kcal_100g — 6-8 rows, versioned

bench/
├── sets/
│   └── fruit-v1/
│       ├── manifest.jsonl        # tracked; images outside git (see .gitignore)
│       └── README.md             # capture protocol, BLS codes, split rule
├── runs/                         # <date>_<model>_<strategy>.json, gitignored
└── reports/
    └── prestudy-v1.md            # tracked once written
```

Notes:
- `bench/` here is a **new sub-package** (`food_vision/bench/`) for the
  runner/metrics *code*, distinct from the top-level `bench/` *data*
  directory that already exists in §13 for manifests/runs/reports. Same
  name, different role — keep this distinction explicit in module
  docstrings to avoid confusion.
- `cli.py` only grows a `bench` command now; `analyze`/`build-reference`
  stay for Phase 0+.
- `fruit_reference.py` is intentionally not `reference/bls.py` — it is a
  throwaway 6–8-row lookup, not the real BLS importer.

## 4. Data contracts

### 4.1 Manifest (already specified, concept §A.2)

`bench/sets/fruit-v1/manifest.jsonl`, one JSON object per image:

```json
{"image": "banana_03_c2.jpg", "specimen_id": "banana_03", "type": "banana",
 "condition": "c2_45deg_ref", "split": "test",
 "whole_g": 184, "edible_g": 121, "length_cm": 21.5, "max_diameter_cm": 3.8,
 "bls_code": "BLS-CODE", "kcal_ref": 113, "captured": "2026-10-10"}
```

Load via a `pydantic.BaseModel` (`ManifestEntry`) in `bench/runner.py` —
validated once at load time, not re-validated per request.

### 4.2 Observation schema (per strategy)

Each strategy gets its own strict JSON schema in `observation/schema.py`,
since S1/S2/S3 ask for different fields:

- **S1 / S3** (`FruitGramEstimate`): `{"edible_g": number}`.
- **S2** (`FruitDimensionEstimate`): `{"length_cm": number, "max_diameter_cm": number}`.

Both are plain `dict[str, Any]` JSON Schema objects passed to
`JsonSchemaFormat(name=..., schema=..., strict=True)` — already supported
by `OpenRouterClient.complete(response_format=...)`. No new Pydantic
response model is required beyond what `observation/adapter.py` validates
on parse.

### 4.3 Run record (`bench/runs/<date>_<model>_<strategy>.json`)

One record per (image, model, strategy, repeat):

```json
{
  "run_id": "2026-10-20_google-gemini-flash_s2",
  "image": "banana_03_c2.jpg",
  "specimen_id": "banana_03",
  "type": "banana",
  "split": "test",
  "model": "google/gemini-flash-...",
  "provider": "google-vertex",
  "strategy": "S2",
  "repeat": 1,
  "resolution_px": 1024,
  "raw_observation": {"length_cm": 19.8, "max_diameter_cm": 3.6},
  "edible_g_pred": 118.4,
  "edible_g_true": 121,
  "kcal_pred": 110.6,
  "kcal_true": 113,
  "schema_valid": true,
  "latency_ms": 842.0,
  "cost_usd": 0.00031,
  "cost_is_estimate": false,
  "attempts": 1,
  "error": null
}
```

One JSON file per (model, strategy, date) holding a list of these records
— matches the file-naming convention already in concept §A.7.

## 5. Component responsibilities

### 5.1 `imaging/preprocess.py`

- `normalize_image(path: Path, max_px: int) -> bytes` — decode, resize so
  the longer edge is `max_px` (1024 default; 768 for the extra run on the
  top-2 models per §A.5), strip EXIF, re-encode JPEG quality ~90.
- No cropping, no reference-card detection logic — the card is for the
  *model* to use as a scale cue, not for our code to process.
- Pure function given bytes in, bytes out; easy to unit test with a tiny
  fixture image.

### 5.2 `observation/prompts/*.md`

- Plain markdown/text prompt bodies, one file per strategy, loaded as
  package data (`importlib.resources`), never string-built inline — keeps
  prompts versionable and hashable for telemetry, consistent with concept
  §9 ("Prompts versioned as package data... referenced by hash").
- S1: "Estimate edible grams" (meal-mode framing: no nutrition, no label
  reading).
- S2: "Measure length and max diameter in cm using the card as scale."
- S3: S1 text plus one line: "A {type} is typically {p10}-{p90} g edible."
  — the prior values come from `data/fruit_kcal_100g.csv`'s sibling table
  or are hardcoded constants in the prompt file per type; no need for the
  full `portion_priors` table from concept §10.

### 5.3 `observation/adapter.py`

Thin wrapper, not a new abstraction layer:

```python
def observe(
    client: OpenRouterClient,
    *,
    image_bytes: bytes,
    strategy: Strategy,
    model: str,
    routing_mode: RoutingMode,
    provider_pin: str | None,
    quantizations: Sequence[str] | None,
) -> ObservationResult:
    """Build the strategy's prompt + schema, call OpenRouterClient.complete,
    parse into FruitGramEstimate | FruitDimensionEstimate, raise on
    schema_valid=False after exhausting the client's own retries."""
```

- Builds the OpenAI-style multimodal message (`image_url` with a base64
  data URL from `image_bytes`, plus the strategy's prompt text).
- Delegates all transport/retry/telemetry to the already-implemented
  `OpenRouterClient` — this module adds zero HTTP logic.
- Returns a small typed result carrying `schema_valid`, the parsed dict
  (or `None`), and everything from `CompletionResult` the runner needs
  (`latency_ms`, `usage`, `attempts`, `provider`).

### 5.4 `domain/calculator.py`

Pure functions, property-tested (concept §10's "pure function,
property-tested" discipline applied narrowly here):

- `banana_volume_g(length_cm, diameter_cm, density_g_cm3) -> float` —
  `a * L * D**2` fit; `a` fitted on dev split, stored as a constant (or a
  small per-type dict) once fitted, not re-fitted per run.
- `ellipsoid_mass_g(length_cm, diameter_cm, density_g_cm3, edible_ratio) -> float`
  for round fruit (apple, orange, pear, kiwi, mandarin).
- `kcal_from_edible_g(edible_g, kcal_100g) -> float` — `edible_g * kcal_100g / 100`.
- No ranges, no `low/mid/high` — Appendix A metrics are point estimates
  (MAPE, slope, bias), per §A.6.

### 5.5 `bench/fruit_reference.py`

- `load_kcal_lookup() -> dict[str, float]` reading `data/fruit_kcal_100g.csv`
  (`bls_code -> kcal_100g`), and `DENSITY_G_CM3: dict[str, float]` +
  `EDIBLE_RATIO: dict[str, float]` per fruit type as module constants
  (seed from the same FAO/INFOODS-style source concept §10 mentions, but
  only the 6–8 rows needed — not the full `density_classes` table).
- This is deliberately *not* `reference/bls.py`; no DB, no build step.

### 5.6 `bench/runner.py`

- `load_manifest(path) -> list[ManifestEntry]`
- `run_one(entry, *, strategy, model, provider_pin, quantizations, resolution_px, repeat) -> RunRecord`
  — preprocess image → `observation.adapter.observe` → (for S2) run through
  `domain.calculator` → compute `kcal_pred` via `fruit_reference` lookup →
  assemble `RunRecord`.
- `run_set(manifest, *, split, strategy, model, ..., repeats=3) -> list[RunRecord]`
  — iterates specimens in the given split, writes
  `bench/runs/<date>_<model>_<strategy>.json`.
- Baselines B0 (dev-set mean weight per type, no model call) and B1 (S2
  formula on true tape measurements, no model call) are handled as
  special-cased strategies in the runner, not via the model adapter.

### 5.7 `bench/metrics.py`

Pure functions over a list of `RunRecord` (or a `pandas`/plain-list
aggregation — decide based on whether `pandas` is already a dependency;
if not, plain `statistics`/`numpy` is enough for this data volume):

- `mape(records) -> float`
- `spearman_rho_and_slope(records) -> tuple[float, float]` (per type, then
  aggregated — needs `scipy.stats.spearmanr` or a small manual
  implementation if we don't want a `scipy` dependency for ~90 points).
- `signed_bias(records) -> float`
- `repeat_cv(records) -> float` (coefficient of variation across the 3
  repeats per image).
- `schema_valid_rate(records) -> float`
- `bootstrap_ci(values, stat_fn, n=2000, alpha=0.05) -> tuple[float, float]`
  on the test split only.

Output feeds `bench/reports/prestudy-v1.md` (table + decision per §A.7);
the report itself can be hand-assembled from a `--report` CLI flag that
dumps a markdown table, or written by hand from the JSON — doesn't need
to be fully automated for a one-off pre-study, but computing the numbers
must be code, not spreadsheet work, so it's reproducible and testable.

### 5.8 `cli.py`

```text
food-vision bench --set fruit-v1 --split {dev,test} \
    --model <openrouter-model-id> --provider <pinned-provider> \
    --quantization <q> [--quantization <q> ...] \
    --strategy {S1,S2,S3,B0,B1} \
    --resolution {1024,768} --repeats 3 \
    [--out bench/runs/]
```

- `typer` app (already a concept §13 convention); one command for Phase P.
- Validates `--model` against `FOOD_VISION_MODELS` (reuses existing
  settings validation) before making any request.
- `--provider`/`--quantization` required when strategy isn't a baseline
  (B0/B1), matching `OpenRouterClient`'s existing requirement that
  `RoutingMode.BENCHMARK` must carry a pin + quantizations.

## 6. Run protocol mapping (concept §A.5) → code

| Protocol requirement | Implementation |
|---|---|
| 6–8 models via OpenRouter | `--model` CLI arg, looped by a shell/make script across the candidate list; not hardcoded in Python |
| Pinned provider, `allow_fallbacks: false`, fixed quantization | `RoutingMode.BENCHMARK` + `--provider`/`--quantization` → already implemented in `OpenRouterClient._build_provider_payload` |
| Temperature 0, strict schema | `OPENROUTER_TEMPERATURE` setting (already defaults to 0) + `JsonSchemaFormat(strict=True)` |
| Same normalized image; 1024px, 768px for top 2 | `imaging.preprocess.normalize_image(max_px=...)`, `--resolution` flag |
| 3 repeats per image | `--repeats 3`, loop in `run_one`/`run_set` |
| Prompts iterated on dev only, frozen for test | `--split dev` while iterating `observation/prompts/*.md`; freeze (git commit) before any `--split test` run |

## 7. Metrics and decision rule (concept §A.6) → code

| Metric | Pass bar | Function |
|---|---|---|
| Edible-gram MAPE | beats B0 by ≥30% relative | `metrics.mape` |
| Spearman ρ / slope | ρ ≥ 0.7, slope 0.7–1.3 | `metrics.spearman_rho_and_slope` |
| Signed bias | within ±10% | `metrics.signed_bias` |
| Reference gain (C3 vs C2 MAPE) | shows scale cues used | `mape` filtered by `condition`, diffed |
| Prior effect (S3 vs S1) | helps without flattening slope | `mape`/`slope` diffed across strategy |
| Repeat CV | ≤5% | `metrics.repeat_cv` |
| Schema-valid rate | ≥98% | `metrics.schema_valid_rate` |
| Latency P50/P95, cost/1000 | tie-breakers | derived from `RunRecord.latency_ms`/`cost_usd` |

All bootstrap CIs computed on the **test** split only, per §A.6.

## 8. Testing plan

- `tests/unit/`
  - `imaging/preprocess`: resize correctness, EXIF stripped, no crop.
  - `observation/schema`: schema shape per strategy is valid JSON Schema.
  - `observation/adapter`: mocked `OpenRouterClient` — message/schema
    built correctly per strategy; `schema_valid=False` handled.
  - `bench/fruit_reference`: lookup returns expected kcal for known codes,
    raises/flags on unknown code.
  - `cli`: argument validation (model allowlist, benchmark routing
    requires provider+quantization), no live network calls.
- `tests/property/`
  - `domain/calculator`: monotonic in length/diameter, unit-safe
    (g, cm, g/cm³ dimensional sanity), banana/ellipsoid formulas against
    hand-computed fixtures.
  - `bench/metrics`: MAPE/bias zero on perfect predictions; CV zero on
    identical repeats; bootstrap CI contains the point estimate.
- `tests/integration/`
  - `bench/runner.run_one` against a recorded/fixture OpenRouter response
    (no live call) — exercises the full preprocess → observe → calculate
    → record path end to end.
- No test depends on `OPENROUTER_API_KEY` or live network, consistent with
  the existing proxy test suite.

## 9. Sequencing

1. `data/fruit_kcal_100g.csv` + `bench/fruit_reference.py` (no model
   dependency; unblocks everything else).
2. `imaging/preprocess.py` + tests.
3. `observation/schema.py` + `observation/prompts/*.md` + `observation/adapter.py`
   (unit-tested against a mocked `OpenRouterClient`).
4. `domain/calculator.py` + property tests.
5. `bench/runner.py` + `bench/metrics.py` + integration test with a fixture
   response.
6. `cli.py` `bench` command wiring everything together.
7. Capture `fruit-v1` images + manifest (concept §A.2) — can happen in
   parallel with 1–6.
8. Dev-split runs: iterate prompts/formula fit per §A.3–A.4.
9. Freeze; one test-split run per final candidate; write
   `bench/reports/prestudy-v1.md`.

## 10. Open questions for before step 5–6

- `scipy` vs. hand-rolled Spearman/bootstrap: pull in `scipy` (small, pure
  numerical dependency, no native build issues) or implement the ~20 lines
  needed for Spearman ρ and percentile bootstrap manually to keep the
  dependency surface minimal? Lean towards `scipy` given `numpy`-style
  math is otherwise absent from this codebase and reimplementing
  bootstrap/rank-correlation correctly is easy to get subtly wrong.
- Candidate model list (6–8 OpenRouter models, §A.5) — needs to be
  confirmed on the day against current ZDR/structured-output/image
  support, not decided in this document.
- Exact `a` constant for the banana volume formula and `density_g_cm3`
  per round-fruit type — fitted on the dev split once images/weights
  exist; placeholder constants should not be committed as if final.
