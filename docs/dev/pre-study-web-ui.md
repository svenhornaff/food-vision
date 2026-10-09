# Pre-study Implementation Concept — Web UI (Appendix A, Phase P)

Status: supersedes the **interface layer** of
`docs/dev/pre-study-implementation.md` (which was CLI-only). The Appendix A
*domain* decisions — dataset, strategies, metrics, decision rule — are
unchanged and are not repeated in full here; see that doc and concept
§Appendix A for the source of truth. What changes is how a run is
triggered and where results live: a local FastAPI + SQLite web app
instead of a `food-vision bench` CLI command, because the actual need is
persisted, queryable evidence and publication-quality charts, not just a
one-off batch sweep.

## 1. Why the interface changed

CLI-only was the right default for "run a sweep, read a markdown report."
It stopped being the right choice once the real requirement surfaced:

- Every run (not just a final frozen sweep) needs to be **persisted and
  queryable** — flat JSON-per-sweep files don't support "show me all S2
  runs on bananas across every model I've tried so far."
- Params (model, provider pin, quantization, strategy, temperature,
  resolution, repeats) need to be **chosen interactively**, including
  outside the frozen protocol, for exploration before/around the frozen
  test-split run.
- Charts must be **reusable in a paper** — static, high-DPI, reproducible
  from stored history, not just printed to a terminal or a one-off script.

A database + a small local web UI solves this proportionately; a full
FastAPI *service* in the production sense is not what this is — see §3 for
the explicit boundary with the concept's future `api/`.

## 2. Still explicitly out of scope

Unchanged from the CLI plan's §2 non-goals — restated because the web UI
shape makes it tempting to reach for more:

- No `matching/`, `enrichment/`, ranges (`low/mid/high`) — single fruit,
  point estimates only.
- No real `reference/` (BLS/FDC/OFF) build pipeline — a 6–8-row kcal/
  density/edible-ratio lookup (`bench/fruit_reference.py`) is enough.
- No multi-user auth, no remote exposure — **localhost only**. This is a
  personal research tool, not a deployment. Document this explicitly in
  the README; do not add CORS/auth scaffolding that implies otherwise.
- No durable job queue (Celery/RQ/etc.) — FastAPI `BackgroundTasks` is
  enough at this volume and concurrency (one researcher, one process).
- No SQLAlchemy/SQLModel — stdlib `sqlite3` is enough at ~3–4k rows; an
  ORM would be a dependency with no payoff here.

## 3. Namespace: kept out of the future production `api/`

Concept §13 reserves `food_vision/api/{app.py,meals.py,foods.py}` for the
real Phase 0+ meal-analysis service. This tool is a different thing
(a research/benchmark instrument) and must not collide with that path
once Phase 0 starts. It lives under a new `food_vision.bench` package,
served at `/prestudy`, with its own Uvicorn entry point:

```text
uvicorn food_vision.bench.web.app:create_app --factory --port 8800
```

`food_vision.bench` holds **code** (domain + persistence + web), while the
existing top-level `bench/` directory keeps holding **data/output**
(`bench/sets/`, `bench/reports/`) — same distinction already called out
in the CLI plan's §3, still true here.

## 4. Package layout

```text
src/food_vision/
├── observation/
│   ├── __init__.py
│   ├── schema.py                # strict JSON schema per strategy (S1/S2/S3)
│   ├── adapter.py                # builds multimodal message, calls OpenRouterClient
│   └── prompts/
│       ├── observe_fruit_s1.md
│       ├── observe_fruit_s2.md
│       └── observe_fruit_s3.md
├── domain/
│   ├── __init__.py
│   └── calculator.py              # pure S2 formulas, kcal_from_edible_g
├── imaging/
│   ├── __init__.py
│   └── preprocess.py              # decode, resize/normalize, strip EXIF, re-encode
└── bench/
    ├── __init__.py
    ├── db.py                       # sqlite3 connection/schema/queries (no ORM)
    ├── dataset.py                  # fruit-v1 manifest import -> samples rows
    ├── runs.py                     # batch lifecycle, repeat execution, baselines
    ├── fruit_reference.py          # tiny kcal/density/edible-ratio lookup
    ├── metrics.py                  # MAPE, Spearman rho+slope, bias, CV, bootstrap CI
    ├── charts.py                   # matplotlib SVG+PNG export from stored history
    └── web/
        ├── __init__.py
        ├── app.py                  # create_app(), lifespan, routes, DI
        ├── templates/
        │   ├── base.html
        │   ├── new_run.html         # upload + param form
        │   ├── batch_status.html    # polling page for a running batch
        │   ├── history.html         # queryable run list/filter
        │   └── charts.html          # chart gallery + download links
        └── static/
            └── prestudy.css

data/
└── fruit_kcal_100g.csv             # bls_code -> kcal_100g, density_g_cm3, edible_ratio

var/prestudy/                        # gitignored: runtime state, not source
├── prestudy.db                      # sqlite3 file
├── uploads/                          # raw uploaded images, content-hashed filenames
└── normalized/                       # preprocessed derivatives used for the actual call

bench/
├── sets/fruit-v1/{manifest.jsonl,README.md}   # tracked; images outside git
└── reports/prestudy-web/charts/     # chart-{id}-{kind}.{svg,png}, tracked once curated
```

Notes:

- `var/prestudy/` (not `bench/runs/`) holds mutable runtime state —
  uploaded images and the SQLite file are not meant to be versioned, so
  they get their own gitignored root distinct from the data directories
  that *are* tracked (`bench/sets/`, `data/`).
- Package data (`observation/prompts/*.md`, `bench/web/templates/*.html`,
  `bench/web/static/*.css`) is included automatically by `uv_build`'s
  default "everything under `src/food_vision/`" packaging — verify with
  `uv build --no-sources` once these files exist; no extra
  `[tool.uv_build]` config expected to be needed.

## 5. Dependencies added

```toml
dependencies = [
  # existing: pydantic, pydantic-settings, openai, httpx
  "fastapi>=0.115.0,<1.0.0",
  "uvicorn[standard]>=0.32.0,<1.0.0",
  "python-multipart>=0.0.12",   # multipart/form-data upload parsing
  "jinja2>=3.1.0,<4.0.0",
  "pillow>=11.0.0,<12.0.0",      # decode/resize/EXIF-strip, decompression-bomb guard
  "matplotlib>=3.9.0,<4.0.0",    # Agg backend, SVG+PNG chart export
]
```

Deliberately **not** added: SQLAlchemy/SQLModel (stdlib `sqlite3` is
enough), pandas/scipy/seaborn (metrics in §8 are small enough for plain
`statistics`/hand-rolled Spearman+bootstrap — resolves the open question
left in the CLI plan), HTMX/any JS framework (plain forms + a few lines of
polling JS are enough for one local user).

## 6. Settings additions (additive only, `config/settings.py`)

```python
PRESTUDY_DB_PATH: Path = Path("var/prestudy/prestudy.db")
PRESTUDY_UPLOAD_DIR: Path = Path("var/prestudy/uploads")
PRESTUDY_NORMALIZED_DIR: Path = Path("var/prestudy/normalized")
PRESTUDY_MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024       # 15 MB
PRESTUDY_MAX_DECODED_PIXELS: int = 40_000_000            # decompression-bomb guard
```

No directories are created at import time or at `Settings()` construction
— creation happens once, in the app's `lifespan`, consistent with the
existing "settings/logging have no import-time side effects" rule
(`config/settings.py` docstring, `utils/log_factory.py`).

## 7. Data model (SQLite, stdlib `sqlite3`)

Why raw `sqlite3` over SQLAlchemy/SQLModel: at ~3–4k rows total, the
query surface is small and known up front (insert a batch+attempts,
update attempt status, filter/aggregate for charts) — parameterized SQL
plus a `schema_version` table covers it with zero extra dependency
surface, and keeps the "framework-free utilities" discipline already used
for logging.

```sql
CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL CHECK (source IN ('fruit_v1', 'manual_upload')),
    image_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    specimen_id TEXT,              -- NULL for manual_upload
    fruit_type TEXT,
    condition TEXT,                -- c1/c2/c3, NULL for manual_upload
    split TEXT,                    -- dev/test, NULL for manual_upload
    whole_g REAL, edible_g REAL, length_cm REAL, max_diameter_cm REAL,
    bls_code TEXT, kcal_ref REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    manifest_sha256 TEXT,          -- NULL for ad-hoc/manual experiments
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE run_batches (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    sample_id INTEGER NOT NULL REFERENCES samples(id),
    strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3','B0','B1')),
    model TEXT,                     -- NULL for B0/B1 (no provider call)
    provider_pin TEXT,
    quantizations TEXT,              -- JSON array as text
    temperature REAL,
    resolution_px INTEGER,
    repeats INTEGER NOT NULL,
    prompt_hash TEXT,
    protocol_compliant INTEGER NOT NULL,  -- 1 iff temp=0, res in {1024,768}, repeats=3
    status TEXT NOT NULL CHECK (status IN
        ('queued','running','completed','failed','interrupted')),
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE run_attempts (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES run_batches(id),
    repeat_index INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','ok','schema_invalid','error')),
    raw_response TEXT,               -- JSON text, model's parsed dict or null
    edible_g_pred REAL,
    kcal_pred REAL,
    length_cm_pred REAL, max_diameter_cm_pred REAL,   -- S2 only
    schema_valid INTEGER,
    latency_ms REAL,
    attempts INTEGER,                -- OpenRouterClient's own retry count
    prompt_tokens INTEGER, completion_tokens INTEGER,
    cost_usd REAL, cost_is_estimate INTEGER,
    provider_used TEXT, model_used TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE chart_exports (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,              -- e.g. 'predicted_vs_true', 'mape_by_model'
    filter_spec TEXT NOT NULL,       -- JSON: experiment, split, strategy, models, etc.
    attempt_ids TEXT NOT NULL,       -- JSON array of run_attempts.id used, frozen at export time
    bootstrap_seed INTEGER,
    svg_path TEXT NOT NULL, png_path TEXT NOT NULL,
    svg_sha256 TEXT NOT NULL, png_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Connection setup in `bench/db.py`: `PRAGMA journal_mode=WAL`,
`PRAGMA foreign_keys=ON`, `PRAGMA busy_timeout=5000`. A batch + its N
queued `run_attempts` rows are inserted in one transaction before any
OpenRouter call is made (so "submitted" is immediately visible and
queryable even before execution starts); the transaction is never held
open across a network call — each attempt's result is written in its own
short transaction as it completes.

### `protocol_compliant` (resolves the architect's escalation)

Per your decision: **allow any value, flag non-protocol runs.** A batch is
`protocol_compliant = 1` iff `temperature == 0 AND resolution_px IN (1024, 768)
AND repeats == 3`; computed once at batch creation in `runs.py`, stored
(not recomputed ad hoc) so chart queries stay simple. `metrics.py`'s
decision-rule aggregations (MAPE/Spearman ρ/slope/bias — the numbers that
go in the paper, concept §Appendix A.6) filter to `protocol_compliant = 1`
**by default**; `history.html` and `charts.html` both expose an explicit
"include exploratory runs" toggle that lifts the filter, and every chart
export's `filter_spec` records whether that toggle was on — so a figure
pulled into a paper is traceably either protocol-only or mixed, never
ambiguous.

## 8. Component responsibilities

### 8.1 `imaging/preprocess.py`

Same contract as the CLI plan's §5.1: `normalize_image(path_or_bytes,
max_px) -> bytes` — Pillow decode (enforcing `PRESTUDY_MAX_DECODED_PIXELS`
before full decode to guard decompression bombs), resize longest edge to
`max_px`, strip EXIF, re-encode JPEG ~q90. Pure function; same unit tests
as before. Also used for the derived file written to
`var/prestudy/normalized/`.

### 8.2 `observation/{schema.py,adapter.py,prompts/}`

Unchanged in responsibility from the CLI plan's §5.2/§5.3: per-strategy
strict JSON Schema, versioned prompt files loaded as package data, a
thin `observe(client, image_bytes, strategy, model, routing_mode,
provider_pin, quantizations, temperature) -> ObservationResult` wrapper
around `OpenRouterClient.complete`. Called from `bench/runs.py` instead of
a CLI command — no change to the adapter itself.

### 8.3 `domain/calculator.py`

Unchanged from the CLI plan's §5.4 — pure S2 formulas, property-tested.

### 8.4 `bench/fruit_reference.py`

Unchanged from the CLI plan's §5.5 — tiny CSV-backed lookup, not a real
`reference/` pipeline.

### 8.5 `bench/dataset.py`

New: `import_manifest(path: Path) -> ImportSummary` — reads
`bench/sets/fruit-v1/manifest.jsonl`, validates each line via a Pydantic
`ManifestEntry` (same shape as the CLI plan's §4.1), upserts one `samples`
row per entry keyed on `specimen_id + condition` (re-import is
idempotent), records the manifest's sha256 on an `experiments` row so a
chart's provenance can point at an exact dataset version. Run once via a
small `python -m food_vision.bench.dataset import bench/sets/fruit-v1/manifest.jsonl`
script (not a web form — this is a one-time data-loading step, not a
per-run action).

### 8.6 `bench/runs.py`

- `create_batch(db, sample_id, strategy, model, provider_pin,
  quantizations, temperature, resolution_px, repeats, experiment_id) ->
  int` — validates params (model against `FOOD_VISION_MODELS` via
  `get_settings()`; provider_pin+quantizations required for S1/S2/S3,
  forbidden/ignored for B0/B1), computes `protocol_compliant`, inserts the
  batch + N queued attempts in one transaction, returns `batch_id`.
- `execute_batch(db, client, batch_id) -> None` — the function passed to
  FastAPI's `BackgroundTasks`. Loads the batch + sample, loops repeats:
  - B0: dev-split per-type mean from `samples` (no provider call).
  - B1: `domain.calculator` on the sample's **true** `length_cm`/
    `max_diameter_cm` (no provider call).
  - S1/S2/S3: `imaging.preprocess` → `observation.adapter.observe` →
    (S2 only) `domain.calculator` → `bench.fruit_reference` for kcal.
  - Writes each attempt's result immediately (own short transaction); on
    exception, writes `status='error'` with the message and continues to
    the next repeat rather than aborting the batch.
  - Marks the batch `completed` (or `failed` if every attempt errored).
- On app startup (`lifespan`), any batch left `queued`/`running` from a
  previous process is marked `interrupted` — makes crash/restart state
  visible instead of silently stuck "running" forever.

### 8.7 `bench/metrics.py`

Same functions as the CLI plan's §5.7 (`mape`, `spearman_rho_and_slope`,
`signed_bias`, `repeat_cv`, `schema_valid_rate`, `bootstrap_ci`), now
reading from `run_attempts`/`samples` via `bench/db.py` queries instead of
a list of `RunRecord` loaded from JSON files. Implemented with plain
`statistics` + a small hand-rolled rank-correlation and percentile
bootstrap (no `pandas`/`scipy` — confirms and resolves the CLI plan's
open question now that the data path is DB rows, not a JSON blob begging
for `pandas.DataFrame`).

### 8.8 `bench/charts.py`

- One function per chart kind, each: runs a parameterized query against
  `bench/db.py` for the given `filter_spec`, builds the plot with
  `matplotlib` (`matplotlib.use("Agg")` set once at module import — no
  display backend needed), saves both `.svg` and 600-DPI `.png` to
  `bench/reports/prestudy-web/charts/chart-{id}-{kind}.{ext}`, inserts a
  `chart_exports` row recording the exact `attempt_ids` used, the
  `filter_spec` (including the exploratory-runs toggle state), and file
  hashes — so a chart is **immutable** once exported (re-running the same
  filter later creates a new `chart_exports` row/id rather than
  overwriting, since new runs may have been added since).
- Chart kinds (same set identified for the CLI plan's visualization need):
  `predicted_vs_true` (scatter, y=x line, per model/strategy, colored by
  fruit type), `mape_by_model` (bar + bootstrap CI, protocol-only by
  default), `latency_distribution` (box/violin per model),
  `cost_per_1000` (bar per model), `repeat_cv_by_model` (bar),
  `condition_effect` (grouped bar, C2 vs C3 MAPE per model),
  `schema_valid_rate` (bar), `signed_bias_by_type` (grouped bar/heatmap).

### 8.9 `bench/web/app.py`

- `create_app() -> FastAPI` factory (not a module-level singleton — keeps
  it testable with per-test temp DB/dirs).
- `lifespan`: calls `configure_logging()`, `get_settings()`, creates
  `PRESTUDY_UPLOAD_DIR`/`PRESTUDY_NORMALIZED_DIR`/DB parent dir if
  missing, opens the `sqlite3` connection, runs schema migration/version
  check, marks stale `queued`/`running` batches `interrupted`, stores
  the connection and a `get_default_provider()`-built `OpenRouterClient`
  on `app.state`.
- Routes under `/prestudy`:
  - `GET /prestudy/` → `new_run.html` (upload form + param form; model
    `<select>` populated from `FOOD_VISION_MODELS`).
  - `POST /prestudy/samples` → multipart upload, validates
    content-type/size, decodes via `imaging.preprocess` (rejecting
    decompression bombs before full decode), stores raw+normalized files
    under content-hashed names (never the user's filename, directly
    addressing the architect's path-safety risk), inserts a `samples` row
    with `source='manual_upload'`, redirects to the param form for that
    sample.
  - `POST /prestudy/batches` → validates params via `runs.create_batch`,
    schedules `runs.execute_batch` as a `BackgroundTasks` job, redirects
    to `batch_status.html?batch_id=...`.
  - `GET /prestudy/batches/{id}` → status + attempts so far (HTML; a
    small inline `fetch`-poll every ~2s re-renders until `status` is
    terminal — no JS framework).
  - `GET /prestudy/history` → `history.html`, filterable table over
    `run_batches`/`run_attempts` (by experiment, split, strategy, model,
    protocol-compliance), with a link to generate a chart from the
    current filter.
  - `GET /prestudy/charts` + `POST /prestudy/charts` → `charts.html`
    gallery of past `chart_exports` + a form to generate a new one from a
    `history.html` filter; served images are static files under
    `bench/reports/prestudy-web/charts/`.
- Dependencies injected via `Depends`: `get_db()`, `get_provider()`
  (overridden with a fake in tests), `get_settings()` (already lazy/
  cached).

## 9. Testing plan

All against a temp SQLite file per test (`tmp_path`), FastAPI
`TestClient`, and a fake `ModelProvider` (the existing `Protocol` in
`proxy/openrouter.py`) — **no live OpenRouter calls**, consistent with the
existing test suite.

- `tests/unit/test_bench_db.py` — schema creation/versioning, insert/query
  round-trips for each table, WAL/foreign-key pragmas applied.
- `tests/unit/test_bench_dataset.py` — manifest import idempotency,
  invalid-entry rejection, experiment manifest-hash recorded.
- `tests/unit/test_bench_runs.py` — `create_batch` validation (rejects
  unknown model, missing provider/quantization for S1–S3, rejects
  provider/quantization for B0/B1), `protocol_compliant` computed
  correctly for edge cases (e.g. temp=0.0 vs 0, resolution 900 → false),
  `execute_batch` with the fake provider: happy path, schema-invalid
  response, provider exception (attempt marked `error`, batch continues),
  B0/B1 make zero fake-provider calls.
- `tests/unit/test_bench_metrics.py` — MAPE/bias zero on perfect
  predictions, CV zero on identical repeats, bootstrap CI contains the
  point estimate, Spearman ρ against a hand-computed fixture.
- `tests/unit/test_bench_charts.py` — each chart function against fixture
  DB rows produces a non-empty `.svg` (contains `<svg`) and a valid PNG
  (correct magic bytes), inserts exactly one `chart_exports` row with the
  right `attempt_ids`; **no pixel comparison**.
- `tests/unit/test_prestudy_web.py` — route-level: GET form pages render;
  POST upload rejects oversized/wrong-mimetype files and sanitizes
  filenames; POST batch rejects bad params (422) and accepts valid ones
  (batch+attempts rows exist before the response returns); status/history
  pages reflect DB state; startup `interrupted`-marking covered by
  seeding a `running` batch before building the app.
- `imaging/preprocess`, `observation/schema`+`adapter`, `domain/calculator`
  tests: same as the CLI plan's §8 — unaffected by the interface change.

## 10. Sequencing

1. `data/fruit_kcal_100g.csv` + `bench/fruit_reference.py` (unblocks
   everything, no provider dependency).
2. `imaging/preprocess.py` + tests.
3. `observation/{schema.py,prompts/*.md,adapter.py}` + tests (mocked
   `OpenRouterClient`).
4. `domain/calculator.py` + property tests.
5. `bench/db.py` (schema + queries) + tests.
6. `bench/dataset.py` + manifest import + tests.
7. `bench/runs.py` (batch lifecycle, baselines) + tests with fake
   provider.
8. `bench/metrics.py` + tests.
9. `bench/charts.py` + tests.
10. `bench/web/app.py` + templates/static + route tests.
11. Settings additions (§6), `pyproject.toml` dependency additions (§5),
    `.gitignore` entry for `var/`.
12. Capture `fruit-v1` images + run `bench.dataset import` (can happen any
    time after step 6).
13. Dev-split exploration through the UI (prompt iteration, parameter
    exploration — flagged non-protocol as appropriate); freeze prompts;
    one protocol-compliant test-split batch per final candidate model;
    export the decision-rule charts for the paper from `history.html`'s
    protocol-only filter.

## 11. Open items deferred, not blocking

- Exact provider-pin/quantization choices per candidate model: left as
  free-text fields validated only for non-emptiness when required by
  strategy — the concrete OpenRouter provider/quantization labels for the
  6–8 candidates are an operational detail decided when the sweep runs,
  not a code structure question.
- Retention/backup policy for `var/prestudy/` (images, DB, raw responses):
  default is "keep everything, no auto-deletion" since this is a personal
  research tool with modest data volume; revisit only if storage becomes
  a real concern.
- `uv_build` package-data inclusion of `templates/`/`static/`/`prompts/`:
  expected to work by default (everything under `src/food_vision/` is
  packaged); verify with a `uv build` dry-run once those files exist
  rather than pre-emptively configuring `[tool.uv_build]`.
