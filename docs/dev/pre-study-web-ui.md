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

This revision folds in a pre-implementation review (see §12) that found
several gaps between what `OpenRouterClient` could actually do and what
this plan assumed it could do, plus validity risks specific to running a
benchmark through an interactive UI instead of a frozen CLI sweep. Fixes
are called out inline where they change a decision already made, and
summarized in §12.

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
    ├── db.py                       # sqlite3 connection-per-call helpers, schema, queries
    ├── dataset.py                  # fruit-v1 manifest import -> samples rows
    ├── runs.py                     # batch lifecycle, repeat execution, baselines
    ├── fruit_reference.py          # tiny kcal/density/edible-ratio lookup
    ├── calculator_fits.py          # stores/looks up fitted S2 constants per experiment
    ├── metrics.py                  # MAPE, Spearman rho+slope, bias, CV, bootstrap CI
    ├── charts.py                   # matplotlib SVG+PNG export from stored history
    └── web/
        ├── __init__.py
        ├── app.py                  # create_app(), lifespan, routes, DI
        ├── templates/
        │   ├── base.html
        │   ├── new_run.html         # upload + param form
        │   ├── batch_status.html    # polling page for a running batch
        │   ├── history.html         # queryable run list/filter + test-split counters
        │   └── charts.html          # chart gallery + download links
        └── static/
            └── prestudy.css

data/
└── fruit_kcal_100g.csv             # bls_code -> kcal_100g, density_g_cm3, edible_ratio

var/prestudy/                        # gitignored: runtime state, not source
├── prestudy.db                      # sqlite3 file (WAL mode)
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
enough), pandas/scipy/seaborn (metrics in §8.7 are small enough for plain
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

This plan now also relies on `OpenRouterClient.complete()` accepting
per-call `temperature`, `seed`, and a `routing_policy: RoutingPolicy`
(`provider_pin` required for `RoutingMode.BENCHMARK`, `quantizations`/
`zdr`/`data_collection` optional overrides) — **already implemented** in
`proxy/openrouter.py` as of this revision; no proxy changes remain
outstanding for this plan to depend on. `CompletionResult` additionally
carries `effective_routing_policy`, `temperature`, and `seed` so a caller
can record exactly what was sent, not just what it asked for.

## 7. Data model (SQLite, stdlib `sqlite3`)

Why raw `sqlite3` over SQLAlchemy/SQLModel: at ~3–4k rows total, the
query surface is small and known up front (insert a batch+attempts,
update attempt status, filter/aggregate for charts) — parameterized SQL
plus a `schema_version` table covers it with zero extra dependency
surface, and keeps the "framework-free utilities" discipline already used
for logging.

### 7.1 Connection handling (fix: no shared connection across threads)

**Do not** open one `sqlite3.connect()` at startup and store it on
`app.state`. `sqlite3` connections default to `check_same_thread=True`,
and FastAPI's `BackgroundTasks` (used for `runs.execute_batch`, §8.6) run
in a thread-pool thread different from the request thread that opened the
connection — sharing one connection across that boundary raises
`sqlite3.ProgrammingError` at the first background-task query.

Instead, `bench/db.py` provides a connection-per-call context manager:

```python
@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

- The FastAPI `get_db()` dependency opens one connection per **request**
  via this context manager (closed when the request finishes).
- `runs.execute_batch` (running inside `BackgroundTasks`, its own thread)
  opens its **own** connection via the same context manager — never reuses
  a connection handed to it from the request that scheduled it.
- WAL mode is what makes this safe and fast: concurrent readers (the
  request thread polling batch status) don't block the writer (the
  background task), and multiple short writer transactions from different
  connections serialize automatically with `busy_timeout` absorbing brief
  contention instead of raising `database is locked`.
- A batch + its N queued `run_attempts` rows are inserted in one
  transaction, in the **request's** connection, before the background task
  is scheduled — so "submitted" is immediately visible and queryable even
  before execution starts. Each attempt's result is then written in its
  own short transaction, on the **background task's own connection**, as
  it completes.

### 7.2 Schema

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

-- Fix #6: protocol_compliant must also account for the prompt, not just
-- temperature/resolution/repeats. A test-split batch only counts as
-- protocol-compliant if its prompt_hash was frozen *before* that batch
-- ran. Freezing is an explicit action (e.g. a small CLI/admin route),
-- never implicit from "I ran it on dev enough times".
CREATE TABLE frozen_prompts (
    id INTEGER PRIMARY KEY,
    strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3')),
    prompt_hash TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    UNIQUE (strategy, prompt_hash)
);

-- Fix #9: fitted S2/B1 constants (banana volume coefficient `a`,
-- per-type density, edible_ratio) are fit data, not code constants —
-- store them keyed by experiment so a batch records exactly which fit it
-- used, and re-fitting doesn't silently change the meaning of old rows.
CREATE TABLE calculator_fits (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    fruit_type TEXT NOT NULL,
    param_name TEXT NOT NULL,       -- 'banana_a' | 'density_g_cm3' | 'edible_ratio'
    value REAL NOT NULL,
    fitted_on_split TEXT NOT NULL CHECK (fitted_on_split = 'dev'),
    fitted_at TEXT NOT NULL,
    notes TEXT,
    UNIQUE (experiment_id, fruit_type, param_name)
);

-- Fix #8: S3's typical-portion prior (p10/p50/p90 per fruit type) must be
-- derived from dev specimens (or an external table), never from test
-- specimens, and the exact values used must be traceable per experiment.
CREATE TABLE prior_values (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    fruit_type TEXT NOT NULL,
    p10_g REAL NOT NULL, p50_g REAL NOT NULL, p90_g REAL NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('dev_specimens', 'external_table')),
    derived_at TEXT NOT NULL,
    UNIQUE (experiment_id, fruit_type)
);

CREATE TABLE run_batches (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    sample_id INTEGER NOT NULL REFERENCES samples(id),
    strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3','B0','B1')),
    model TEXT,                     -- NULL for B0/B1 (no provider call)
    -- Fix #4: the full effective RoutingPolicy (provider_pin REQUIRED for
    -- S1/S2/S3; quantizations OPTIONAL — several candidates, e.g. Gemini/
    -- GPT/Claude on OpenRouter, carry no quantization label at all, so
    -- requiring one would just exclude them; zdr/data_collection default
    -- to Settings unless explicitly overridden), stored verbatim from
    -- CompletionResult.effective_routing_policy so a failing provider/
    -- privacy-policy combination is one row to inspect, not an env-file
    -- edit and restart.
    routing_policy TEXT,             -- JSON: {order, allow_fallbacks, quantizations?, zdr, data_collection}
    temperature REAL,
    seed INTEGER,
    resolution_px INTEGER,
    repeats INTEGER NOT NULL,
    prompt_hash TEXT,
    calculator_fit_experiment_id INTEGER REFERENCES experiments(id),  -- which fit set S2/B1 used
    -- Fix #7: a test-split batch requires this explicit flag, set only
    -- via a confirmation checkbox in the UI — never a default, never
    -- inferred from split alone, so burning the frozen set is never one
    -- accidental click.
    is_frozen_run INTEGER NOT NULL DEFAULT 0,
    protocol_compliant INTEGER NOT NULL,
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
    filter_spec TEXT NOT NULL,       -- JSON: experiment, split, strategy, models, exploratory-toggle, etc.
    attempt_ids TEXT NOT NULL,       -- JSON array of run_attempts.id used, frozen at export time
    bootstrap_seed INTEGER,
    svg_path TEXT NOT NULL, png_path TEXT NOT NULL,
    svg_sha256 TEXT NOT NULL, png_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

### 7.3 `protocol_compliant` (fix #6 extends the earlier escalation answer)

Per your earlier decision: **allow any value, flag non-protocol runs.** A
batch is `protocol_compliant = 1` iff:

```text
temperature == 0
AND resolution_px IN (1024, 768)
AND repeats == 3
AND (sample.split != 'test' OR (strategy IN ('B0','B1')) OR
     EXISTS frozen_prompts WHERE strategy = batch.strategy
                             AND prompt_hash = batch.prompt_hash)
```

Computed once at batch creation in `runs.py`, stored (not recomputed ad
hoc) so chart queries stay simple. Without the prompt clause, a test-split
run made with a prompt tweaked *after* dev-split iteration would still
read as protocol-compliant — the one case the earlier check missed,
because temperature/resolution/repeats were satisfied but the prompt
itself was never frozen. `metrics.py`'s decision-rule aggregations
(MAPE/Spearman ρ/slope/bias — the numbers that go in the paper, concept
§Appendix A.6) filter to `protocol_compliant = 1` **by default**;
`history.html` and `charts.html` both expose an explicit "include
exploratory runs" toggle that lifts the filter, and every chart export's
`filter_spec` records whether that toggle was on — so a figure pulled
into a paper is traceably either protocol-only or mixed, never ambiguous.

## 8. Component responsibilities

### 8.1 `imaging/preprocess.py`

Same contract as the CLI plan's §5.1: `normalize_image(path_or_bytes,
max_px) -> bytes` — Pillow decode (enforcing `PRESTUDY_MAX_DECODED_PIXELS`
before full decode to guard decompression bombs), resize longest edge to
`max_px`, strip EXIF, re-encode JPEG ~q90. Pure function; same unit tests
as before. Also used for the derived file written to
`var/prestudy/normalized/`.

### 8.2 `observation/{schema.py,adapter.py,prompts/}`

Per-strategy strict JSON Schema, versioned prompt files loaded as package
data, a thin wrapper around `OpenRouterClient.complete`:

```python
def observe(
    client: ModelProvider,
    *,
    image_bytes: bytes,
    strategy: Strategy,
    model: str,
    routing_mode: RoutingMode,
    routing_policy: RoutingPolicy,     # provider_pin required; quantizations optional
    temperature: float,
    seed: int | None,
) -> ObservationResult: ...
```

`routing_policy`/`temperature`/`seed` are **passed through from the
batch's stored params** (`run_batches.routing_policy`/`temperature`/
`seed`), not re-read from `Settings` here — this is what makes per-batch
exploration outside the frozen protocol possible at all (fix #1: before
this revision, `OpenRouterClient.complete()` only read temperature/seed
from process-wide `Settings`, which made a per-run `temperature` column in
`run_batches` meaningless; `complete()` now accepts both as optional
per-call overrides). Called from `bench/runs.py` instead of a CLI command.

### 8.3 `domain/calculator.py`

Pure S2 formulas, property-tested — unchanged signatures from the CLI
plan's §5.4, but the constants they're called with now come from
`bench/calculator_fits.py` (§8.4a) keyed by `experiment_id`, not from
module-level literals.

### 8.4 `bench/fruit_reference.py`

Unchanged from the CLI plan's §5.5 — tiny CSV-backed lookup
(`kcal_100g`), not a real `reference/` pipeline.

### 8.4a `bench/calculator_fits.py` (new, fix #9)

- `record_fit(db, experiment_id, fruit_type, param_name, value, notes=None) -> None`
  — inserts/updates a `calculator_fits` row; raises if `fitted_on_split`
  would be anything other than `'dev'` (the column's `CHECK` constraint is
  the hard backstop; this function is the only intended writer).
- `get_fits(db, experiment_id, fruit_type) -> dict[str, float]` — returns
  `{"banana_a": ..., "density_g_cm3": ..., "edible_ratio": ...}` for the
  given experiment/type, raising a clear error if a required param is
  missing rather than silently falling back to a hardcoded guess.
- `runs.py` passes `get_fits(db, batch.calculator_fit_experiment_id, sample.fruit_type)`
  into `domain.calculator` for S2/B1 batches. A batch created before any
  fit has been recorded for its experiment fails fast at creation time
  with a message pointing at which fit is missing, rather than producing
  a silently wrong S2/B1 estimate.

### 8.4b `bench/prior_values.py` (new, fix #8)

- `record_priors(db, experiment_id, fruit_type, p10_g, p50_g, p90_g, source) -> None`
  — `source` must be `'dev_specimens'` (computed from dev-split
  `samples` rows for that type) or `'external_table'` (e.g. BLS/NVS II
  portion data copied in by hand); never computed from test-split rows —
  enforced by the function only ever querying `samples WHERE split =
  'dev'` when `source='dev_specimens'`.
- `get_priors(db, experiment_id, fruit_type) -> PriorValues` — read by the
  S3 prompt-building step in `observation/adapter.py` (via `runs.py`) so
  the exact p10/p50/p90 used in a given S3 batch is traceable back to one
  `prior_values` row, not a hardcoded line in a prompt file.

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

- `create_batch(db, *, sample_id, strategy, model, routing_policy,
  temperature, seed, resolution_px, repeats, experiment_id,
  is_frozen_run=False) -> int`:
  - Validates `model` against `FOOD_VISION_MODELS` via `get_settings()`.
  - For S1/S2/S3: `routing_policy.provider_pin` is **required**;
    `routing_policy.quantizations` is **optional** (fix #3 — previously
    both were required, which made benchmark routing unusable for
    candidates that don't publish a quantization label at all).
  - For B0/B1: no provider call is made, so `model`/`routing_policy` are
    ignored if supplied (and the UI doesn't show those fields for these
    strategies).
  - For S2/B1: resolves `calculator_fit_experiment_id` (defaults to
    `experiment_id` unless the caller points at a different experiment's
    fit set) and calls `calculator_fits.get_fits` eagerly — fails fast
    here, not mid-run, if the fit is missing.
  - For S3: resolves and validates a `prior_values` row exists for the
    sample's `fruit_type` under `experiment_id` — same fail-fast reasoning.
  - **Fix #7**: if `sample.split == 'test'` and `is_frozen_run` is not
    explicitly `True`, raises `ValueError` — the web form only sets this
    from an explicit, unchecked-by-default "I confirm this is a frozen
    test-split run" checkbox (§8.9); there is no default path that spends
    a test-split sample.
  - Computes `protocol_compliant` per §7.3 (including the frozen-prompt
    check), inserts the batch + N queued `run_attempts` rows in one
    transaction, returns `batch_id`.
- `execute_batch(db_path, client, batch_id) -> None` — the function passed
  to FastAPI's `BackgroundTasks`; opens its **own** `bench.db.connect()`
  connection (§7.1 — never reuses the request's connection). Loads the
  batch + sample, loops repeats:
  - B0: dev-split per-type mean from `samples` (no provider call).
  - B1: `domain.calculator` (with `calculator_fits.get_fits`) on the
    sample's **true** `length_cm`/`max_diameter_cm` (no provider call).
  - S1/S2/S3: `imaging.preprocess` → `observation.adapter.observe` (passing
    the batch's `routing_policy`/`temperature`/`seed`) → (S2 only)
    `domain.calculator` → `bench.fruit_reference` for kcal.
  - Writes each attempt's result immediately (own short transaction),
    including `effective_routing_policy` from `CompletionResult` so what
    was *actually* sent is recorded even if it differs from what was
    requested (e.g. a provider substitution); on exception, writes
    `status='error'` with the message and continues to the next repeat
    rather than aborting the batch.
  - Marks the batch `completed` (or `failed` if every attempt errored).
- On app startup (`lifespan`), any batch left `queued`/`running` from a
  previous process is marked `interrupted` — makes crash/restart state
  visible instead of silently stuck "running" forever.

### 8.7 `bench/metrics.py`

Same functions as the CLI plan's §5.7 (`mape`, `spearman_rho_and_slope`,
`signed_bias`, `repeat_cv`, `schema_valid_rate`, `bootstrap_ci`), reading
from `run_attempts`/`samples` via `bench/db.py` queries. Implemented with
plain `statistics` + a small hand-rolled rank-correlation and percentile
bootstrap (no `pandas`/`scipy`). All decision-rule aggregations default to
`protocol_compliant = 1` rows only (§7.3), with an explicit parameter to
include exploratory rows when the caller (a chart function, or
`history.html`) asks for it.

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
- Chart kinds: `predicted_vs_true` (scatter, y=x line, per model/strategy,
  colored by fruit type), `mape_by_model` (bar + bootstrap CI,
  protocol-only by default), `latency_distribution` (box/violin per
  model), `cost_per_1000` (bar per model), `repeat_cv_by_model` (bar),
  `condition_effect` (grouped bar, C2 vs C3 MAPE per model),
  `schema_valid_rate` (bar), `signed_bias_by_type` (grouped bar/heatmap).
- **Methodology caveat to render on `latency_distribution`/resolution-
  comparison charts** (hygiene item, see §12): providers apply their own
  image tiling before inference (e.g. Gemini tiles around 768px, OpenAI
  around 512px tiles), so the 768-vs-1024px comparison this study runs
  (concept §A.5) is partly confounded by provider-side resizing that
  happens regardless of what we upload. Keep the comparison — it's still
  informative about end-to-end behavior — but the chart/report must say
  so explicitly rather than presenting it as a clean controlled variable.

### 8.9 `bench/web/app.py`

- `create_app() -> FastAPI` factory (not a module-level singleton — keeps
  it testable with per-test temp DB/dirs).
- `lifespan`: calls `configure_logging()`, `get_settings()`, creates
  `PRESTUDY_UPLOAD_DIR`/`PRESTUDY_NORMALIZED_DIR`/DB parent dir if
  missing, runs schema migration/version check (opening and closing its
  own short-lived connection via `bench.db.connect()` — **not** one held
  on `app.state`, per §7.1), marks stale `queued`/`running` batches
  `interrupted`, and stores only the `db_path` and a
  `get_default_provider()`-built `OpenRouterClient` on `app.state` (both
  stateless/reusable across threads; the connection itself never is).
- Routes under `/prestudy`:
  - `GET /prestudy/` → `new_run.html` (upload form + param form; model
    `<select>` populated from `FOOD_VISION_MODELS`; provider-pin is a
    required text field, quantization an optional one — fix #3; split is
    read from the selected sample, and if it's `'test'` the form renders
    an unchecked "I confirm this is a frozen test-split run" checkbox
    that must be ticked to submit — fix #7).
  - `POST /prestudy/samples` → multipart upload, validates
    content-type/size, decodes via `imaging.preprocess` (rejecting
    decompression bombs before full decode), stores raw+normalized files
    under content-hashed names (never the user's filename), inserts a
    `samples` row with `source='manual_upload'`, redirects to the param
    form for that sample.
  - `POST /prestudy/batches` → validates params via `runs.create_batch`
    (which itself enforces the frozen-run checkbox and fit/prior
    prerequisites), schedules `runs.execute_batch` as a `BackgroundTasks`
    job, redirects to `batch_status.html?batch_id=...`.
  - `GET /prestudy/batches/{id}` → status + attempts so far (HTML; a
    small inline `fetch`-poll every ~2s re-renders until `status` is
    terminal — no JS framework).
  - `GET /prestudy/history` → `history.html`, filterable table over
    `run_batches`/`run_attempts` (by experiment, split, strategy, model,
    protocol-compliance), **plus a counter of test-split batches per
    (model, strategy)** (fix #7 — makes "did I already burn the test set
    for this model/strategy" auditable at a glance instead of having to
    scroll history), with a link to generate a chart from the current
    filter.
  - `GET /prestudy/charts` + `POST /prestudy/charts` → `charts.html`
    gallery of past `chart_exports` + a form to generate a new one from a
    `history.html` filter; served images are static files under
    `bench/reports/prestudy-web/charts/`.
- Dependencies injected via `Depends`: `get_db()` (opens a fresh
  connection per request, §7.1), `get_provider()` (overridden with a fake
  in tests), `get_settings()` (already lazy/cached).

## 9. Testing plan

All against a temp SQLite file per test (`tmp_path`), FastAPI
`TestClient`, and a fake `ModelProvider` (the existing `Protocol` in
`proxy/openrouter.py`) — **no live OpenRouter calls**, consistent with the
existing test suite.

- `tests/unit/test_bench_db.py` — schema creation/versioning, insert/query
  round-trips for each table, WAL/foreign-key pragmas applied, **a
  connection opened on one thread and used from another does not raise**
  (regression test for fix #5 — simulates the request/background-task
  split directly).
- `tests/unit/test_bench_dataset.py` — manifest import idempotency,
  invalid-entry rejection, experiment manifest-hash recorded.
- `tests/unit/test_bench_calculator_fits.py` — `record_fit`/`get_fits`
  round-trip; `get_fits` raises a clear error when a required param is
  missing; rejects a non-`'dev'` `fitted_on_split`.
- `tests/unit/test_bench_prior_values.py` — `record_priors` with
  `source='dev_specimens'` only ever reads `samples WHERE split='dev'`
  (assert via a fixture with both dev and test rows for the same type);
  `get_priors` round-trip.
- `tests/unit/test_bench_runs.py` — `create_batch` validation: rejects
  unknown model; requires `routing_policy.provider_pin` for S1–S3 but
  **accepts a missing `quantizations`** (regression test for fix #3);
  rejects/ignores routing params for B0/B1; **rejects a test-split batch
  when `is_frozen_run` is not `True`, accepts it when `True`** (fix #7);
  fails fast when a required `calculator_fits`/`prior_values` row is
  missing; `protocol_compliant` computed correctly including the
  frozen-prompt clause (fix #6: a test-split batch with an *unfrozen*
  prompt_hash is not compliant even with temp=0/res=1024/repeats=3;
  becomes compliant once the same `(strategy, prompt_hash)` is inserted
  into `frozen_prompts`). `execute_batch` with the fake provider: happy
  path records `effective_routing_policy`/`temperature`/`seed` from
  `CompletionResult`, schema-invalid response, provider exception
  (attempt marked `error`, batch continues), B0/B1 make zero
  fake-provider calls.
- `tests/unit/test_bench_metrics.py` — MAPE/bias zero on perfect
  predictions, CV zero on identical repeats, bootstrap CI contains the
  point estimate, Spearman ρ against a hand-computed fixture,
  protocol-only filtering excludes non-compliant rows by default.
- `tests/unit/test_bench_charts.py` — each chart function against fixture
  DB rows produces a non-empty `.svg` (contains `<svg`) and a valid PNG
  (correct magic bytes), inserts exactly one `chart_exports` row with the
  right `attempt_ids`; **no pixel comparison**.
- `tests/unit/test_prestudy_web.py` — route-level: GET form pages render,
  including the frozen-run checkbox only appearing for `split='test'`
  samples; POST upload rejects oversized/wrong-mimetype files and
  sanitizes filenames; POST batch rejects bad params (422), rejects an
  unchecked frozen-run attempt on a test sample, and accepts valid ones
  (batch+attempts rows exist before the response returns); status/history
  pages reflect DB state including the test-split counter; startup
  `interrupted`-marking covered by seeding a `running` batch before
  building the app; a background task and its triggering request use
  independent DB connections without error (integration-level version of
  the fix #5 regression test).
- `imaging/preprocess`, `observation/schema`+`adapter`, `domain/calculator`
  tests: same as the CLI plan's §8 — unaffected by the interface change,
  except `observation/adapter` tests now assert `temperature`/`seed`/
  `routing_policy` are forwarded from the call, not read from `Settings`.
- `proxy/openrouter.py`'s own tests (already updated in this revision):
  `RoutingPolicy` optional-quantizations behavior, per-call
  temperature/seed override, `extra_body.usage.include=True` always sent,
  `CompletionResult.effective_routing_policy` populated.

## 10. Sequencing

1. `data/fruit_kcal_100g.csv` + `bench/fruit_reference.py` (unblocks
   everything, no provider dependency).
2. `imaging/preprocess.py` + tests.
3. `observation/{schema.py,prompts/*.md,adapter.py}` + tests (mocked
   `OpenRouterClient`, asserting `temperature`/`seed`/`routing_policy`
   pass-through).
4. `domain/calculator.py` + property tests.
5. `bench/db.py` (connection-per-call helper, schema, queries) + tests,
   including the cross-thread-connection regression test.
6. `bench/dataset.py` + manifest import + tests.
7. `bench/calculator_fits.py` + `bench/prior_values.py` + tests.
8. `bench/runs.py` (batch lifecycle, baselines, frozen-run gate,
   frozen-prompt-aware `protocol_compliant`) + tests with fake provider.
9. `bench/metrics.py` + tests.
10. `bench/charts.py` + tests (including the resizing-confound caveat
    text on the relevant chart).
11. `bench/web/app.py` + templates/static + route tests, including the
    frozen-run checkbox and test-split counter in the UI.
12. Settings additions (§6), `pyproject.toml` dependency additions (§5),
    `.gitignore` entry for `var/`.
13. Capture `fruit-v1` images + run `bench.dataset import` (can happen any
    time after step 6).
14. Dev-split exploration through the UI (prompt iteration, parameter
    exploration, fitting `calculator_fits`/`prior_values` from dev
    specimens — flagged non-protocol as appropriate); freeze prompts
    (insert into `frozen_prompts`); one protocol-compliant, explicitly
    frozen test-split batch per final candidate model; export the
    decision-rule charts for the paper from `history.html`'s
    protocol-only filter.

## 11. Open items deferred, not blocking

- Retention/backup policy for `var/prestudy/` (images, DB, raw responses):
  default is "keep everything, no auto-deletion" since this is a personal
  research tool with modest data volume; revisit only if storage becomes
  a real concern.
- `uv_build` package-data inclusion of `templates/`/`static/`/`prompts/`:
  expected to work by default (everything under `src/food_vision/` is
  packaged); verify with a `uv build` dry-run once those files exist
  rather than pre-emptively configuring `[tool.uv_build]`.
- Which specific OpenRouter provider slugs/quantization labels exist per
  candidate model is an operational detail decided when the sweep runs
  (now that quantizations is optional, there's no structural blocker —
  just look up what each candidate's OpenRouter provider page offers).

## 12. Pre-implementation review — fixes applied in this revision

A review against the actual `OpenRouterClient` implementation and against
the validity risks of running a frozen-protocol benchmark through an
interactive UI found the following, all addressed above:

| # | Issue | Fix |
|---|---|---|
| 1 | `complete()` only read temperature/seed from `Settings`; this plan needed per-run values. | Added `temperature`/`seed` params to `complete()`, falling back to `Settings` (`proxy/openrouter.py`). |
| 2 | Cost telemetry was always flagged as an estimate — OpenRouter only reports real cost when asked. | Added `extra_body.usage.include = True` to every request. |
| 3 | `quantizations` was mandatory for `RoutingMode.BENCHMARK`, excluding candidates with no quantization label (Gemini, GPT, Claude). | Made `quantizations` optional; only `provider_pin` is required, carried on a new `RoutingPolicy`. |
| 4 | `zdr`+`deny`+single-pin+no-fallback is a narrow intersection; some candidates will have no matching provider. | `RoutingPolicy` is now a per-call object (`provider_pin`, `quantizations`, `zdr`, `data_collection`); the effective policy is recorded on `run_batches.routing_policy` and `CompletionResult.effective_routing_policy`, so a failing combination is one row to inspect. |
| 5 | A shared `sqlite3` connection on `app.state` plus `BackgroundTasks` would raise `ProgrammingError` across threads. | Connection-per-request and connection-per-background-task via `bench.db.connect()`, WAL mode, no connection stored on `app.state` (§7.1). |
| 6 | `protocol_compliant` ignored the prompt; a test-split run with a tweaked prompt would wrongly count as protocol. | Added `frozen_prompts` table; compliance now requires `split != 'test' OR prompt_hash` frozen (§7.3). |
| 7 | Nothing stopped one click from burning the frozen test split during exploration. | `create_batch` requires an explicit `is_frozen_run=True` for `split='test'`, surfaced as an unchecked-by-default UI checkbox; `history.html` shows a per-(model,strategy) test-split-batch counter. |
| 8 | S3's prior source ("CSV or hardcoded") risked leaking test-split information or being untraceable. | `prior_values` table, `source` pinned to `'dev_specimens'` or `'external_table'`, recorded per experiment/type. |
| 9 | Fitted constants (banana `a`, round-fruit densities) were planned as code constants despite being fit on dev data. | `calculator_fits` table keyed by `experiment_id`; `run_batches.calculator_fit_experiment_id` records which fit an S2/B1 batch used. |
| hygiene | Docstrings/boilerplate carried changelog prose ("the previous version was..."); belongs in commit messages. | Trimmed in `model_provider.py` and `docs/dev/boilerplate.md`. |
| hygiene | `.python-version` pinned 3.12 while the concept doc said 3.13. | Concept doc's runtime line and ADR-002 updated to Python 3.12 (3.13 acceptable) — one source of truth, matching what's actually pinned. |
| hygiene | `per-file-ignores = ["S101"]` in `pyproject.toml` was dead config (`S` ruleset not selected). | Removed. |
| hygiene | Provider-side image resizing (Gemini ~768px tiles, OpenAI ~512px tiles) confounds the 768-vs-1024px resolution comparison. | Comparison kept; `bench/charts.py`'s resolution-related charts/report carry an explicit caveat (§8.8) rather than presenting it as a clean controlled variable. |
