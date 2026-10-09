# Pre-study Implementation Concept — Web UI (Appendix A, Phase P)

> **Archived (2026-10-10).** Superseded by `docs/dev/pre-study.md`, the lean
> ECUSTFD-based pre-study. This full hybrid CLI + web UI design (SQLite,
> content-addressed sweeps/protocols, lease-based executor, chart
> provenance) was not built. Revive it only if the transfer check in
> `pre-study.md` §6 fails, or if the pre-study becomes a publication and
> needs this level of rigor/reproducibility.

Status: supersedes the **interface layer** of
`docs/dev/pre-study-implementation.md` (which was CLI-only). The Appendix A
*domain* decisions — dataset, strategies, metrics, decision rule — are
unchanged and are not repeated in full here; see that doc and concept
§Appendix A for the source of truth. What changes is how a run is
triggered and where results live: a local FastAPI + SQLite web app
instead of a `food-vision bench` CLI command, because the actual need is
persisted, queryable evidence and publication-quality charts, not just a
one-off batch sweep.

This revision folds in two review passes:

1. A pre-implementation review against the actual `OpenRouterClient`
   implementation (fixes applied directly to `proxy/openrouter.py`; see
   §12 for the log).
2. An external methodology/engineering review (scored 6.5/10: research
   validity 8, reproducibility 5, data model 6, engineering 6,
   proportionality 5) arguing the plan stores *what was asked for, not
   what was returned*, is missing a parent "sweep" object above
   `run_batches`, under-specifies the independent variables worth
   controlling, and should be checked against Inspect AI before more
   custom engineering goes in. That check happened (§13) and the decision
   was to **keep the custom build** and absorb the review's missing
   fields (§7, §8, §9) rather than adopt an external harness — recorded
   here so the trade-off isn't silently lost.

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
- No SQLAlchemy/SQLModel — stdlib `sqlite3` is enough even at the larger
  row counts this revision implies (§7); an ORM would be a dependency
  with no payoff here.
- No adoption of Inspect AI / promptfoo as the execution engine — evaluated
  in §13, decision was to keep this system self-contained.

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
│       ├── observe_fruit_s1_v1_en.md
│       ├── observe_fruit_s2_v1_en.md
│       ├── observe_fruit_s3_v1_en.md
│       └── ...                   # one file per (strategy, version, language) — §7.4
├── domain/
│   ├── __init__.py
│   └── calculator.py              # pure S2 formulas, kcal_from_edible_g
├── imaging/
│   ├── __init__.py
│   ├── preprocess.py              # decode, resize/normalize, strip EXIF, re-encode
│   └── covariates.py              # extract phone/focal-length/etc. from EXIF BEFORE stripping
└── bench/
    ├── __init__.py
    ├── db.py                       # sqlite3 connection-per-call helpers, schema, queries
    ├── dataset.py                  # fruit-v1 manifest import -> samples rows
    ├── sweeps.py                   # NEW: sweep creation, environment snapshot capture
    ├── runs.py                     # batch lifecycle, repeat execution, baselines
    ├── fruit_reference.py          # tiny kcal/density/edible-ratio lookup
    ├── calculator_fits.py          # stores/looks up fitted S2 constants per experiment
    ├── prior_values.py             # dev-only S3 prior provenance
    ├── metrics.py                  # MAPE, Spearman, Bland-Altman, ICC, median APE, R^2, CCC, bootstrap CI
    ├── charts.py                   # matplotlib SVG+PNG export from stored history
    └── web/
        ├── __init__.py
        ├── app.py                  # create_app(), lifespan, routes, DI
        ├── templates/
        │   ├── base.html
        │   ├── new_sweep.html        # NEW: define a sweep's fixed env/prompt/schema
        │   ├── new_run.html         # upload + param form (within a sweep)
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

- `imaging/covariates.py` is new (external review: phone model, focal
  length, capture distance, lighting, background, scale device/resolution
  are "cheap to record now and impossible to reconstruct later"). It runs
  **before** `preprocess.py` strips EXIF, writes the extracted fields to
  `samples` (§7.2), and the raw EXIF bytes are never sent to a model and
  never leave the local DB.
- `bench/sweeps.py` is new — captures the per-sweep environment snapshot
  (§7.3) once per sweep, not once per batch.
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
  "pillow>=11.0.0,<12.0.0",      # decode/resize/EXIF-read-then-strip, decompression-bomb guard
  "matplotlib>=3.9.0,<4.0.0",    # Agg backend, SVG+PNG chart export
]
```

Deliberately **not** added: SQLAlchemy/SQLModel, pandas/scipy/seaborn (the
expanded metrics in §9 are still implementable with plain
`statistics`/hand-rolled rank-correlation, Bland–Altman, and ICC — these
are closed-form/small-N computations, not a `pandas` justification by
themselves), HTMX/any JS framework, Inspect AI/promptfoo (§13).

## 6. Settings additions (additive only, `config/settings.py`)

```python
PRESTUDY_DB_PATH: Path = Path("var/prestudy/prestudy.db")
PRESTUDY_UPLOAD_DIR: Path = Path("var/prestudy/uploads")
PRESTUDY_NORMALIZED_DIR: Path = Path("var/prestudy/normalized")
PRESTUDY_MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024       # 15 MB
PRESTUDY_MAX_DECODED_PIXELS: int = 40_000_000            # decompression-bomb guard
```

No directories are created at import time or at `Settings()` construction
— creation happens once, in the app's `lifespan`.

This plan relies on `OpenRouterClient.complete()` accepting per-call
`temperature`, `seed`, and a `routing_policy: RoutingPolicy`
(`provider_pin` required for `RoutingMode.BENCHMARK`, `quantizations`/
`zdr`/`data_collection` optional overrides) — **already implemented**.
This revision adds a further requirement, not yet implemented in
`proxy/openrouter.py` (listed as outstanding in §12's new row): extending
`complete()`/`CompletionResult` to accept `top_p`, `max_tokens`, a
`reasoning` config (`effort: none|minimal|low|medium|high` or a token
budget, plus `exclude`), and a `structured_output_mode` switch
(`json_schema`/`tool_call`/`json_object`), and to **surface** on
`CompletionResult` what OpenRouter actually returned: `model_resolved`,
`system_fingerprint`, `generation_id`, `native_finish_reason`,
`reasoning_tokens`, `cached_tokens`, and whether a sent `seed` was
honoured (OpenRouter sometimes drops it silently — the only way to know
is to compare what was sent to what the provider's response implies, and
providers don't uniformly echo this back, so `seed_honoured` will often
have to be `NULL`/unknown rather than a confident boolean). This is a
proxy-layer change, out of scope for this doc to implement, but the data
model in §7 assumes it exists.

## 7. Data model (SQLite, stdlib `sqlite3`)

Why raw `sqlite3` over SQLAlchemy/SQLModel: the query surface (insert a
sweep/batch/attempts, update attempt status, filter/aggregate for charts)
is small and known up front even with the expanded column set below —
parameterized SQL plus a `schema_version` table covers it with zero extra
dependency surface, and keeps the "framework-free utilities" discipline
already used for logging. Row-count estimate is revised upward from the
original ~3–4k: a 54-image × 6-model × 3-repeat sweep alone is ~1,000
attempts, and several sweeps are expected across the study — still
trivially within SQLite's comfort zone (it scales to tens of millions of
rows on a single file), so this doesn't change the SQLite-vs-DB-server
decision, only confirms a `sweeps` parent object is needed to keep that
volume navigable (external review's "data model" finding).

### 7.1 Connection handling (unchanged: no shared connection across threads)

**Do not** open one `sqlite3.connect()` at startup and store it on
`app.state`. `sqlite3` connections default to `check_same_thread=True`,
and FastAPI's `BackgroundTasks` (used for `runs.execute_batch`, §8.6) run
in a thread-pool thread different from the request thread that opened the
connection — sharing one connection across that boundary raises
`sqlite3.ProgrammingError` at the first background-task query.

`bench/db.py` provides a connection-per-call context manager (WAL mode,
`busy_timeout`); the request's `get_db()` dependency and
`runs.execute_batch`'s background task each open their own connection via
this helper — never share one. See the original design rationale (still
valid) in this doc's git history if more detail is needed; unchanged by
this revision.

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
    -- Covariates (external review: "cheap to record now, impossible to
    -- reconstruct later"). Extracted from EXIF by imaging/covariates.py
    -- BEFORE imaging/preprocess.py strips EXIF for the outbound request;
    -- these never leave the local DB.
    phone_model TEXT,
    focal_length_mm REAL,
    capture_distance_cm REAL,       -- manual entry at capture time, not from EXIF
    lighting TEXT,                   -- controlled vocabulary, e.g. 'daylight'|'indoor_artificial'|'mixed'
    background TEXT,
    scale_model TEXT,                -- physical scale used for this sample's ground truth
    scale_resolution_g REAL,         -- e.g. 1.0, 0.1
    created_at TEXT NOT NULL
);

CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    manifest_sha256 TEXT,          -- NULL for ad-hoc/manual experiments
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE frozen_prompts (
    id INTEGER PRIMARY KEY,
    strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3')),
    prompt_hash TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    UNIQUE (strategy, prompt_hash)
);

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

CREATE TABLE prior_values (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    fruit_type TEXT NOT NULL,
    p10_g REAL NOT NULL, p50_g REAL NOT NULL, p90_g REAL NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('dev_specimens', 'external_table')),
    derived_at TEXT NOT NULL,
    UNIQUE (experiment_id, fruit_type)
);

-- NEW (external review "data model" finding #1): the parent object a
-- batch of batches belongs to. One sweep = one launch sharing the same
-- code version, prompt text, schema version, and settings snapshot —
-- e.g. "54 images x 6 models x 3 repeats" is one sweep containing ~324
-- batches, not 324 unrelated rows with no common anchor. This is also
-- where the ISERN-guideline-style "exact model version and run date,
-- full config" requirement gets satisfied: one row per sweep, not
-- reconstructed after the fact from scattered batch columns.
CREATE TABLE sweeps (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    name TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    git_dirty INTEGER NOT NULL,
    food_vision_version TEXT NOT NULL,
    openai_sdk_version TEXT NOT NULL,
    pillow_version TEXT NOT NULL,
    prompt_text TEXT NOT NULL,        -- full snapshot, not only a hash
    prompt_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    settings_snapshot TEXT NOT NULL,  -- JSON, secrets stripped (no API keys)
    created_at TEXT NOT NULL
);

CREATE TABLE run_batches (
    id INTEGER PRIMARY KEY,
    sweep_id INTEGER REFERENCES sweeps(id),       -- NULL only for pre-sweep ad-hoc exploration
    experiment_id INTEGER REFERENCES experiments(id),
    -- Fix (external review "data model" finding #2, multi-view support):
    -- a batch no longer points at exactly one sample. `primary_sample_id`
    -- is the specimen whose ground truth this batch is scored against;
    -- `batch_images` (below) lists every image actually attached to the
    -- request, so a `views='c1_c2'` batch has two rows there (both
    -- conditions of the same specimen) while a single-view batch has one.
    primary_sample_id INTEGER NOT NULL REFERENCES samples(id),
    strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3','B0','B1')),
    model TEXT,                     -- NULL for B0/B1 (no provider call)
    routing_policy TEXT,             -- JSON: {order, allow_fallbacks, quantizations?, zdr, data_collection}
    -- Generation knobs with a known/suspected confound effect get their
    -- own column (filterable, chartable); everything else lives in
    -- generation_config (JSON) to keep the schema stable as more knobs
    -- get added, same discipline as routing_policy above.
    temperature REAL,
    seed INTEGER,
    top_p REAL,
    max_tokens INTEGER,
    reasoning_effort TEXT CHECK (
        reasoning_effort IN ('none','minimal','low','medium','high') OR reasoning_effort IS NULL
    ),
    structured_output_mode TEXT CHECK (
        structured_output_mode IN ('json_schema','tool_call','json_object')
    ),
    generation_config TEXT,          -- JSON: reasoning_max_tokens, reasoning_exclude, etc.
    resolution_px INTEGER,
    image_format TEXT CHECK (image_format IN ('JPEG','WEBP','PNG')),
    image_detail TEXT CHECK (image_detail IN ('low','high','auto') OR image_detail IS NULL),
    views TEXT NOT NULL DEFAULT 'c1' CHECK (views IN ('c1','c2','c1_c2')),
    image_config TEXT,               -- JSON: jpeg_quality, etc.
    prompt_id TEXT,
    prompt_version INTEGER,
    prompt_hash TEXT,
    prompt_config TEXT,              -- JSON: system_vs_user, persona_enabled,
                                      --       reference_card_stated, prompt_language,
                                      --       field_order, confidence_field_enabled
    schema_version TEXT,
    repeats INTEGER NOT NULL,        -- UI label: "epochs" (Inspect AI's term, per review)
    calculator_fit_experiment_id INTEGER REFERENCES experiments(id),
    prior_experiment_id INTEGER REFERENCES experiments(id),  -- S3 prior_source pointer
    is_frozen_run INTEGER NOT NULL DEFAULT 0,
    protocol_compliant INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('queued','running','completed','failed','interrupted')),
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE batch_images (
    batch_id INTEGER NOT NULL REFERENCES run_batches(id),
    sample_id INTEGER NOT NULL REFERENCES samples(id),
    role TEXT NOT NULL CHECK (role IN ('primary', 'secondary')),
    PRIMARY KEY (batch_id, sample_id)
);

CREATE TABLE run_attempts (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES run_batches(id),
    repeat_index INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','ok','schema_invalid','error','refused')),
    raw_response TEXT,               -- full JSON, not just parsed fields — re-parse without re-paying
    edible_g_pred REAL,
    kcal_pred REAL,
    length_cm_pred REAL, max_diameter_cm_pred REAL,   -- S2 only
    confidence_pred REAL,             -- when confidence_field_enabled
    schema_valid INTEGER,
    schema_valid_first_try INTEGER,   -- distinct from schema_valid: did it need a repair pass?
    repair_attempted INTEGER,
    parse_error TEXT,
    refusal INTEGER,
    -- Reproducibility fields (external review "reproducibility" finding:
    -- "stores what you asked for, not what you got"). Requires the
    -- proxy-layer CompletionResult extension noted in §6.
    model_requested TEXT,
    model_resolved TEXT,              -- aliases move; group/re-baseline by this + system_fingerprint
    system_fingerprint TEXT,
    provider_served TEXT,
    openrouter_generation_id TEXT,     -- lets native token counts/upstream ID be fetched later
    finish_reason TEXT,
    native_finish_reason TEXT,         -- detects truncation/refusals/content filters the normalized field hides
    prompt_tokens INTEGER, completion_tokens INTEGER,
    reasoning_tokens INTEGER, cached_tokens INTEGER,
    cost_usd REAL, cost_is_estimate INTEGER,
    latency_ms REAL,
    started_at TEXT, ended_at TEXT,     -- UTC ISO-8601 — correlates drift with provider-side deployments
    attempts INTEGER,                   -- OpenRouterClient's own retry count
    request_sha256 TEXT,                -- canonical request minus image bytes — exact replay
    image_sent_sha256 TEXT,             -- proves which bytes were actually evaluated
    seed_sent INTEGER,
    seed_honoured INTEGER,              -- nullable: NULL when the provider gives no way to tell
    reasoning_text TEXT,                 -- optional, kept separate from raw_response for readability
    human_reviewed INTEGER NOT NULL DEFAULT 0,  -- ISERN guideline: validate a subset by hand
    human_review_notes TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE chart_exports (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,              -- e.g. 'predicted_vs_true', 'bland_altman', 'mape_by_model'
    filter_spec TEXT NOT NULL,       -- JSON: sweep, experiment, split, strategy, models, exploratory-toggle, etc.
    attempt_ids TEXT NOT NULL,       -- JSON array of run_attempts.id used, frozen at export time
    bootstrap_seed INTEGER,
    svg_path TEXT NOT NULL, png_path TEXT NOT NULL,
    svg_sha256 TEXT NOT NULL, png_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

### 7.3 Per-sweep environment snapshot (`bench/sweeps.py`)

`create_sweep(db, experiment_id, name, prompt_path, schema_version) -> int`
reads `importlib.metadata.version("food_vision")`,
`importlib.metadata.version("openai")`,
`importlib.metadata.version("pillow")`, runs `git rev-parse HEAD` +
`git status --porcelain` (dirty flag) against the repo root, reads the
full prompt text at `prompt_path` (and hashes it), computes a schema hash
from `observation/schema.py`'s current output for that strategy, and
serializes `get_settings()` with every field whose name contains `KEY`,
`SECRET`, or `TOKEN` stripped — all inserted as one `sweeps` row before
any batch under it is created. A sweep is immutable once created (no
update path); starting a new sweep is the only way to change prompt/code/
settings mid-study, which is the point — it's what makes "group results
by fingerprint and re-baseline when it changes" (external review) and
ISERN's "exact model version and run date, full config" both just a
`sweeps` row lookup instead of archaeology.

### 7.4 `protocol_compliant`

A batch is `protocol_compliant = 1` iff:

```text
temperature == 0
AND resolution_px IN (1024, 768)
AND repeats == 3
AND (primary_sample.split != 'test' OR (strategy IN ('B0','B1')) OR
     EXISTS frozen_prompts WHERE strategy = batch.strategy
                             AND prompt_hash = batch.prompt_hash)
```

Computed once at batch creation in `runs.py`, stored (not recomputed ad
hoc). `metrics.py`'s decision-rule aggregations (MAPE/Spearman ρ/slope/
bias/Bland–Altman/ICC/median APE/R²/CCC — concept §Appendix A.6 plus §9's
additions) filter to `protocol_compliant = 1` **by default**;
`history.html` and `charts.html` both expose an explicit "include
exploratory runs" toggle, and every chart export's `filter_spec` records
whether that toggle was on.

Unchanged from the prior revision otherwise: the frozen-test-split
guard (`is_frozen_run`) and per-(model, strategy) test-split counter in
`history.html` apply exactly as before.

## 8. Component responsibilities

### 8.0 `imaging/covariates.py` (new)

`extract_covariates(raw_image_bytes) -> SampleCovariates` — reads EXIF via
Pillow (`Image.getexif()`) **before** `preprocess.normalize_image` strips
it: camera make/model, focal length (35mm-equivalent where available).
Capture distance/lighting/background/scale-model/scale-resolution are not
in EXIF — they're entered once per capture session alongside the
manifest (`fruit-v1`) or via a small form field for manual uploads, and
`bench/dataset.py`/the upload route write them onto the `samples` row.
Pure function on image bytes; no network, no DB — easy to unit test
against a handful of fixture images with known EXIF.

### 8.1 `imaging/preprocess.py`

Unchanged contract: `normalize_image(path_or_bytes, max_px, *, image_format='JPEG', jpeg_quality=90) -> bytes`
— Pillow decode (enforcing `PRESTUDY_MAX_DECODED_PIXELS` before full
decode), resize longest edge to `max_px`, strip EXIF, re-encode. `format`/
`jpeg_quality` are now parameters (external review: independent variables
to make configurable), not hardcoded — called once per image per batch's
`image_format`/`jpeg_quality`/`resolution_px`.

### 8.2 `observation/{schema.py,adapter.py,prompts/}`

Expanded surface vs. the prior revision:

```python
def observe(
    client: ModelProvider,
    *,
    image_bytes: list[bytes],          # 1 entry for single-view, 2 for views='c1_c2'
    strategy: Strategy,
    model: str,
    routing_mode: RoutingMode,
    routing_policy: RoutingPolicy,
    temperature: float,
    seed: int | None,
    top_p: float | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
    reasoning_config: ReasoningConfig | None,   # max_tokens budget, exclude
    structured_output_mode: StructuredOutputMode,  # json_schema | tool_call | json_object
    prompt_id: str,
    prompt_version: int,
    prompt_config: PromptConfig,        # system_vs_user, persona_enabled,
                                         # reference_card_stated, language, field_order,
                                         # confidence_field_enabled
    schema_version: str,
) -> ObservationResult: ...
```

- Prompt files are now named `{strategy}_v{version}_{language}.md`
  (§4), loaded via `prompt_id`+`prompt_version`+`prompt_config.language`;
  `prompt_config.field_order='observations_first'` picks a schema/prompt
  variant that asks for a free-text `observations` field *before*
  `edible_g` — in-schema chain-of-thought without spending reasoning
  tokens (external review).
- `structured_output_mode` selects how the schema is enforced: strict
  `response_format: json_schema` (current default), an OpenRouter
  `tools`/function-call shape, or a `json_object` mode where the schema is
  described in the prompt text only and validated after the fact — not
  every candidate model supports `json_schema`, and the mode itself can
  change results, so it has to be a logged variable, not an assumption.
  **`structured_output_mode` is itself a `run_batches` column** (§7.2),
  filterable in `history.html`/charts.
- `views='c1_c2'` attaches both condition images in one message — cheap
  multi-view condition to test (external review cites Cal AI using this).
  `observe()` takes `image_bytes: list[bytes]` for this reason even though
  almost all batches pass a single-element list.
- A reasoning-exhaustion failure mode is explicitly handled, not silently
  mis-scored: OpenRouter reasoning tokens count against `max_tokens`; when
  reasoning consumes the whole budget, the response comes back with
  `finish_reason: length` and empty `content`. `adapter.observe` detects
  this (`finish_reason == 'length' and not content`) and returns a
  distinct `ObservationResult.error = "reasoning_budget_exhausted"` rather
  than a generic schema-parse failure — this is a capacity/config problem,
  not a model-capability one, and the two must not be conflated in
  `schema_valid_rate`.
- `routing_policy`/`temperature`/`seed`/everything else above is passed
  through from the batch's stored params, never re-read from `Settings`
  here — unchanged reasoning from the prior revision, just a longer
  parameter list now.

### 8.3 `domain/calculator.py`

Unchanged: pure S2 formulas, property-tested, called with constants from
`bench/calculator_fits.py` keyed by `experiment_id`.

### 8.4 `bench/fruit_reference.py` / `bench/calculator_fits.py` / `bench/prior_values.py`

Unchanged from the prior revision.

### 8.5 `bench/dataset.py`

Unchanged core contract (`import_manifest`), extended to also populate the
new `samples` covariate columns (§7.2) from manifest fields (if the
manifest schema is extended to carry them — see §11 open item) or from a
companion per-specimen metadata file if not.

### 8.6 `bench/sweeps.py` (new, see §7.3)

`create_sweep(...) -> int` as described in §7.3. Called once per sweep
from `new_sweep.html` (§8.9), not per batch.

### 8.7 `bench/runs.py`

- `create_batch(db, *, sweep_id, primary_sample_id, secondary_sample_id=None,
  views, strategy, model, routing_policy, temperature, seed, top_p,
  max_tokens, reasoning_effort, reasoning_config, structured_output_mode,
  resolution_px, image_format, jpeg_quality, image_detail, prompt_id,
  prompt_version, prompt_config, schema_version, repeats,
  experiment_id, calculator_fit_experiment_id=None, prior_experiment_id=None,
  is_frozen_run=False) -> int`:
  - All validation from the prior revision still applies (model
    allowlist, `routing_policy.provider_pin` required for S1–S3 and
    `quantizations` optional, B0/B1 ignore routing/generation params, S2/
    B1 require a resolvable `calculator_fits` row, S3 requires a
    resolvable `prior_values` row, frozen-test-split guard, `prompt_hash`
    resolved from the sweep's frozen prompt text + `prompt_config`).
  - `views='c1_c2'` requires `secondary_sample_id` to be a sample with the
    same `specimen_id` and the complementary condition (`c2` when primary
    is `c1`); inserts two `batch_images` rows.
  - `structured_output_mode` defaults to `'json_schema'` but is validated
    against a small per-model capability table (not every model supports
    strict JSON Schema) rather than assumed universal.
  - Computes `protocol_compliant` per §7.4, inserts the batch + N queued
    `run_attempts` rows in one transaction, returns `batch_id`.
- `execute_batch(db_path, client, batch_id)` — unchanged connection
  discipline (§7.1: own connection, never the request's). For each repeat:
  resolves image(s) via `batch_images`, builds the request via
  `observation.adapter.observe` with the full parameter set above, and
  writes every reproducibility field in §7.2's `run_attempts` columns from
  the (proxy-layer-extended, §6) `CompletionResult`: `model_resolved`,
  `system_fingerprint`, `openrouter_generation_id`, `native_finish_reason`,
  `reasoning_tokens`, `cached_tokens`, `request_sha256`/
  `image_sent_sha256` (computed here, not by the proxy — the proxy
  shouldn't need to know about canonicalization rules), `started_at`/
  `ended_at` (UTC, bracketing the call). A `reasoning_budget_exhausted`
  result is written with `status='error'`, `parse_error` set to that
  string, and is excluded from `schema_valid_rate` denominators (it's not
  a parse attempt at all).
- Startup `interrupted`-marking: unchanged from the prior revision.

### 8.8 `bench/metrics.py`

Existing functions (`mape`, `spearman_rho_and_slope`, `signed_bias`,
`repeat_cv`, `schema_valid_rate`, `bootstrap_ci`) plus, per the external
review's "metrics to add":

- `bland_altman(predicted, true) -> BlandAltmanResult` — mean bias, SD of
  differences, 95% limits of agreement (`bias ± 1.96·SD`).
- `icc(repeated_predictions_by_group) -> float` — intraclass correlation
  across repeats (two-way random, agreement form — the variant used in
  the cited 2026 GPT-5.2/Gemini-3-Flash/Claude-Sonnet-4.6 weighed-meals
  study), computed as a one-way ANOVA-based ICC(1) to stay inside plain
  `statistics` rather than pulling in `pingouin`/`statsmodels`.
- `median_ape(predicted, true) -> float` — more robust to outlier
  specimens than mean MAPE; reported alongside it, not instead of it.
- `r_squared(predicted, true) -> float`.
- `lins_ccc(predicted, true) -> float` — Lin's concordance correlation
  coefficient, combines precision (Pearson r) and accuracy (bias from the
  45° line) into one number; complements Spearman ρ (rank-only) and the
  Bland–Altman bias/LoA (additive-bias-focused).

All of the above take an explicit `protocol_compliant_only: bool = True`
filter argument, same discipline as the existing metrics.

### 8.9 `bench/charts.py`

Existing chart kinds unchanged, plus:

- `bland_altman_plot` (difference vs. mean, bias line, LoA band, per
  model/strategy) — a standard figure type in the domain literature cited
  by the review (Cureus 2026, Nakagawa & Yamamoto 2026, ACETADA); directly
  reusable in a paper.
- `calibration_plot` — when `confidence_field_enabled`, predicted
  confidence vs. empirical accuracy, binned; "nearly free" per the review
  since `confidence_pred` is already a stored column.
- `reasoning_effort_tradeoff` — cost/latency/MAPE vs. `reasoning_effort`,
  the parameter the review flags as "mov[ing] cost, latency and probably
  accuracy more than temperature does."

Resolution-comparison charts keep the existing provider-side-tiling
caveat (Gemini ~768px tiles, OpenAI ~512px tiles confound the 768-vs-1024
comparison) — unchanged from the prior revision.

### 8.10 `bench/web/app.py`

New `new_sweep.html` form (git SHA/dirty flag and package versions are
read automatically, not entered by hand; the researcher picks experiment,
prompt file + version, schema version, and names the sweep) feeding
`bench/sweeps.py`, sitting in front of the existing `new_run.html` (which
now also exposes the expanded parameter set: `top_p`, `max_tokens`,
reasoning effort/budget/exclude, `structured_output_mode`, image
format/quality/detail, `views`, prompt placement/persona/reference-card-
stated/language/field-order/confidence-field toggles — grouped into
collapsible sections in the template so the common path, S1/S2/S3 with
mostly-default params, isn't buried). Connection handling (§7.1),
frozen-run checkbox (§7.4), and history/chart routes are otherwise
unchanged from the prior revision.

## 9. Open-model baseline and contamination (external review)

Per the ISERN LLM-study guidelines cited in the review (exact model
version + run date, full config, exact prompts, **an open-model
baseline**, **a contamination check**, human validation of a subset):

- **Open-model baseline**: add an open-weights candidate to the model
  list (the review names Qwen3-VL as the general candidate and Food-R1 —
  a food-specialized Qwen3-VL-8B with published Nutrition5k numbers — as
  a food-specific one). No code/schema change: it's just another `model`
  value on OpenRouter (or a self-hosted endpoint behind the same
  `ModelProvider` Protocol, if not available on OpenRouter) going through
  the same batch/attempt pipeline. Tracked as an open item (§11) because
  model-candidate selection is an operational decision made when the
  sweep runs, consistent with how the rest of this doc treats the
  candidate list.
- **Contamination**: `fruit-v1`'s own photos sidestep training-set
  contamination by construction (they're not a published benchmark like
  Nutrition5k) — worth stating explicitly in the eventual report rather
  than leaving it implicit.
- **Human validation of a subset**: `run_attempts.human_reviewed`/
  `human_review_notes` (§7.2) exist so a researcher can mark a sampled
  subset of attempts as manually checked against the image — a workflow
  note (check a stratified sample after each sweep), not a feature to
  build; `history.html`'s filter can select `human_reviewed = 0` rows to
  sample from.

## 10. Testing plan

All against a temp SQLite file per test (`tmp_path`), FastAPI
`TestClient`, and a fake `ModelProvider` — **no live OpenRouter calls**.

- `tests/unit/test_bench_db.py` — schema creation/versioning, insert/query
  round-trips for every table including the new `sweeps`/`batch_images`,
  WAL/foreign-key pragmas applied, cross-thread connection regression
  test (unchanged from the prior revision).
- `tests/unit/test_bench_sweeps.py` — `create_sweep` captures git SHA/
  dirty flag/package versions correctly (mocked `subprocess`/
  `importlib.metadata`), strips secret-like settings keys, is immutable
  (no update path exists).
- `tests/unit/test_imaging_covariates.py` — EXIF extraction against
  fixture images with known/missing EXIF (missing EXIF yields `None`
  fields, not an exception); confirms the function never touches the
  network or DB.
- `tests/unit/test_bench_dataset.py`, `test_bench_calculator_fits.py`,
  `test_bench_prior_values.py` — unchanged from the prior revision.
- `tests/unit/test_bench_runs.py` — all prior-revision cases, plus:
  `views='c1_c2'` requires a matching-specimen secondary sample and
  inserts two `batch_images` rows; `structured_output_mode` validated
  against a per-model capability table; `execute_batch` writes every new
  reproducibility column from a fake `CompletionResult` carrying
  `model_resolved`/`system_fingerprint`/etc.; a `finish_reason: length`
  with empty content is recorded as `reasoning_budget_exhausted` and
  excluded from `schema_valid_rate`, not conflated with a generic parse
  failure.
- `tests/unit/test_bench_metrics.py` — existing cases plus: Bland–Altman
  bias/LoA against a hand-computed fixture, ICC against a known-value
  fixture (e.g. perfect agreement → ICC ≈ 1, independent noise → ICC ≈ 0),
  median APE and R² on simple fixtures, Lin's CCC against a published
  worked example.
- `tests/unit/test_bench_charts.py` — existing cases plus `bland_altman_plot`/
  `calibration_plot`/`reasoning_effort_tradeoff` produce valid SVG/PNG
  output; no pixel comparison.
- `tests/unit/test_prestudy_web.py` — existing cases plus `new_sweep.html`
  round-trip (create sweep, then a batch referencing it); the expanded
  `new_run.html` param form rejects an unsupported
  `structured_output_mode` for the selected model (422).

## 11. Open items deferred, not blocking

- Retention/backup policy for `var/prestudy/`: unchanged — keep
  everything, no auto-deletion.
- `uv_build` package-data inclusion: unchanged — verify with a dry-run
  once files exist.
- Exact OpenRouter provider slugs/quantization labels per candidate: an
  operational decision made when the sweep runs.
- **Final candidate model list**, including whether/which open-weights
  model (Qwen3-VL, Food-R1) is actually reachable (via OpenRouter or a
  self-hosted endpoint) — decide when assembling the first sweep.
- Whether `fruit-v1`'s manifest format should be extended with the new
  `samples` covariate columns directly, or whether those are entered via
  a separate per-specimen metadata file joined at import time — either
  works; decide when capturing the dataset, not now.
- `structured_output_mode`'s per-model capability table (which candidates
  support strict `json_schema` vs. need `tool_call`/`json_object`) is
  itself something that has to be discovered empirically per candidate —
  not assumed, not hardcoded speculatively now.
- `ICC`'s exact variant (one-way vs. two-way random/mixed, agreement vs.
  consistency) — this doc picks ICC(1) one-way for simplicity; revisit if
  the repeat-measurement design turns out to need a different model
  (e.g. if "model" should be treated as a fixed rather than random
  factor).

## 12. Pre-implementation review — fixes applied in this revision (log, continued)

Carried forward from the prior revision (already applied in
`proxy/openrouter.py`): per-call `temperature`/`seed`, `usage.include`
cost telemetry, optional `quantizations`, `RoutingPolicy`, connection-
per-task SQLite handling, frozen-prompt-aware `protocol_compliant`,
frozen-test-split guard, dev-only fit/prior provenance, image-resizing
confound caveat, and the hygiene items (docstrings, Python version,
dead ruff config) — see this doc's git history for the full table.

New from the external methodology/engineering review, applied in this
revision:

| Area | Finding | Fix |
|---|---|---|
| Reproducibility | Stores what was asked for, not what was returned (no `system_fingerprint`, generation ID, provider served, request hash, reasoning/cached tokens, or exact timestamps). | New `run_attempts` columns (§7.2); requires a `proxy/openrouter.py` `CompletionResult` extension, tracked as outstanding in §6 — **not yet implemented**. |
| Data model | `run_batches` had one `sample_id` with no parent object; a 54×6 sweep is 324 unrelated-looking rows. | New `sweeps` table (§7.2/§7.3) as the parent; `run_batches.sweep_id`. |
| Data model (secondary) | No way to express a multi-view (`c1`+`c2` together) batch under a one-sample-per-batch model. | `primary_sample_id`/`batch_images` join table (§7.2), `views` column. |
| Proportionality | Web layer/chart gallery/history filters risk costing more than the study they serve; mature tools (Inspect AI, promptfoo) already do much of this. | Evaluated Inspect AI directly (§13) — OpenRouter provider, image input, structured output, and epochs all confirmed to exist. Decision: **keep the custom build** and absorb the missing fields instead of adopting it, made explicitly rather than by default. |
| Independent variables | Generation config (`reasoning` effort/budget, `top_p`, `max_tokens`), `structured_output_mode`, image format/quality/detail, `views`, and prompt-construction knobs (placement, persona, reference-card-stated, language, field order, confidence field) were not configurable/logged at all. | Added as explicit `run_batches` columns or a scoped `*_config` JSON column (§7.2); `observation/adapter.observe`'s signature extended (§8.2). |
| Engineering | Reasoning-token budget exhaustion (`finish_reason: length`, empty content) would otherwise be mis-scored as a generic schema failure. | Detected explicitly in `adapter.observe`, recorded as `reasoning_budget_exhausted`, excluded from `schema_valid_rate` (§8.2/§8.7). |
| Metrics | Missing Bland–Altman, ICC, median APE, R², Lin's CCC — all used in the cited 2026 domain studies. | Added to `bench/metrics.py` + corresponding chart kinds (§8.8/§8.9). |
| Methodology (ISERN) | No open-model baseline, no explicit contamination statement, no human-validation-of-a-subset workflow. | §9 added: Qwen3-VL/Food-R1 as an open-model-candidate open item, contamination statement noted, `human_reviewed` column + sampling workflow. |

## 13. Build-vs-buy: Inspect AI evaluation

The external review's recommendation — "before building the web UI, spend
half a day checking whether Inspect AI covers the harness" — was checked
directly rather than assumed either way:

- **Confirmed**: Inspect AI (UK AISI) has an OpenRouter provider
  (`openrouter/<model>`, with `extra_body` passthrough for
  `order`/`allow_fallbacks`/`require_parameters`-style routing), accepts
  image inputs, supports structured output via `GenerateConfig.
  response_schema`, supports repeated `epochs` with configurable score
  reduction, ships a log viewer (`inspect view`), and exposes
  `evals_df()`/`samples_df()`/`messages_df()`/`events_df()` for pandas-
  based analysis.
- **Gaps found**: no confirmed built-in capture of `system_fingerprint`;
  cost tracking is a static price table (`set_model_cost()`), not
  OpenRouter's live `usage.include`-reported cost — both would need a
  custom metadata hook even under Inspect. A documented open issue
  affects Gemini reasoning models via OpenRouter (reasoning-block/thought-
  signature handling breaks tool calls) — not a direct blocker here since
  this study doesn't use tool calling, but worth re-checking if
  `structured_output_mode='tool_call'` is ever used with a Gemini
  candidate.
- **UI mismatch**: Inspect's interface is a log viewer over completed
  eval runs, not an interactive "upload a photo, pick params, click run"
  browser form — the explicit requirement behind building this web app in
  the first place.
- **Decision**: keep the custom FastAPI+SQLite build, absorb the review's
  missing fields (§7–§9) rather than adopt Inspect AI or promptfoo as the
  execution engine. Recorded here so this trade-off — more total
  engineering, but one system fully owned and an interactive UI preserved
  — isn't silently lost if revisited later.
