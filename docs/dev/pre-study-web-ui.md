# Pre-study Implementation Concept — Phase P (hybrid CLI + web UI)

**Status:** implementation spec for concept Appendix A. Self-contained: no
rule in this document depends on an earlier revision or on git history.
**Supersedes:** `docs/dev/pre-study-implementation.md` (CLI-only plan) in
full. That file is kept for history only.
**Scope:** single whole fruit, edible grams, model selection. Nothing from
Phase 0+ (matching, enrichment, ranges, reference store, meal API).

## 1. Goal, research questions, exit criterion

**Goal.** Select a default and a fallback vision model, and a prompting
strategy, on test-set evidence that would survive peer review. Or establish
that no model perceives fruit size well enough, and stop.

**Research questions** (pre-registered; RQ1 decides, the rest inform):

| ID | Question | Analysis |
|---|---|---|
| RQ1 | Which model/strategy estimates edible grams best, and is it better than knowing only the fruit type? | §6.3 decision rule |
| RQ2 | Does the model perceive size, or answer from a type prior? | size slope β (§6.2) |
| RQ3 | Does "measure dimensions, compute in code" (S2) beat "estimate grams" (S1)? | paired S2−S1 per model |
| RQ4 | Does the reference card help? | paired c3−c2 per model |
| RQ5 | Does stating a prior (S3) help, or flatten β? | paired S3−S1, Δβ |
| RQ6 | Does reasoning (provider default vs off) change accuracy, cost, latency? | secondary arm, top-2 models |

**Exit.** `bench/reports/prestudy-v1/` contains the frozen protocol, the
analysis results JSON, tables and charts, and a written decision. The
decision is either (default, fallback, strategy) or "stop: no eligible
model".

## 2. Approach: hybrid, one core, two front-ends

```text
                 ┌───────────────────────── food_vision.prestudy ─────────────────────────┐
  CLI (typer)    │ dataset · fits · configs · protocols · sweeps · executor · analysis   │   Web UI (FastAPI, 127.0.0.1)
  batch work ───►│                     ▲                          │                       │◄── interactive work
  import, fit,   │                     │                          ▼                       │    probe one image,
  freeze, run,   │            observation.adapter  ───►  proxy.OpenRouterClient          │    browse sweeps/attempts,
  resume,        │            imaging.preprocess / covariates                             │    human review,
  analyze        │                     │                                                  │    view analysis + charts
                 │                     ▼                                                  │
                 │          SQLite var/prestudy/prestudy.db (WAL)                         │
                 └────────────────────────────────────────────────────────────────────────┘
```

| Concern | CLI | Web UI |
|---|---|---|
| Dataset import, fits, protocol freeze | yes | no |
| Sweep create / estimate / run / resume | yes | read-only progress |
| Test-split anything | yes (protocol sweeps only) | **never** (test images not addressable) |
| Single-image probe (dev or ad-hoc upload, ≤3 models) | no | yes |
| Attempt explorer, raw response, human review | no | yes |
| Analysis run + chart export | yes | renders stored results; can trigger dev-only exploratory analysis |

Rationale: long sweeps must not live inside a web request worker (reloads
kill them, `BackgroundTasks` has no resume). Interactive probing and visual
review need a browser. Both call the same service functions; neither
front-end contains logic.

## 3. Out of scope

- Matching, enrichment, ranges, BLS/FDC/OFF pipeline (a CSV of 6–8 fruit
  rows suffices), meal API under `food_vision/api/`.
- Auth, remote exposure, CORS. The web UI binds `127.0.0.1` only.
- Job queues, ORMs, pandas/scipy. `numpy` is used directly (already a
  transitive dependency of matplotlib).
- Inspect AI / promptfoo as engine. They don't offer a frozen-protocol
  guard, specimen-level statistics or an interactive probe; the
  content-addressed config store (§5) covers their reproducibility value.

## 4. Experimental design

### 4.1 Dataset `fruit-v1`

| Item | Value |
|---|---|
| Fruit types | banana, apple, orange, pear, kiwi, mandarin |
| Specimens per type | **8**, chosen to span smallest → largest available |
| Conditions per specimen | c1 top-down + card · c2 45° + card · c3 45° no card |
| Images | 6 × 8 × 3 = **144** |
| Split (by specimen) | dev **3**/type (size ranks 2, 5, 7) · test **5**/type (ranks 1, 3, 4, 6, 8) |
| Test set | 30 specimens, 90 images |

Why 8 per type: with the original 3 test specimens per type, within-type
statistics are degenerate (Spearman ρ can only take a handful of values)
and specimen-clustered CIs are uninformative. With 30 test specimens and a
typical APE SD of ~0.15, the 95% CI half-width on MAPE is about ±5 pp;
paired model differences are tighter. Differences below ~5 pp are reported
as unresolved, not as wins.

**Capture protocol.**
- Same phone, same neutral board, daylight, no zoom, 30–40 cm.
- ID-1 card (85.60 × 53.98 mm), flat, fully visible, blank side up.
- No cropping. HEIC accepted (decoded via `pillow-heif`).
- One capture session per day; the session sheet records lighting,
  background, distance, phone and scale.

**Ground truth per specimen.**
- `whole_g`, and `waste_g` (weighed after peeling/coring);
  `edible_g = whole_g − waste_g`.
- `length_cm`, `max_diameter_cm` (banana: outer curve + mid diameter).
- `bls_code`, `kcal_100g` (BLS 4.0); `kcal_ref = edible_g × kcal_100g / 100`.
- `scale_model`, `scale_resolution_g` (≤1 g required).

**Manifest.** `bench/sets/fruit-v1/specimens.csv` (one row per specimen,
ground truth + split + size rank) and `bench/sets/fruit-v1/images.csv` (one
row per image: specimen key, condition, filename, session id).
`bench/sets/fruit-v1/sessions.csv` holds the capture-session covariates.
All three are tracked; images are not. The dataset version is the sha256
of the three files concatenated in that order.

### 4.2 Factors: what varies, what is fixed

Every parameter is **recorded** (§5). Only these are **varied** in Phase P:

| Factor | Levels | Where |
|---|---|---|
| Model | 6–8 candidates (§4.4) | primary |
| Strategy | S1, S2, S3 (+ B0, B1 computed, no calls) | primary |
| Condition | c1, c2, c3 (a property of the image) | primary |
| Reasoning | `off` vs provider default | secondary, top-2 models, test c2 only |
| Resolution | 1024 vs 768 px | secondary, top-2 models, test c2 only |
| Views | c2 alone vs c1+c2 in one request | secondary, top-2 models |

Fixed at defaults for all primary arms (changeable only in exploratory
dev sweeps, never pooled with protocol results):

| Parameter | Value |
|---|---|
| `temperature` | `0`, or `null` (provider default) where the endpoint rejects 0; whichever is used is part of the arm |
| `top_p`, `seed` | unset / `20261009` |
| `max_tokens` | 2,048 (headroom for reasoning-by-default models) |
| `reasoning` | provider default (recorded); `off` only in the secondary arm |
| `structured_output_mode` | `json_schema` strict; `json_object` only for models without strict support (per-model capability, §7.3) |
| Repair on invalid output | **disabled** in protocol arms (measures capability, not our parser) |
| Image | JPEG q90, long edge 1024 px, EXIF-transposed then stripped, `detail` unset |
| Prompt | English, user message, persona line on, card dimensions stated, `observations` field before the answer, no confidence field |
| Routing | one pinned provider per model, `allow_fallbacks: false`, `zdr: true`, `data_collection: deny`, quantization pinned only for open-weight models |
| Repeats | 3 |

If a candidate cannot be served under `zdr: true`, it is either dropped or
run with `zdr: false` as a documented exception in the protocol file. It is
never relaxed silently.

### 4.3 Strategies and baselines

| ID | Model is asked for | Prediction |
|---|---|---|
| S1 | `edible_g` | direct |
| S2 | `length_cm`, `max_diameter_cm` | `calculator(type, L, D; fit)`; fit from dev specimens only |
| S3 | `edible_g`, prompt states dev-derived p10–p90 for the type | direct |
| B0 | — | dev mean `edible_g` per type |
| B1 | — | `calculator` on **true** tape measurements |

B0 and B1 are computed in the analysis, not executed as sweeps. Fits (S2
constants, S3 priors, B0 means) are produced once by `prestudy fit` from
dev specimens, content-hashed, and referenced by hash from every config
that uses them (§5.2). A fit derived from any test specimen is rejected by
a CHECK constraint.

### 4.4 Candidates

6–8 models: 2–3 frontier (Gemini Flash tier, Claude Sonnet tier, GPT
tier), 2 economical, 2 open-weight (Qwen3-VL as the general open baseline;
Food-R1 only if reachable behind the `ModelProvider` Protocol). The list is
fixed in the protocol file at freeze time after a capability smoke test
(`prestudy smoke`: image input, strict schema, ZDR route available, one
call each).

### 4.5 Contamination and human validation

- `fruit-v1` photos are new, so training-set contamination is excluded by
  construction. The report states this.
- After each protocol sweep, a stratified 10% sample of attempts (by model ×
  strategy) is human-reviewed in the web UI: was the right object
  measured, was the card used, was the output plausible. Inter-rater
  reliability is not applicable (single reviewer); the report says so.

## 5. Data model (SQLite, stdlib `sqlite3`, WAL)

### 5.1 Principles

- **Content-addressed configuration.** Everything that defines a
  measurement, except the model and the image, is one canonical JSON
  `run_config` identified by its sha256. Two results are comparable iff
  their config hashes are equal. A protocol is a set of
  (config hash, model, provider) arms. "Protocol-compliant" is a set
  membership test, not a formula.
- **Immutable inputs, append-only outputs.** Prompts, schemas, fits,
  configs and protocols are insert-only and keyed by hash. Attempts are
  updated only by the executor during their own lifecycle.
- **Idempotent work items.** A task is unique on
  (sweep, config, model, provider, image). Creating a sweep twice or
  resuming it never duplicates work.
- **Store what was returned, not only what was asked.**

### 5.2 Schema

```sql
CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);

CREATE TABLE datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,                       -- 'fruit-v1'
  content_sha256 TEXT NOT NULL UNIQUE,      -- §4.1
  imported_at TEXT NOT NULL
);

CREATE TABLE capture_sessions (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  session_key TEXT NOT NULL,
  phone_model TEXT, lighting TEXT, background TEXT,
  capture_distance_cm REAL, scale_model TEXT, scale_resolution_g REAL,
  captured_on TEXT,
  UNIQUE (dataset_id, session_key)
);

CREATE TABLE specimens (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  specimen_key TEXT NOT NULL,               -- 'banana_03'
  fruit_type TEXT NOT NULL,
  split TEXT NOT NULL CHECK (split IN ('dev','test')),
  size_rank INTEGER NOT NULL,
  whole_g REAL NOT NULL, waste_g REAL NOT NULL, edible_g REAL NOT NULL,
  length_cm REAL NOT NULL, max_diameter_cm REAL NOT NULL,
  bls_code TEXT NOT NULL, kcal_100g REAL NOT NULL, kcal_ref REAL NOT NULL,
  UNIQUE (dataset_id, specimen_key),
  CHECK (abs(whole_g - waste_g - edible_g) < 0.5)
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  specimen_id INTEGER REFERENCES specimens(id),        -- NULL = ad-hoc upload
  session_id INTEGER REFERENCES capture_sessions(id),
  source TEXT NOT NULL CHECK (source IN ('dataset','upload')),
  condition TEXT CHECK (condition IN ('c1','c2','c3')),
  raw_path TEXT NOT NULL,
  raw_sha256 TEXT NOT NULL UNIQUE,
  exif_make TEXT, exif_model TEXT, exif_focal_mm REAL, exif_focal_35mm REAL,
  width_px INTEGER NOT NULL, height_px INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (specimen_id, condition)
);

CREATE TABLE prompts (
  id INTEGER PRIMARY KEY,
  strategy TEXT NOT NULL CHECK (strategy IN ('S1','S2','S3')),
  version INTEGER NOT NULL, language TEXT NOT NULL,
  text TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  UNIQUE (strategy, version, language)
);

CREATE TABLE output_schemas (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL, version INTEGER NOT NULL,
  json TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  UNIQUE (name, version)
);

CREATE TABLE fits (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  kind TEXT NOT NULL CHECK (kind IN ('calculator','prior','baseline_b0')),
  derived_from_split TEXT NOT NULL CHECK (derived_from_split = 'dev'),
  payload TEXT NOT NULL,                    -- JSON, per fruit type
  sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE run_configs (
  id INTEGER PRIMARY KEY,
  canonical_json TEXT NOT NULL,             -- §5.3, sorted keys, no whitespace
  sha256 TEXT NOT NULL UNIQUE,
  label TEXT                                 -- human name, not part of identity
);

CREATE TABLE protocols (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,                 -- 'prestudy-v1'
  toml_text TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  git_sha TEXT NOT NULL,                      -- freeze refused if worktree dirty
  frozen_at TEXT NOT NULL,
  supersedes_id INTEGER REFERENCES protocols(id),
  supersede_reason TEXT,
  CHECK ((supersedes_id IS NULL) = (supersede_reason IS NULL))
);

CREATE TABLE protocol_arms (
  protocol_id INTEGER NOT NULL REFERENCES protocols(id),
  arm_key TEXT NOT NULL,                      -- 'gemini-flash/S2'
  run_config_id INTEGER NOT NULL REFERENCES run_configs(id),
  model TEXT NOT NULL, provider TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('primary','secondary')),
  PRIMARY KEY (protocol_id, arm_key)
);

CREATE TABLE sweeps (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('exploratory','protocol','probe')),
  protocol_id INTEGER REFERENCES protocols(id),
  split TEXT NOT NULL CHECK (split IN ('dev','test','adhoc')),
  budget_usd REAL NOT NULL,
  git_sha TEXT NOT NULL, git_dirty INTEGER NOT NULL,
  package_versions TEXT NOT NULL,             -- JSON: food_vision, openai, pillow, numpy, python
  settings_snapshot TEXT NOT NULL,            -- JSON, secrets removed
  status TEXT NOT NULL CHECK (status IN ('created','running','paused','completed','failed')),
  created_at TEXT NOT NULL, finished_at TEXT,
  CHECK (kind <> 'protocol' OR protocol_id IS NOT NULL),
  CHECK (split <> 'test' OR kind = 'protocol'),
  CHECK (kind <> 'protocol' OR git_dirty = 0)
);
CREATE UNIQUE INDEX one_test_sweep_per_protocol
  ON sweeps(protocol_id) WHERE split = 'test';

CREATE TABLE tasks (
  id INTEGER PRIMARY KEY,
  sweep_id INTEGER NOT NULL REFERENCES sweeps(id),
  run_config_id INTEGER NOT NULL REFERENCES run_configs(id),
  model TEXT NOT NULL, provider TEXT NOT NULL,
  primary_image_id INTEGER NOT NULL REFERENCES images(id),
  UNIQUE (sweep_id, run_config_id, model, provider, primary_image_id)
);

CREATE TABLE task_images (
  task_id INTEGER NOT NULL REFERENCES tasks(id),
  position INTEGER NOT NULL,
  image_id INTEGER NOT NULL REFERENCES images(id),
  PRIMARY KEY (task_id, position)
);

CREATE TABLE attempts (
  id INTEGER PRIMARY KEY,
  task_id INTEGER NOT NULL REFERENCES tasks(id),
  repeat_index INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('queued','claimed','ok','invalid','refused','truncated','error','skipped_budget')),
  claimed_at TEXT, heartbeat_at TEXT,         -- lease for resume (§7.2)
  -- request
  request_sha256 TEXT,                        -- canonical request, images replaced by their hashes
  images_sent_sha256 TEXT,                    -- JSON array, bytes actually sent
  -- response (what was returned)
  model_resolved TEXT, provider_served TEXT, system_fingerprint TEXT,
  generation_id TEXT, finish_reason TEXT, native_finish_reason TEXT,
  raw_response TEXT,                           -- full JSON body
  reasoning_text TEXT,
  parsed TEXT,                                 -- JSON or NULL
  schema_valid INTEGER,                        -- first try; no repair in protocol arms
  parse_error TEXT,
  -- prediction
  edible_g_pred REAL,                          -- S1/S3 direct, S2 via fit
  length_cm_pred REAL, max_diameter_cm_pred REAL,
  -- telemetry
  prompt_tokens INTEGER, completion_tokens INTEGER,
  reasoning_tokens INTEGER, cached_tokens INTEGER,
  cost_usd REAL, cost_is_estimate INTEGER,
  latency_ms REAL, client_retries INTEGER,
  started_at TEXT, ended_at TEXT,
  error TEXT,
  UNIQUE (task_id, repeat_index)
);
CREATE INDEX attempts_status ON attempts(status);

CREATE TABLE reviews (
  attempt_id INTEGER PRIMARY KEY REFERENCES attempts(id),
  verdict TEXT NOT NULL CHECK (verdict IN ('plausible','wrong_object','ignored_card','implausible','other')),
  notes TEXT, reviewed_at TEXT NOT NULL
);

CREATE TABLE analysis_runs (
  id INTEGER PRIMARY KEY,
  protocol_id INTEGER REFERENCES protocols(id),
  sweep_ids TEXT NOT NULL,                     -- JSON array
  scope TEXT NOT NULL CHECK (scope IN ('protocol_test','exploratory_dev')),
  n_boot INTEGER NOT NULL, bootstrap_seed INTEGER NOT NULL,
  results TEXT NOT NULL,                       -- JSON, §6
  git_sha TEXT NOT NULL, created_at TEXT NOT NULL
);

CREATE TABLE chart_exports (
  id INTEGER PRIMARY KEY,
  analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id),
  kind TEXT NOT NULL,
  svg_path TEXT NOT NULL, png_path TEXT NOT NULL,
  svg_sha256 TEXT NOT NULL, png_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

Charts are derived from an `analysis_run`, never directly from a filter,
so a figure, its numbers and its inputs always share one provenance chain.

### 5.3 `run_config` canonical JSON

```json
{
  "strategy": "S2",
  "prompt_sha256": "…", "output_schema_sha256": "…",
  "fit_sha256": "…",
  "generation": {"temperature": 0, "top_p": null, "seed": 20261009, "max_tokens": 2048,
                 "reasoning": {"mode": "default"}, "structured_output_mode": "json_schema",
                 "repair": false},
  "image": {"long_edge_px": 1024, "format": "JPEG", "quality": 90, "detail": null,
            "views": ["primary"]},
  "routing": {"zdr": true, "data_collection": "deny", "quantizations": null},
  "repeats": 3
}
```

`reasoning.mode` ∈ `default | off | effort` (with `effort` ∈
`minimal|low|medium|high|xhigh|max`) `| budget` (with `max_tokens`).
`views` is `["primary"]` or `["c1","c2"]` (the task's primary image plus
the same specimen's other condition). Model and provider are deliberately
outside the config so one config can be crossed with every candidate.

### 5.4 Configuration files (TOML, stdlib `tomllib`)

- `bench/configs/*.toml`: named exploratory configs (dev only).
- `bench/protocols/prestudy-v1.toml`: the pre-registration. It lists the
  dataset hash, arms (model × provider × config), the secondary arms, the
  primary condition set, the metric definitions version, the decision rule
  parameters (§6.3) and documented exceptions (e.g. a model without ZDR).
  `prestudy protocol freeze` resolves every referenced prompt, schema and
  fit to hashes, refuses on a dirty worktree, stores `toml_text` + sha256 +
  `git_sha`, and writes the resolved arms to `protocol_arms`.

## 6. Analysis

### 6.1 Units and aggregation

- **Analysis unit: specimen.** Images and repeats are nested within
  specimens and are never treated as independent.
- Per image and arm: prediction = **median** of valid repeats.
- Per specimen and arm (primary metric): mean of the c1 and c2 image
  predictions (card present). c3 is analysed separately (RQ4).
- **Missing data, intention-to-score:** an image with no valid repeat gets
  the B0 prediction for its type. This penalises unreliable models instead
  of silently dropping their hard cases. A per-protocol variant (exclusions
  instead) is reported alongside.
- `edible_g` is the target; kcal is reported as a derived column only
  (it equals grams × constant per type).

### 6.2 Metrics

| Metric | Definition | Role |
|---|---|---|
| MAPE | mean over specimens of \|pred − true\| / true | primary |
| MedAPE | median of the same | secondary (robust) |
| Relative gain vs B0 | 1 − MAPE(arm) / MAPE(B0), paired | gate |
| **Size slope β** | OLS of log(pred) on log(true) with fruit-type fixed effects, image level, c1+c2 | gate (RQ2); β ≈ 0 means prior-only |
| Geometric bias | exp(mean log(pred/true)) − 1 | gate |
| Validity | share of attempts with `schema_valid = 1` and status `ok` | gate |
| Repeatability | median within-image CV over repeats; ICC(2,1) absolute agreement (images × repeats) | gate (CV), descriptive (ICC) |
| Bland–Altman | on log scale: mean log-ratio, 95% limits of agreement, back-transformed to % | descriptive, figure |
| Lin's CCC, R² | on edible grams, specimen level | descriptive |
| Reference gain | paired MAPE(c3) − MAPE(c2) | RQ4 |
| S2 vs S1, S3 vs S1 | paired ΔMAPE, Δβ | RQ3, RQ5 |
| B1 | MAPE of calculator on true dims | ceiling for S2 |
| Cost / 1,000 images, latency P50/P95 | provider-reported cost; wall-clock per attempt | tie-breakers |
| Reasoning share | reasoning_tokens / completion_tokens | RQ6 |

Within-type Spearman ρ is reported per type as descriptive only (n = 5).

**Uncertainty.** Every interval is a 95% percentile interval from a
**specimen-cluster bootstrap**, stratified by fruit type, `n_boot` = 10,000,
fixed seed recorded in `analysis_runs`. Paired comparisons resample
specimens once and compute both arms on the same resample.

### 6.3 Decision rule (parameters fixed in the protocol file)

Eligibility gates, evaluated on the **test** split, primary arms, c1+c2:

- validity ≥ 98%
- β point estimate in [0.7, 1.3] **and** β CI lower bound > 0.3
- geometric bias within ±10%
- relative gain vs B0 ≥ 30% **and** its CI lower bound > 0
- median within-image CV ≤ 5%

Selection:

- Per model, its strategy is S1 unless S2 (or S3) has a paired ΔMAPE CI
  entirely below 0, in which case the better one is used. Simpler wins ties.
- Rank eligible (model, strategy) arms by MAPE. Any arm whose paired ΔMAPE
  vs the best has a CI containing 0 is **tied**. Among tied arms, choose
  lowest cost per 1,000 images, then lowest P95 latency.
- Fallback: the best eligible arm from a **different vendor**.
- **Stop rule:** if no arm passes the gates, the decision is "stop"; Phase 0
  does not start on photo estimation as designed.

Only RQ1's selection is confirmatory. RQ2–RQ6 are reported with intervals
but no significance claims; with ~20 comparisons, a few "significant"
differences are expected by chance.

## 7. Components

### 7.1 Package layout

```text
src/food_vision/
├── cli.py                              # typer root; `food-vision prestudy …`
├── imaging/
│   ├── preprocess.py                   # decode (+HEIC), exif_transpose, pixel guard, resize, strip, encode
│   └── covariates.py                   # EXIF make/model/focal read BEFORE stripping
├── observation/
│   ├── schema.py                       # output schemas S1/S3, S2; registered in output_schemas
│   ├── prompts/                        # fruit_s1_v1_en.md, fruit_s2_v1_en.md, fruit_s3_v1_en.md
│   └── adapter.py                      # observe(): messages, mode, parse, classify outcome
├── domain/
│   └── calculator.py                   # pure S2 formulas; kcal_from_edible_g
└── prestudy/
    ├── db.py                           # connect() ctx manager, migrations, queries
    ├── dataset.py                      # import specimens/images/sessions CSVs
    ├── fits.py                         # calculator, prior, B0 from dev; hashing
    ├── configs.py                      # TOML → canonical run_config; hashing; capability table
    ├── protocols.py                    # freeze, validate, arms
    ├── sweeps.py                       # create (env snapshot), expand to tasks, estimate
    ├── executor.py                     # leased worker pool, budget, resume, drain
    ├── analysis.py                     # §6 metrics, bootstrap, decision rule → analysis_runs
    ├── charts.py                       # matplotlib Agg, SVG + 600-dpi PNG from analysis_runs
    ├── report.py                       # results.json, tables.md, decision.md
    └── web/
        ├── app.py                      # create_app(); routes; Depends(get_db)
        ├── templates/                  # probe, sweeps, sweep_detail, attempts, attempt, analysis
        └── static/prestudy.css

bench/
├── sets/fruit-v1/{specimens.csv,images.csv,sessions.csv,README.md}   # images untracked
├── configs/*.toml                      # exploratory configs
├── protocols/prestudy-v1.toml          # pre-registration
└── reports/prestudy-v1/                # results.json, tables.md, decision.md, charts/
data/fruit_reference.csv                # bls_code, fruit_type, kcal_100g, waste/density priors
var/prestudy/                           # gitignored: prestudy.db, raw/, sent/
```

The package is named `prestudy`, not `bench`, so it can't be confused with
the top-level `bench/` data directory.

### 7.2 Executor (`prestudy/executor.py`)

- **Claiming.** Workers claim one queued attempt at a time with
  `UPDATE attempts SET status='claimed', claimed_at=?, heartbeat_at=?
  WHERE id = (SELECT id FROM attempts WHERE status='queued' AND task_id IN
  (…sweep…) LIMIT 1) RETURNING id`. Each worker uses its own connection.
- **Concurrency.** `ThreadPoolExecutor(max_workers=4)` by default, plus a
  per-provider semaphore (default 2) to stay under rate limits. 429/5xx
  retries remain the proxy's responsibility.
- **Lease and resume.** Workers heartbeat every 15 s. `prestudy sweep
  resume` (and every `run` start) returns `claimed` attempts with a
  heartbeat older than 120 s to `queued`. A crash, Ctrl-C or reboot loses
  at most the in-flight calls.
- **Budget.** Before claiming, the worker checks
  `spent + reserved + estimate_per_attempt ≤ budget_usd`. If exceeded, the
  sweep is set `paused`, and the remaining attempts stay `queued` (resumable
  with a raised budget). Estimates come from `prestudy sweep estimate`,
  which runs one real attempt per (config, model) on a dev image and
  extrapolates.
- **Graceful drain.** SIGINT stops claiming, waits for in-flight calls,
  writes their results, sets the sweep to `paused`.
- **Outcome classification** (from `adapter.observe`):
  `ok` · `invalid` (schema/parse failure) · `refused` (refusal text or
  content-filter `native_finish_reason`) · `truncated` (`finish_reason =
  length`; empty content with reasoning tokens means the reasoning budget
  was exhausted) · `error` (transport after retries). Only `ok` produces a
  prediction; validity counts `invalid` + `refused` + `truncated` as
  failures, `error` is retried by resume and excluded once permanent.

### 7.3 Observation adapter (`observation/adapter.py`)

`observe(provider, *, images: list[bytes], config: RunConfig, model, routing)
-> Observation`

- Builds one user message: prompt text, then image parts as base64 data
  URLs in `views` order.
- `structured_output_mode`: `json_schema` → `response_format` strict;
  `json_object` → schema text appended to the prompt, validated after.
  Supported modes per model come from `configs.CAPABILITIES`, filled by
  `prestudy smoke` and committed. A config requesting an unsupported mode
  for a model is rejected at sweep creation, not at call time.
- `repair` is `false` in protocol arms; exploratory configs may enable one
  repair round, recorded in the config.
- Prompts never contain numeric anchors except S3's stated prior. All
  prompts define "edible portion" per type (banana/orange/mandarin/kiwi:
  without peel; apple/pear: without core and stem).

### 7.4 Proxy extension (`proxy/openrouter.py`), required before §7.2

`complete()` gains `top_p`, `max_tokens`, `reasoning: dict | None`
(passed as OpenRouter `reasoning`), and `response_format` for both modes.
`CompletionResult` gains:

| Field | Source |
|---|---|
| `generation_id` | `response.id` |
| `model_resolved` | `response.model` |
| `system_fingerprint` | `response.system_fingerprint` (often null) |
| `native_finish_reason` | `choices[0].native_finish_reason` |
| `reasoning_text` | `choices[0].message.reasoning` |
| `reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens` |
| `cached_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| `raw` | full response as dict |

`prestudy sweep enrich <id>` optionally back-fills provider-side latency
and native token counts from OpenRouter's generation stats by
`generation_id`. Seed honouring is not stored per attempt; it is tested
once per model in `prestudy smoke` (two seeds, five calls each, compare
dispersion) and recorded in the capability table.

### 7.5 Imaging

- `preprocess.normalize(raw: bytes, cfg: ImageConfig) -> bytes`: refuse
  above `PRESTUDY_MAX_DECODED_PIXELS` before full decode, `ImageOps.exif_transpose`
  (otherwise portrait phone photos are sent sideways), resize long edge,
  drop all metadata, encode. Deterministic: same input + config → same bytes
  (tested by hash).
- `covariates.extract(raw: bytes) -> ExifCovariates`: make, model, focal
  length, 35 mm equivalent, dimensions. Stored in `images`, never sent.
- Bytes actually sent are written to `var/prestudy/sent/<sha256>.jpg` once,
  so any attempt can be replayed exactly.

### 7.6 Web UI (`prestudy/web/`)

Served by `food-vision prestudy serve` on `127.0.0.1:8800`. Pages:

| Page | Purpose |
|---|---|
| Probe | Pick a dev image or upload one (stored as `source='upload'`), pick ≤3 models and one named exploratory config, run. Creates a `probe` sweep with ≤9 attempts, executed in-process with the same executor (1 worker). Shows results side by side with the image. |
| Sweeps | List with kind, split, progress, cost vs budget, status. Read-only. |
| Sweep detail | Per-arm progress and outcome counts, cost so far, failures. |
| Attempts | Filterable table (sweep, model, strategy, status, fruit type, condition). |
| Attempt | Sent image, prompt, raw response, reasoning text, parsed output, prediction vs truth, review form. |
| Analysis | Renders a stored `analysis_run` (tables + charts). Can trigger an `exploratory_dev` analysis; protocol analysis is CLI-only. |

Hard rule: no route accepts or returns a test-split image, specimen or
attempt except Analysis pages rendering a completed `protocol_test` run.

### 7.7 CLI (`food-vision prestudy …`)

```text
import-dataset bench/sets/fruit-v1/
fit --dataset fruit-v1                               # calculator, prior, B0 from dev
smoke --models bench/configs/candidates.toml         # capability + seed probe, writes CAPABILITIES
sweep create --config bench/configs/dev-explore.toml --split dev --budget 5
sweep estimate <id> | run <id> [--workers 4] | resume <id> [--budget 20] | enrich <id>
protocol freeze bench/protocols/prestudy-v1.toml
sweep create --protocol prestudy-v1 --split test --budget 60 --confirm-test
analyze --protocol prestudy-v1 --out bench/reports/prestudy-v1/
serve
```

`--confirm-test` prints the arm list, image count, attempt count and
estimated cost, then requires typing the protocol name.

### 7.8 Settings additions

```python
PRESTUDY_DB_PATH: Path = Path("var/prestudy/prestudy.db")
PRESTUDY_DATA_DIR: Path = Path("var/prestudy")        # raw/, sent/
PRESTUDY_MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024
PRESTUDY_MAX_DECODED_PIXELS: int = 40_000_000
PRESTUDY_WORKERS: int = 4
PRESTUDY_PER_PROVIDER_CONCURRENCY: int = 2
```

No directory creation at import or `Settings()` time; `db.connect()` and
`serve` create what they need.

### 7.9 Dependencies

```toml
"typer>=0.12", "fastapi>=0.115,<1", "uvicorn[standard]>=0.32,<1",
"python-multipart>=0.0.12", "jinja2>=3.1,<4",
"pillow>=11,<12", "pillow-heif>=0.18", "numpy>=2.0", "matplotlib>=3.9,<4",
```

## 8. Test-split protection (summary of enforced rules)

| Rule | Enforced by |
|---|---|
| Test images only in `protocol` sweeps | `sweeps` CHECK; `sweeps.create` |
| Protocol sweeps only with clean worktree | `sweeps` CHECK on `git_dirty` |
| Only frozen arms in a protocol sweep | task expansion reads `protocol_arms` only |
| One test sweep per protocol | partial unique index; reruns via `resume` |
| A new test sweep needs a new protocol with `supersedes_id` + reason | `protocols` CHECK; report lists all protocols |
| Fits from dev only | `fits` CHECK |
| Web UI never exposes test data before analysis | route filters; tested |
| Exploratory analysis is dev-only | `analysis_runs.scope`; `analysis.run` |

## 9. Testing

No live network calls; fake `ModelProvider`; temp SQLite per test.

| Area | Tests |
|---|---|
| `imaging` | determinism by hash; exif_transpose applied; metadata absent after; pixel-bomb rejected; HEIC fixture decodes |
| `observation` | message shape per views; mode selection; outcome classification for ok / invalid / refused / truncated (length + reasoning tokens + empty content) |
| `calculator`, `fits` | property tests (monotone in L and D, positive, unit sanity); fit rejects any test specimen |
| `configs`, `protocols` | canonical JSON stable across key order; hash changes on any field change; freeze refuses dirty tree and unknown prompt/fit; unsupported mode rejected |
| `sweeps` | expansion counts; idempotent re-create; test sweep without protocol rejected; second test sweep rejected |
| `executor` | concurrent claim never double-claims (threads); stale lease requeued; budget pause and resume; SIGINT drain leaves no `claimed` rows |
| `analysis` | MAPE/bias/β/ICC/CCC/Bland–Altman against fixtures computed offline with scipy/pingouin and committed as JSON; cluster bootstrap resamples specimens, not images; ITT imputation; decision rule on synthetic arms (eligible, tied, stop) |
| `charts` | each kind writes valid SVG and PNG, registers one `chart_exports` row |
| `web` | probe flow end-to-end with fake provider; every route returns 404 for test-split ids; upload size/type limits |

Coverage floor raised to 85% for `prestudy/`, `observation/`, `imaging/`,
`domain/`.

## 10. Milestones

| M | Content | Exit | Status |
|---|---|---|---|
| M1 | proxy extension (§7.4), imaging, observation, calculator | `observe()` returns a classified Observation from a recorded fixture | **done** |
| M2 | db, dataset, fits, configs, sweeps, executor, CLI | dev sweep of 1 model × S1 runs, is killed, resumes, completes within budget | not started |
| M3 | capture `fruit-v1` (can run in parallel from day 1) | 144 images, CSVs validated by `import-dataset` | not started |
| M4 | smoke, dev exploration, prompt iteration, fits | capability table committed; prompts at v-final | not started |
| M5 | analysis, charts, report; web UI | `analyze` reproduces fixtures; probe + attempt review usable | not started |
| M6 | protocol freeze → test sweep → analyze → decision | `bench/reports/prestudy-v1/decision.md` | not started |

### M1 — done when / test / left over

| Step | DONE WHEN | TEST | LEFT OVER |
|---|---|---|---|
| Proxy extension (§7.4) | `complete()` accepts `top_p`/`max_tokens`/`reasoning`; `response_format` accepts `JsonObjectFormat`; `CompletionResult` carries `generation_id`, `model_resolved`, `system_fingerprint`, `native_finish_reason`, `reasoning_text`, `reasoning_tokens`, `cached_tokens`, `raw` | `tests/unit/test_openrouter.py` (25 tests: new fields present when returned, default to `None`/`{}` when absent, `top_p`/`max_tokens`/`reasoning` only sent when given, `JsonObjectFormat` payload shape) | Seed-honouring verification (per model, via `prestudy smoke`) is M4, not implemented. Provider-side latency/native-token back-fill by `generation_id` (`prestudy sweep enrich`) is M2+. |
| `domain/calculator.py` | `banana_volume_g`, `ellipsoid_mass_g`, `kcal_from_edible_g` are pure, positive, monotone in length/diameter, reject non-finite/negative/out-of-range inputs | `tests/unit/test_calculator.py` (28 tests: positivity, monotonicity, linear scaling, sphere-degeneracy cross-check, invalid-input rejection) | No fitted constants ship in this module by design — `shape_constant`/`density_g_cm3`/`edible_ratio` are required caller-supplied arguments. M2's `prestudy fit` produces and hashes the real dev-fitted values; nothing here needs to change when it lands, only callers gain a source for the numbers. |
| `imaging/preprocess.py` | `normalize(raw, cfg)` is deterministic (same bytes in → same bytes out), applies `exif_transpose`, resizes to `long_edge_px`, strips all metadata, rejects images over `max_decoded_pixels` before full decode, decodes HEIC | `tests/unit/test_preprocess.py` (6 tests, incl. a real in-memory HEIC fixture via `pillow-heif`) | `PRESTUDY_MAX_DECODED_PIXELS` is not yet a `Settings` field — callers pass `ImageConfig.max_decoded_pixels` explicitly. Wiring it into `Settings` is M2 (alongside the other `PRESTUDY_*` additions in §7.8), once there's a CLI/executor to read it from. |
| `imaging/covariates.py` | `extract(raw)` reads make/model/focal length/35mm-equivalent/dimensions without mutating or being affected by `normalize()` | `tests/unit/test_covariates.py` (5 tests, incl. missing-EXIF and EXIF-IFD-without-focal-length cases) | Nothing stores `ExifCovariates` yet — the `images` table it's destined for (§5.2) is M2. |
| `observation/schema.py` | Per-strategy strict schemas (S1/S3: `observations`+`edible_g`; S2: `observations`+`length_cm`+`max_diameter_cm`), `observations` always first, no confidence field, `validate_observation()` rejects missing/extra/wrong-typed/non-finite/negative fields | `tests/unit/test_schema.py` (18 tests) | Schema registration in the `output_schemas` table (§5.2, content-hashed) is M2. |
| `observation/prompts/*.md` | One markdown file per strategy, loaded as package data via `importlib.resources`, no numeric anchors except S3's `{prior_low_g}`/`{prior_high_g}` placeholders | Covered indirectly by `test_observation_adapter.py`'s prompt-rendering tests (S3 placeholder substitution) | Prompts are v1, not v-final — M4's dev exploration and prompt iteration may change the wording; the filename convention (`_v1_`) already reserves room for `_v2_` etc. without breaking `_PROMPT_FILENAMES`. |
| `observation/adapter.py` | `observe()` builds one user message (prompt + images in view order), selects `json_schema` vs `json_object` mode, classifies the outcome as `ok`/`invalid`/`refused`/`truncated` (incl. the reasoning-budget-exhaustion case), rejects `repair=True` and view/image-count mismatches | `tests/unit/test_observation_adapter.py` (14 tests, fake `ModelProvider`, no network) | §7.3's real signature takes M2's `configs.RunConfig`; this module defines a local `ObservationConfig` substitute instead (documented in the module docstring) — message-building/classification logic is expected to carry over unchanged, only the config source changes. Per-model `structured_output_mode` capability resolution (the `CAPABILITIES` table) is M2/M4, not enforced here — the caller picks the mode. The `error` outcome (transport failure) isn't produced here; it surfaces as `OpenRouterError` for the M2 executor to classify. |

All of M1 lint/type/test-clean: `make ci` passes (ruff format/check, mypy
--strict, pytest with the §9 85% coverage floor — actual ~98%).

Estimated test-sweep size: 7 models × 3 strategies × 90 images × 3 repeats
≈ 5,700 attempts, plus ~1,100 for secondary arms. The budget is set from
`sweep estimate`, not guessed.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Candidate unavailable under ZDR + pinned provider | smoke test before freeze; documented exception or drop |
| Provider silently changes model mid-sweep | `model_resolved` + `system_fingerprint` per attempt; analysis flags arms with >1 value |
| Reasoning models ignore temperature/seed | temperature `null` allowed per arm; seed probe in smoke |
| Test split burned by curiosity | §8 rules are constraints, not conventions |
| Too few specimens for confident ranking | 30 test specimens; ties declared from paired CIs; cost decides ties |
| Fit overfits 3 dev specimens per type | B1 reports formula error on test; S2 adopted only if it beats S1 on paired CI |
| Cost overrun | per-sweep hard budget, estimate from real calls, pause not fail |
| Results don't transfer to meals | concept A.8 sanity check on Nutrition5k before Phase 0 |

## 12. Deviations from concept Appendix A

| Appendix A (previous) | This spec | Why |
|---|---|---|
| 5 specimens/type, 2 dev / 3 test, 90 images | 8/type, 3 dev / 5 test, 144 images | within-type statistics on 3 specimens are degenerate; CIs need ~30 test specimens |
| Spearman ρ ≥ 0.7 within type | pooled log-log slope β with type fixed effects | ρ on n = 3 takes only a few values |
| Signed relative bias | geometric bias (mean log ratio) | symmetric for over/under-estimation |
| Image-level bootstrap | specimen-cluster bootstrap | images and repeats aren't independent |
| Temperature 0 required | 0 or provider default, fixed per arm | some reasoning endpoints reject 0 |
| 768 px for top-2 only | unchanged, plus reasoning and views secondary arms | RQ6 |
| `food-vision bench …` single command | `food-vision prestudy …` command group + web UI | hybrid |