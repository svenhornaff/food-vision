# food-vision — Project Boilerplate

> Adapted from the general Python project boilerplate for this specific
> project. See `food-vision-concept.md` for architecture, roadmap and the
> full planned package layout; this doc covers local project setup, dev
> workflow and what's actually implemented *today*.
> Target: macOS + `uv` + Python 3.12 (3.13 acceptable per the concept doc;
> `.python-version` currently pins 3.12.13).

**Scope note — read this before assuming more exists than does.** The
concept doc describes a full system: FastAPI API, Pydantic domain model,
observation schema + prompts, enrichment, matching, a local reference
store (BLS/FDC/OFF), a CLI, a benchmark harness. **None of that is built
yet.** What exists today is the OpenRouter transport layer
(`src/food_vision/proxy/`), settings (`src/food_vision/config/`), and
logging (`src/food_vision/utils/`) — the plumbing every later phase will
sit on top of. This boilerplate doc is scoped to that reality; sections
below that describe future work say so explicitly rather than implying
it's already there.

---

## 0. Principles

- **One tool for the toolchain**: `uv` replaces `pyenv`, `virtualenv`,
  `pip`, `pip-tools`, `pipx`. No exceptions.
- **`uv sync` is the only setup command** after cloning. Everything derives
  from `pyproject.toml` + `uv.lock`.
- **`.venv` lives in the project root** — discovered automatically by uv,
  VS Code, pyright/pyright-based tooling.
- **Settings are import-safe, instantiation is lazy.** Importing
  `food_vision.config.settings` must never raise just because
  `OPENROUTER_API_KEY` isn't set in the current shell — credentials are
  only required when something actually calls `get_settings()` /
  constructs a client that needs them. Keep this as a hard rule.
- **Centralized logging from day one** — one `configure_logging()` call at
  startup, `get_logger(__name__)` everywhere else, no scattered
  `logging.basicConfig()` calls, no side effects (no forced file handler,
  no forced `logs/` directory) just from importing a module.
- **No agent framework.** Per the concept doc §5: OpenRouter is called
  directly via the `openai` SDK with a `base_url` override. LangChain was
  removed from this project for exactly this reason — it added a
  multi-provider/embeddings abstraction layer nothing here needs, and the
  concept doc explicitly calls for a thin adapter instead.
- **Protocol-based provider seam, not a cache/registry.** `proxy/openrouter.py`
  defines a `ModelProvider` Protocol so a second provider (or the future
  `observation/adapter.py`) can be added by implementing the Protocol, not
  by growing a class-level cache of provider instances.

---

## 1. uv — bootstrap

```bash
# one-time machine setup
brew install uv
uv python install 3.12

# project (already scaffolded — repeated here for reference)
uv python pin 3.12            # writes .python-version — committed
```

## 2. uv — virtual environment

```bash
uv sync                       # reads .python-version, creates .venv/, installs everything
```

- No activation needed: `uv run <cmd>` runs inside `.venv` automatically.
- `.venv/` is always gitignored — disposable, `rm -rf .venv && uv sync`
  rebuilds identically.
- The package installs in editable mode (`uv_build` + `src` layout), so
  `import food_vision` resolves to `src/food_vision/` without any
  `PYTHONPATH` or `sys.path` hacking.

## 3. uv — dependencies

```bash
# core (today: OpenRouter proxy + settings only)
uv add pydantic pydantic-settings openai httpx

# dev group
uv add --dev pytest pytest-cov ruff mypy
```

Lock rule: **`uv.lock` committed** — reproducible builds, same as any
application (not a published library).

**Deliberately not a dependency, and why:**

| Not used | Reason |
|---|---|
| `langchain`, `langchain-core`, `langchain-community`, `langchain-openai` | Concept doc §5: "AI gateway: OpenRouter via `openai` client with `base_url` override; thin adapter; no agent framework." LangChain was pulling in an agent-framework abstraction (multi-provider chat model base classes, embeddings, tool-calling scaffolding) that nothing in the concept plan uses. Calling OpenRouter's OpenAI-compatible endpoint directly with the `openai` SDK is both what the concept asks for and strictly less code. |

---

## 4. Project structure (current, not the full concept layout)

```
food-vision/
├── .python-version
├── .venv/                        # gitignored
├── pyproject.toml
├── uv.lock                       # committed
├── Makefile
├── .env.example                  # committed; .env is not
├── src/
│   └── food_vision/
│       ├── __init__.py
│       ├── py.typed
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py       # pydantic-settings, lazy get_settings()
│       ├── proxy/                # OpenRouter transport layer
│       │   ├── __init__.py
│       │   ├── openrouter.py     # OpenRouterClient, ModelProvider Protocol
│       │   └── model_provider.py # get_default_provider() convenience factory
│       ├── domain/               # pure, deterministic calculations
│       │   ├── __init__.py
│       │   └── calculator.py     # S2/B1 mass+kcal formulas (no fitted constants baked in)
│       ├── imaging/              # decode/normalize; EXIF covariates
│       │   ├── __init__.py
│       │   ├── preprocess.py     # normalize(): HEIC decode, exif_transpose, resize, strip
│       │   └── covariates.py     # extract(): make/model/focal length, read before stripping
│       ├── observation/          # prompts, schemas, observe() adapter
│       │   ├── __init__.py
│       │   ├── schema.py         # per-strategy strict JSON schemas + validation
│       │   ├── adapter.py        # observe(): message build, mode select, outcome classify
│       │   └── prompts/          # fruit_s{1,2,3}_v1_en.md, loaded as package data
│       └── utils/
│           ├── __init__.py
│           └── log_factory.py    # configure_logging() / get_logger()
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       ├── test_settings.py
│       ├── test_log_factory.py
│       ├── test_openrouter.py
│       ├── test_model_provider.py
│       ├── test_calculator.py
│       ├── test_preprocess.py
│       ├── test_covariates.py
│       ├── test_schema.py
│       └── test_observation_adapter.py
└── docs/
    └── dev/
        ├── food-vision-concept.md
        ├── pre-study-web-ui.md    # implementation spec for the current phase (Phase P / M1–M6)
        ├── pre-study-implementation.md  # superseded by the above; kept for history
        └── boilerplate.md        # this file
```

**Built so far: M1 only** (docs/dev/pre-study-web-ui.md §10 milestones).
M1's exit criterion — "`observe()` returns a classified `Observation` from
a recorded fixture" — is met: `domain/calculator.py`, `imaging/`,
`observation/` all exist, are fully unit-tested with no network I/O, and
`proxy/openrouter.py` carries the §7.4 telemetry extension (`generation_id`,
`model_resolved`, `system_fingerprint`, `native_finish_reason`,
`reasoning_text`/`reasoning_tokens`, `cached_tokens`, `raw`, plus `top_p`/
`max_tokens`/`reasoning` on `complete()`).

**Not built yet** (M2–M6, same doc): `prestudy/` (db, dataset, fits,
configs, protocols, sweeps, executor, analysis, charts, report, web UI),
`cli.py`. Also still deferred to Phase 0+ (concept §13): `api/`,
`enrichment/`, `matching/`, `reference/`, `pipeline/`. Adding any of these
before there's a reason to is the "over-engineering before evidence" risk
the concept doc calls out in §15.

---

## 5. `pyproject.toml` — current configuration

```toml
[project]
name = "food-vision"
requires-python = ">=3.12"
dependencies = [
  "pydantic>=2.11.7,<3.0.0",
  "pydantic-settings>=2.10.1,<3.0.0",
  "openai>=1.60.0",
  "httpx>=0.27.0",
  "pillow>=11,<12",
  "pillow-heif>=0.18",
  "numpy>=2.0",
]

[dependency-groups]
dev = ["pytest>=8.3.0", "pytest-cov>=5.0.0", "ruff>=0.9.0", "mypy>=1.13.0"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "N"]

[tool.mypy]
strict = true
exclude = ["tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--tb=short", "-p", "no:logging",
           "--cov=src/food_vision", "--cov-report=term-missing", "--cov-fail-under=85"]
```

Notes:

- `-p no:logging` in `addopts` disables pytest's own live-log-capturing
  plugin. Without it, pytest attaches its own `LogCaptureHandler` to the
  root logger around every test phase, which fights with
  `configure_logging()`'s "don't touch an already-configured root logger"
  guard and makes the logging tests observe pytest's handler instead of
  ours. If a future contributor wants `caplog` for some other test, scope
  `-p logging` back on for just that module via `@pytest.mark.parametrize`
  or a dedicated `pytest.ini` marker — don't remove the global flag without
  re-checking `tests/unit/test_log_factory.py`.
- `--cov-fail-under=85` is pre-study-web-ui.md §9's floor for
  `prestudy/`/`observation/`/`imaging/`/`domain/`; actual coverage is
  currently ~98% (one global floor, not split per package —
  `pytest-cov` doesn't make a per-path floor convenient, and the whole
  codebase is small enough that splitting it wouldn't buy much yet).
- No `[project.scripts]` entry point yet. The concept doc's `typer`-based
  CLI (`food-vision analyze|bench|build-reference`) is Phase 0+ work;
  it'll get its own `cli.py` and script entry when it exists, not before.
- `pyrightconfig` lives inline as `[tool.pyright]` (`venvPath = "."`,
  `venv = ".venv"`) so editor/LSP tooling resolves the project's own
  virtualenv instead of whatever interpreter happens to be first on `PATH`.

**Dropped from a more elaborate general boilerplate, and why (same spirit
as the etf-portfolio boilerplate doc's own table):**

| Dropped | Reason |
|---|---|
| `bandit`, `detect-secrets`, `pip-audit`, `pre-commit` | No web-facing API, no DB, no auth surface exists yet — there's nothing for a SAST/secrets/CVE scan to meaningfully cover beyond what `ruff` already catches in a ~190-line codebase. Revisit when the FastAPI/API layer (concept §7) lands and there's a real attack surface. |
| SBOM / license-gate / complexity-gate tooling | Same reasoning as etf-portfolio's boilerplate: audit-evidence tooling for CRA/BSI-regulated commercial software, not relevant to a prototype. |
| `httpx`/`respx`-based integration tests against a live OpenRouter | The concept doc's own test strategy (§13: "pytest, pytest-asyncio, respx, hypothesis") plans this, but it's for testing a full `observation/adapter.py` against realistic HTTP responses. Today's unit tests mock at the `OpenRouterClient._client` seam instead, which is simpler and sufficient for a transport-layer-only phase; add `respx`-based tests when there's an HTTP-shaped integration surface worth exercising end to end. |

---

## 6. Config & logging

- **`src/food_vision/config/settings.py`**: one `Settings` class
  (`pydantic-settings`), accessed via `get_settings()` — an `lru_cache`'d
  function, not a module-level singleton constructed at import time, so
  importing this module never fails just because `OPENROUTER_API_KEY`
  isn't set; validation happens once, on first `get_settings()` call.
- Settings today cover exactly what `proxy/` needs: OpenRouter
  credentials/endpoint, the `FOOD_VISION_MODELS` allowlist + default/
  fallback models (concept §9), provider-routing flags
  (`require_parameters`, `data_collection`, `zdr`, `allow_fallbacks`,
  `provider_order`), determinism (`temperature`, `seed`), timeouts/retries,
  and `LOG_LEVEL`/`LOG_JSON`. Nothing else — no DB URL, no auth, because
  none of that exists yet.
- **`src/food_vision/utils/log_factory.py`**: `configure_logging(*,
  json_output=False, level=logging.INFO)` once at process startup (a
  future CLI's `main()`, a future FastAPI `lifespan`, or a test fixture);
  `get_logger(__name__)` everywhere else. No class, no singleton, no
  filesystem side effects, idempotent (repeated `configure_logging()`
  calls are a no-op once the root logger has handlers) — the pattern in
  `../bulliexplorer/app/utils/log_factory.py`.
- **Never log secrets, prompts, images or meal content** (concept §11).
  `OpenRouterClient.complete()` logs model, provider, routing mode, attempt
  count, latency, token counts and cost only — never message content,
  never the API key.
- Skipped for now, same call as etf-portfolio's boilerplate: `ContextVar`-
  based trace-ID correlation middleware. Nothing here is a concurrent
  request-serving process yet.

---

## 7. Testing

- `tests/unit/` only, for now — no `integration/` directory because there's
  nothing requiring a live dependency (DB, HTTP server) to test against
  yet, and no `property/` directory because the calculator's property
  checks (monotonicity, positivity) are written as parametrized `pytest`
  cases rather than `hypothesis` so far — add `hypothesis` when M2's
  `fits.py` needs it, not before. The concept doc's
  `tests/{unit,property,integration}` layout (§13) is still the target
  once the API layer exists.
- All OpenRouter tests are mocked at `OpenRouterClient._client`; all
  `observation.adapter.observe()` tests use a fake `ModelProvider` — no
  network I/O, no real API key required to run the suite, matching
  pre-study-web-ui.md §9's "no live network calls" rule.
- Minimum bar covered today: settings import-safety + allowlist
  validation, logging idempotency/no-filesystem-side-effects, the proxy's
  request-building/routing-mode/retry/backoff/telemetry behavior
  (`test_openrouter.py`), image normalization determinism + EXIF
  orientation + metadata stripping + the decoded-pixel guard + a real HEIC
  fixture (`test_preprocess.py`), EXIF covariate extraction without
  mutating the source bytes (`test_covariates.py`), per-strategy schema
  validation (`test_schema.py`), and `observe()`'s message shape, mode
  selection and `ok`/`invalid`/`refused`/`truncated` outcome
  classification including the reasoning-budget-exhaustion case
  (`test_observation_adapter.py`). Pure-formula monotonicity/positivity
  checks for `domain/calculator.py` (`test_calculator.py`).
- Run: `make test` (`uv run pytest`, coverage floor 85%, currently ~98%).

---

## 8. OpenRouter proxy — what it actually does today

Scoped strictly to concept doc §9's request/operational requirements —
nothing about meal observation, prompts, or schemas (that's
`observation/adapter.py`, a later phase):

- `OpenRouterClient` wraps `openai.OpenAI(base_url=..., api_key=...)`
  directly; the SDK's own retry logic is disabled (`max_retries=0`) so the
  adapter's own policy is the only one in effect.
- Every request sets `temperature` and `seed` from settings by default,
  each overridable per call (`complete(temperature=..., seed=...)`) —
  the pre-study harness varies these per run without env-file edits — and
  an OpenRouter `provider` block with `require_parameters`,
  `data_collection` (`"deny"` by default) and `zdr`.
- `RoutingMode.PRODUCTION` uses the configured provider order with
  `allow_fallbacks` from settings; `RoutingMode.BENCHMARK` takes a
  `RoutingPolicy` (`provider_pin` required, `quantizations`/`zdr`/
  `data_collection` optional overrides) and forces `allow_fallbacks: false`
  — reproducibility for benchmark runs per §9. `quantizations` is
  deliberately optional: several candidates (Gemini/GPT/Claude on
  OpenRouter) publish no quantization label at all, so requiring one would
  silently exclude them rather than do nothing.
- Every request sets `extra_body.usage.include = true` so OpenRouter
  reports actual provider cost on the response; without it `usage.cost` is
  absent on most requests and every row would be flagged
  `cost_is_estimate=True` regardless of what actually happened.
- `CompletionResult.effective_routing_policy` carries the exact `provider`
  block that was sent, so a caller that varies routing per call can record
  what was effective alongside the result instead of reconstructing it
  from inputs later.
- Retries only on HTTP 429/5xx, capped at `OPENROUTER_MAX_RETRIES` (default
  2 retries → 3 attempts total), with a jittered exponential backoff
  (`sleep`/`random_fn` are constructor-injectable for tests).
- Connect/total timeouts default to 5s/30s (`httpx.Timeout`), per §9.
- `response_format` accepts a caller-supplied `JsonSchemaFormat` (name +
  JSON Schema + `strict`); the adapter has no opinion on what schema that
  is — no observation schema lives here.
- Usage/cost telemetry (`Usage`) is extracted from OpenRouter's response
  and logged; `cost_is_estimate=True` only when OpenRouter still didn't
  report a cost despite `usage.include`, so a caller never silently treats
  a missing cost as zero.
- `ModelProvider` is a `Protocol`, not a base class — `observation/adapter.py`
  depends on this Protocol (and a test fake implementing it) without
  importing `OpenRouterClient` directly, matching the concept's "adapter
  Protocol + OpenRouter implementation" package design (§13).
- `complete()` also accepts `top_p`, `max_tokens` and `reasoning` (a dict
  forwarded as OpenRouter's `reasoning` block), all `None`/omitted by
  default; `response_format` accepts `JsonObjectFormat` as well as
  `JsonSchemaFormat`, for models without strict-schema support.
  `CompletionResult` carries `generation_id`, `model_resolved`,
  `system_fingerprint`, `native_finish_reason`, `reasoning_text`/
  `reasoning_tokens`, `cached_tokens` and the full `raw` response dict
  (pre-study-web-ui.md §7.4) — all extracted defensively (`getattr(...,
  None)`) so minimal/mocked responses never raise, and all default to
  `None`/`{}` so existing callers that construct a bare `CompletionResult`
  don't break.

---

## 9. Makefile

```makefile
SRC := src tests

build-env:  ## uv sync
format:     ## ruff format + fix (mutating)
lint:       ## ruff format --check, ruff check, mypy src
test:       ## uv run pytest (coverage floor via addopts)
clean:      ## remove caches/build artefacts
ci:         ## lint + test
```

**Dropped from a more elaborate general boilerplate:** `security`,
`db-upgrade`, `db-revision`, `docs-serve`/`docs-build`, `dev` (no DB, no
Alembic, no FastAPI app, no docs site exist yet — see §5's dropped-tooling
table for the reasoning on the first three).

---

## 10. New-project checklist (for the *next* phase, not re-doing this one)

```
[x] uv sync → .venv/ created, src-layout editable install works
[x] pyproject.toml: dependency groups + ruff/mypy/pytest config
[x] src/food_vision/: config/settings.py (lazy), utils/log_factory.py,
    proxy/ (OpenRouterClient + ModelProvider Protocol)
[x] tests/unit/: settings, logging, proxy — no network I/O
[x] .gitignore + .env.example + this boilerplate doc
[x] Makefile: build-env/format/lint/test/clean/ci
[x] M1 (pre-study-web-ui.md §10): proxy/openrouter.py §7.4 telemetry
    extension, domain/calculator.py, imaging/ (preprocess + covariates),
    observation/ (schema + prompts + adapter) — 98% coverage, no network I/O
[ ] M2: prestudy/{db,dataset,fits,configs,protocols,sweeps,executor}.py,
    cli.py — dev sweep of 1 model x S1 runs, is killed, resumes, completes
    within budget
[ ] M3: capture fruit-v1 (144 images; can run in parallel from day 1)
[ ] M4: smoke, dev exploration, prompt iteration, fits
[ ] M5: prestudy/{analysis,charts,report}.py, prestudy/web/ — analyze
    reproduces fixtures; probe + attempt review usable
[ ] M6: protocol freeze → test sweep → analyze → decision.md
[ ] reference/ (BLS 4.0 + FDC + OFF → reference.db) — Phase 0 (concept §14)
[ ] api/ (FastAPI routes) — Phase 1
[ ] enrichment/, matching/ — Phase 1
```

Each unchecked item is either a pre-study milestone
(docs/dev/pre-study-web-ui.md §10) or a concept-doc phase (§14) with its
own exit criteria — don't pull one forward without the evidence/dependency
that milestone/phase calls for.
