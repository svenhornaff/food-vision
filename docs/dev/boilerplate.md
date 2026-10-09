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
  constructs a client that needs them. This was a real bug fixed in this
  refactor (the previous `Settings()` instantiated itself at module import
  time) and is worth keeping as a hard rule.
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
  `import food_vision` resolves to `src/food_vision/` without any `PYTHONPATH`
  or `sys.path` hacking — note the previous `Settings.__init__` inserted
  `PROJECT_ROOT` into `sys.path` manually; that's gone, it was never needed.

## 3. uv — dependencies

```bash
# core (today: OpenRouter proxy + settings only)
uv add pydantic pydantic-settings openai httpx

# dev group
uv add --dev pytest pytest-cov ruff mypy
```

Lock rule: **`uv.lock` committed** — reproducible builds, same as any
application (not a published library).

**Dropped from the previous dependency set, and why:**

| Dropped | Reason |
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
│       ├── proxy/                # OpenRouter transport layer (this phase)
│       │   ├── __init__.py
│       │   ├── openrouter.py     # OpenRouterClient, ModelProvider Protocol
│       │   └── model_provider.py # get_default_provider() convenience factory
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
│       └── test_model_provider.py
└── docs/
    └── dev/
        ├── food-vision-concept.md
        └── boilerplate.md        # this file
```

**Not built yet** (concept doc §13's full layout — `api/`, `domain/`,
`observation/`, `enrichment/`, `matching/`, `reference/`, `imaging/`,
`pipeline/`, `cli.py`, `bench/`, `data/`): these are later phases (Phase 0
onward in the concept's roadmap §14) and intentionally don't exist yet.
Adding them before there's a reason to is the "over-engineering before
evidence" risk the concept doc itself calls out in §15.

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
           "--cov=src/food_vision", "--cov-report=term-missing", "--cov-fail-under=60"]
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
- `--cov-fail-under=60` is a floor, not a target — raise it as real
  coverage grows past the current ~97% (easy right now because the surface
  area is small; don't let it regress as the domain/API layers land).
- No `[project.scripts]` entry point. The previous `pyproject.toml`
  declared `food-vision = "food_vision:main"` but `main` never existed —
  a broken console script. The concept doc's `typer`-based CLI
  (`food-vision analyze|bench|build-reference`) is Phase 0+ work; it'll get
  its own `cli.py` and script entry when it exists, not before.
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
  function, not a module-level singleton constructed at import time. This
  is the one change in this refactor worth over-explaining: the previous
  version ran `settings = Settings()` at module scope, so *any* import of
  this module (including from a test that doesn't care about OpenRouter)
  raised `RuntimeError` if `OPENROUTER_API_KEY` wasn't in the environment.
  `get_settings()` defers that to first call.
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
  calls are a no-op once the root logger has handlers). This follows the
  pattern in `../bulliexplorer/app/utils/log_factory.py` rather than the
  previous `LoggingFactory` singleton class, which unconditionally created
  a `logs/` directory and a `RotatingFileHandler` on every `get_logger()`
  call — a side effect on *every import*, including in tests, and it wiped
  the root logger's existing handlers in `__new__`.
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
  yet. The concept doc's `tests/{unit,property,integration}` layout (§13)
  is the target once the domain/calculator and API layers exist.
- All OpenRouter tests are mocked at `OpenRouterClient._client` — no
  network I/O, no real API key required to run the suite.
- Minimum bar covered today: settings import-safety + allowlist
  validation, logging idempotency/no-filesystem-side-effects, and the
  proxy's request-building, routing-mode, retry/backoff and telemetry
  behavior (see `tests/unit/test_openrouter.py` for the full list:
  deterministic request shape, allowlist rejection, production vs
  benchmark provider-routing payloads, 429/5xx-only retries capped at
  `OPENROUTER_MAX_RETRIES`, non-retryable 4xx failing immediately, and
  `cost_is_estimate` flagging when OpenRouter doesn't report a cost).
- Run: `make test` (`uv run pytest`, coverage floor 60%, currently ~97%).

---

## 8. OpenRouter proxy — what it actually does today

Scoped strictly to concept doc §9's request/operational requirements —
nothing about meal observation, prompts, or schemas (that's
`observation/adapter.py`, a later phase):

- `OpenRouterClient` wraps `openai.OpenAI(base_url=..., api_key=...)`
  directly; the SDK's own retry logic is disabled (`max_retries=0`) so the
  adapter's own policy is the only one in effect.
- Every request sets `temperature` from settings (default `0`), an
  optional fixed `seed`, and an OpenRouter `provider` block with
  `require_parameters`, `data_collection` (`"deny"` by default) and `zdr`.
- `RoutingMode.PRODUCTION` uses the configured provider order with
  `allow_fallbacks` from settings; `RoutingMode.BENCHMARK` requires an
  explicit `provider_pin` + `quantizations` and forces
  `allow_fallbacks: false` — reproducibility for benchmark runs per §9.
- Retries only on HTTP 429/5xx, capped at `OPENROUTER_MAX_RETRIES` (default
  2 retries → 3 attempts total), with a jittered exponential backoff
  (`sleep`/`random_fn` are constructor-injectable for tests).
- Connect/total timeouts default to 5s/30s (`httpx.Timeout`), per §9.
- `response_format` accepts a caller-supplied `JsonSchemaFormat` (name +
  JSON Schema + `strict`); the adapter has no opinion on what schema that
  is — no observation schema lives here.
- Usage/cost telemetry (`Usage`) is extracted from OpenRouter's response
  and logged; `cost_is_estimate=True` when OpenRouter didn't report a cost,
  so a caller never silently treats a missing cost as zero.
- `ModelProvider` is a `Protocol`, not a base class — a future
  `observation/adapter.py` implementation can depend on this Protocol
  without importing `OpenRouterClient` directly, matching the concept's
  "adapter Protocol + OpenRouter implementation" package design (§13)
  without building the rest of that package yet.

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
[x] tests/unit/: settings, logging, proxy — ~97% coverage, no network I/O
[x] .gitignore + .env.example + this boilerplate doc
[x] Makefile: build-env/format/lint/test/clean/ci
[ ] domain/ (MealAnalysis, Item, Observation, calculator) — Phase 0/1
[ ] observation/ (Protocol + OpenRouter implementation using proxy/,
    schema.py, prompts/) — Phase 0
[ ] reference/ (BLS 4.0 + FDC + OFF → reference.db) — Phase 0
[ ] cli.py (typer: analyze, bench, build-reference) — Phase 0
[ ] api/ (FastAPI routes) — Phase 1
[ ] enrichment/, matching/ — Phase 1
[ ] bench/ harness — Phase 2
```

Each unchecked item is a concept-doc phase (§14) with its own exit
criteria — don't pull one forward without the evidence/dependency that
phase's exit criteria calls for.
