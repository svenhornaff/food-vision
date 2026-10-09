# food-vision

Photo meal logging with database-grounded nutrition estimates. See
[`docs/dev/food-vision-concept.md`](docs/dev/food-vision-concept.md) for
architecture, roadmap and rationale, and
[`docs/dev/boilerplate.md`](docs/dev/boilerplate.md) for local setup, dev
workflow and what's implemented today vs. planned.

## Status

Early prototype. Implemented so far: settings (`src/food_vision/config/`),
logging (`src/food_vision/utils/`), and the OpenRouter transport layer
(`src/food_vision/proxy/`). No API, CLI, domain model or reference store
yet — see the boilerplate doc's checklist for what's next.

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY at minimum
make test
```

## Dev workflow

```bash
make lint     # ruff format --check, ruff check, mypy
make format   # ruff format + fix (mutating)
make test     # pytest, coverage floor 60%
make ci       # lint + test
```
