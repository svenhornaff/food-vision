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
make test     # pytest, coverage floor 85%
make ci       # lint + test
```

## Dataset (pre-study)

The lean pre-study (`docs/dev/pre-study.md`) uses the ECUSTFD dataset via
its [Hugging Face mirror](https://huggingface.co/datasets/ai5labsOfficial/ecustfd)
(fetched automatically by `food_vision.prestudy.ecustfd.prepare`, cached
under `~/.cache/huggingface/`). CC-BY-4.0; cite Liang & Li,
[arXiv:1705.07632](https://arxiv.org/abs/1705.07632).

Some one-off analysis scripts (`bench/scripts/`) need the original
per-image XML annotations, which the HF mirror flattens into a single CSV.
For those, clone the [source repo](https://github.com/Liang-yc/ECUSTFD-resized-)
locally (not tracked by git — ~140MB, gitignored under `/data/raw/`):

```bash
git clone --depth 1 -b master https://github.com/Liang-yc/ECUSTFD-resized-.git data/raw/ecustfd
rm -rf data/raw/ecustfd/.git
```
