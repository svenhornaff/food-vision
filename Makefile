SRC := src tests

.PHONY: build-env
build-env: ## Create .venv + install all dependencies
	uv sync

.PHONY: format
format: ## Format code (mutating)
	uv run ruff format $(SRC)
	uv run ruff check $(SRC) --fix

.PHONY: lint
lint: ## Lint + types (report only)
	uv run ruff format $(SRC) --check
	uv run ruff check $(SRC)
	uv run mypy src

.PHONY: test
test: ## Run tests (coverage floor via addopts)
	uv run pytest

.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +

.PHONY: ci
ci: lint test ## Full CI pipeline
