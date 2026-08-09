.PHONY: sync lint format typecheck test check

sync:
	uv sync --group dev

lint:
	uv run ruff check src

format:
	uv run ruff format src

typecheck:
	uv run ty check src

test:
	uv run pytest

check: lint typecheck
