install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest

precommit:
	uv run pre-commit run --all-files

run:
	uv run uvicorn urban_ml.api.main:app --reload
