all: lint format typecheck test precommit check-unused-vars check-unused-code

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

check-unused-vars:
	uv run ruff check src scripts tests --select F401,F841

check-unused-code:
	uv run vulture src scripts tests --min-confidence 80

ingest-gbfs:
	uv run python scripts/ingest_gbfs.py

run:
	uv run uvicorn urban_ml.api.main:app --reload
