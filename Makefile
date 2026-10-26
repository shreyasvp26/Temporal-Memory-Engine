.PHONY: check lint typecheck imports test-unit test-int

check: lint typecheck imports test-unit

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy packages/memory_engine/src

imports:
	uv run lint-imports

test-unit:
	uv run pytest tests/unit --disable-socket -m "not integration"

test-int:
	uv run pytest tests/integration -m "integration"