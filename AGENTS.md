# AGENTS.md — Constitutional Rules for Coding Agents

You are implementing the **Temporal Memory Engine** (`second-brain`), governed strictly by `docs/LLD.md` (Frozen Baseline v1.0) and `docs/HLD.md`.

## 1. Non-Negotiable Architecture & Code Rules
1. **Zero Spec Drift:** Never invent new class names, method signatures, enum values, config keys, or SQL schemas. Every public interface, dataclass, and SQL query is `[FROZEN]` in `docs/LLD.md`.
2. **Strict Import Boundaries (enforced by `import-linter`):**
   - `memory_engine.domain` imports NOTHING from the project (pure Python stdlib only).
   - `memory_engine.ports` imports ONLY from `memory_engine.domain`.
   - `memory_engine.ingest`, `retrieve`, `lifecycle` import ONLY from `domain` and `ports` (NEVER `db`, `llm`, or `embed` adapters).
   - `memory_engine.db`, `llm`, `embed` import ONLY from `domain` and `ports`.
   - Only `apps/*` and `eval/*` wire concrete adapters together.
3. **Code Style & Types:**
   - Python 3.12, full type annotations everywhere (`mypy --strict` must pass with 0 errors).
   - Single quotes for strings (`ruff format`), private attributes/methods prefixed with `_` (ruff `SLF001`).
   - All domain entities and value objects use `@dataclass(frozen=True, slots=True)`.
   - All boundary models (API, LLM schemas) use Pydantic v2 with `model_config = ConfigDict(extra='forbid', frozen=True)`.
4. **Time & IO Discipline:**
   - Never call `datetime.now()` or `datetime.utcnow()`. Always use the injected `Clock` port.
   - All `datetime` objects must be timezone-aware UTC (`tzinfo is not None`), or raise `InvariantViolation`.
   - Never run network/LLM/embedding calls inside a database write transaction (`UnitOfWork`).
   - Unit tests in `tests/unit/` must run with network sockets disabled (`--disable-socket`).

## 2. How You Must Work on Every Task
1. **Read Only What You Need:** Read the specific section/lines of `docs/LLD.md` referenced in the prompt and `TASKS.md`. Do not scan the entire repo unnecessarily.
2. **Small Blast Radius:** Touch at most 1–3 files per turn.
3. **Mandatory Self-Verification:** Before declaring any task complete, run:
   - `uv run ruff check --fix . && uv run ruff format --check .`
   - `uv run mypy packages/memory_engine/src apps eval tests`
   - `uv run lint-imports`
   - `uv run pytest tests/unit -m "not integration"`
4. **Update State:** Only check off a task in `TASKS.md` when all 4 verification commands exit with code `0`.