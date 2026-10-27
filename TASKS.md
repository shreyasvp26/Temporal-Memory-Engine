# TASKS.md — Atomic Execution Plan for Agents

Complete tasks strictly in order. One task = One Git commit. Never start a task until the previous task is `[x]` and `make check` passes.

## Milestone M0: Skeleton, Domain & Ports
- [x] **M0-01**: Create folder layout per `docs/LLD.md` Section 2.2 (`packages/memory_engine/src/memory_engine/...`, `apps/`, `eval/`, `tests/`). Add empty `__init__.py` files and verify `make check` runs cleanly.
- [x] **M0-02**: Implement `domain/ids.py`, `domain/enums.py` (LLD Section 4.3, 5.1) and `domain/errors.py` (LLD Section 17.1). Add unit tests in `tests/unit/test_domain_primitives.py`.
- [x] **M0-03**: Implement `domain/intervals.py` (`TimeInterval`, `TimePoint` per LLD Section 5.2) and `ports/clock.py` (`Clock`, `SystemClock`, `FixedClock` per LLD Section 4.1, 5.2). Add unit tests for UTC enforcement and half-open interval overlap/contains.
- [x] **M0-04**: Implement `domain/models.py` (`Source`, `Entity`, `Predicate`, `Fact`, `FactEvidence`, `FactRelation`, `Job`, `Message`, `ExtractedFact`, `ExtractionResult`, `Candidate` per LLD Section 5.3, 5.4, 5.7). Add unit tests verifying invariants I-1 to I-9.
- [x] **M0-05**: Implement `ports/repositories.py`, `ports/uow.py`, `ports/llm.py`, `ports/embed.py` (LLD Section 6.6, 6.7, 7.1, 7.4). Verify with `mypy --strict` and `lint-imports`.
- [x] **M0-06**: Implement `config.py` (`Settings` per LLD Section 21) and `obs/logging.py` (LLD Section 18.1).
- [x] **M0-07**: Create SQL migrations `000` through `005` in `db/migrations/` and migration runner `db/migrate.py` (LLD Section 5.5, 6.1, 6.2, 6.3, 6.4).

## Milestone M2: Pure Domain Logic (Zero IO, Sockets Disabled)
- [x] **M2-01**: Implement `ingest/gate.py` (`IngestGate`) and `ingest/windowing.py` (`NoteChunker`) per LLD Section 8.1. Pass `T-ING` unit tests.
- [x] **M2-02**: Implement `ingest/time_resolver.py` (`TimeResolver` grammar rows 1-14 per LLD Section 8.5). Pass `T-TIME` unit tests (including Oct 2026 -> Mar 2026 and Mar 2026 -> Mar 2025 anchors).
- [x] **M2-03**: Implement `ingest/conflict/plan.py`, `registry.py`, `single.py`, `multi.py`, `event.py` (`ConflictResolver` & strategies per LLD Section 8.8). Pass `T-CR-01` through `T-CR-15` (including `T-CR-05` backfill test and Hypothesis property test G-1).
- [x] **M2-04**: Implement `retrieve/planner/cues.py`, `intent.py`, `rules.py` (`RulesPlanner` per LLD Section 9.2) and `retrieve/lanes/filter.py` (`TemporalFilter` per LLD Section 9.3). Pass `T-PLAN` and `T-RET` filter tests.
- [x] **M2-05**: Implement `retrieve/fusion.py` (`RrfFuser`), `retrieve/decay.py` (`DecayScorer`), `retrieve/assembler.py` (`ContextAssembler`), and `retrieve/abstain.py` (`AbstentionPolicy`) per LLD Section 9.4–9.6.

## Milestone M3: PostgreSQL Persistence, SQL Catalog & LLM/Embedder Adapters
- [x] **M3-01**: Implement `db/pool.py`, `db/tx.py`, and `db/uow.py` (`PostgresUnitOfWork` & `PostgresUnitOfWorkFactory` with `SET LOCAL app.user_id`, `lock_user` advisory lock, and vector codec registration per LLD Section 6.5, 6.7). Add unit tests with mock/fake asyncpg connections.
- [x] **M3-02**: Implement `db/sql.py` containing all frozen SQL queries `Q-01` through `Q-18` (including CR-01, CR-02, CR-03 updates from LLD Section 6.8) and row-to-domain hydration helpers in `db/mappers.py`.
- [x] **M3-03**: Implement `db/repos/sources.py`, `db/repos/entities.py`, `db/repos/predicates.py`, `db/repos/messages.py`, `db/repos/audit.py`, and `db/repos/profile.py` (LLD Section 6.6, 6.8). Add unit tests verifying SQL parameter binding and row mapping.
- [x] **M3-04**: Implement `db/repos/facts.py`, `db/repos/search.py` (with `SET LOCAL hnsw.iterative_scan = 'relaxed_order'`), and `db/repos/jobs.py` (outbox claim `Q-01`, heartbeat `Q-02`, complete/fail `Q-03`, coalesced upsert `Q-04` per LLD Section 6.6, 6.8).
- [x] **M3-05**: Implement `llm/gateway.py`, `llm/prompts.py`, `llm/schemas.py`, and provider clients (`llm/groq.py`, `llm/gemini.py`, `llm/ollama.py`) plus `embed/fastembed_adapter.py` and `embed/cached.py` per LLD Section 7.1–7.5. Add unit/contract tests (`tests/contract/test_llm_and_embed.py`) with mocked HTTP/socket-free transports.

## Milestone M4: Ingestion Pipeline, Entity Linking & Worker Loop
- [x] **M4-01**: Implement `ingest/entity_linker.py` (`EntityLinker` 5-step cascade: self pronouns, exact normalized name, fuzzy trigram/cosine >= 0.92, create new entity per LLD Section 8.6) and `ingest/canonicalizer.py` (`Canonicalizer` predicate mapping & text rendering per LLD Section 8.7).
- [x] **M4-02**: Implement `ingest/extractor.py` (`FactExtractor` window prompt builder, JSON repair retry, and evidence span validation per LLD Section 8.3–8.4) and `ingest/plan_executor.py` (executing `ResolutionPlan` `Op` commands against `FactRepository` under `uow.lock_user` per LLD Section 8.8–8.9).
- [x] **M4-03**: Implement `ingest/service.py` (`IngestService`: `enqueue_turn`, `accept_note`, and 4-phase out-of-transaction `run_extract_job` Phase A/B/C/D per LLD Section 8.1, 8.2) and `apps/worker/main.py` (`WorkerRunner` poll/heartbeat/reaper loop per LLD Section 10). Add end-to-end unit tests using fake in-memory UoW/Repositories (`tests/unit/test_ingest_pipeline.py`).

## Milestone M5: Retrieval Orchestrator, Lifecycle & FastAPI Server
- [x] **M5-01**: Implement `retrieve/lanes/vector.py`, `fts.py`, `graph.py`, `temporal.py` and `retrieve/orchestrator.py` (`RetrievalOrchestrator` parallel lane execution, fault tolerance when 1 lane fails, RRF fusion, decay, and context assembly per LLD Section 9.1, 9.3–9.6).
- [x] **M5-02**: Implement `retrieve/chat_service.py` (`ChatService` statement-only bypass, retrieval + streaming LLM answer, citation extraction, and turn enqueueing per LLD Section 9.1) and `lifecycle/service.py` (`MemoryLifecycleService`: manual edit, retract, delete, GDPR purge, export per LLD Section 12).
- [ ] **M5-03**: Implement `apps/api/` (`main.py`, `deps.py`, `auth.py`, `rate_limit.py`, `errors.py`, and routes for `/v1/chat`, `/v1/notes`, `/v1/facts`, `/v1/jobs`, `/v1/profile`, `/healthz`, `/readyz` per LLD Section 11). Add API unit/contract tests using `httpx.AsyncClient` (`tests/unit/test_api_routes.py`).