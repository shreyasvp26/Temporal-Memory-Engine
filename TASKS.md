# TASKS.md — Atomic Execution Plan for Agents

Complete tasks strictly in order. One task = One Git commit. Never start a task until the previous task is `[x]` and `make check` passes.

## Milestone M0: Skeleton, Domain & Ports
- [x] **M0-01**: Create folder layout per `docs/LLD.md` Section 2.2 (`packages/memory_engine/src/memory_engine/...`, `apps/`, `eval/`, `tests/`). Add empty `__init__.py` files and verify `make check` runs cleanly.
- [x] **M0-02**: Implement `domain/ids.py`, `domain/enums.py` (LLD Section 4.3, 5.1) and `domain/errors.py` (LLD Section 17.1). Add unit tests in `tests/unit/test_domain_primitives.py`.
- [x] **M0-03**: Implement `domain/intervals.py` (`TimeInterval`, `TimePoint` per LLD Section 5.2) and `ports/clock.py` (`Clock`, `SystemClock`, `FixedClock` per LLD Section 4.1, 5.2). Add unit tests for UTC enforcement and half-open interval overlap/contains.
- [x] **M0-04**: Implement `domain/models.py` (`Source`, `Entity`, `Predicate`, `Fact`, `FactEvidence`, `FactRelation`, `Job`, `Message`, `ExtractedFact`, `ExtractionResult`, `Candidate` per LLD Section 5.3, 5.4, 5.7). Add unit tests verifying invariants I-1 to I-9.
- [x] **M0-05**: Implement `ports/repositories.py`, `ports/uow.py`, `ports/llm.py`, `ports/embed.py` (LLD Section 6.6, 6.7, 7.1, 7.4). Verify with `mypy --strict` and `lint-imports`.
- [ ] **M0-06**: Implement `config.py` (`Settings` per LLD Section 21) and `obs/logging.py` (LLD Section 18.1).
- [ ] **M0-07**: Create SQL migrations `000` through `005` in `db/migrations/` and migration runner `db/migrate.py` (LLD Section 5.5, 6.1, 6.2, 6.3, 6.4).

## Milestone M2: Pure Domain Logic (Zero IO, Sockets Disabled)
- [ ] **M2-01**: Implement `ingest/gate.py` (`IngestGate`) and `ingest/windowing.py` (`NoteChunker`) per LLD Section 8.1. Pass `T-ING` unit tests.
- [ ] **M2-02**: Implement `ingest/time_resolver.py` (`TimeResolver` grammar rows 1-14 per LLD Section 8.5). Pass `T-TIME` unit tests (including Oct 2026 -> Mar 2026 and Mar 2026 -> Mar 2025 anchors).
- [ ] **M2-03**: Implement `ingest/conflict/plan.py`, `registry.py`, `single.py`, `multi.py`, `event.py` (`ConflictResolver` & strategies per LLD Section 8.8). Pass `T-CR-01` through `T-CR-15` (including `T-CR-05` backfill test and Hypothesis property test G-1).
- [ ] **M2-04**: Implement `retrieve/planner/cues.py`, `intent.py`, `rules.py` (`RulesPlanner` per LLD Section 9.2) and `retrieve/lanes/filter.py` (`TemporalFilter` per LLD Section 9.3). Pass `T-PLAN` and `T-RET` filter tests.
- [ ] **M2-05**: Implement `retrieve/fusion.py` (`RrfFuser`), `retrieve/decay.py` (`DecayScorer`), `retrieve/assembler.py` (`ContextAssembler`), and `retrieve/abstain.py` (`AbstentionPolicy`) per LLD Section 9.4–9.6.