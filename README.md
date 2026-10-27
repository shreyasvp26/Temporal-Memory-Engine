<div align="center">

# 🧠 Temporal Memory Engine (`second-brain`)

### Production-Grade Bitemporal Episodic & Semantic Memory System for LLM Agents

[![CI](https://github.com/shreyasvp26/Temporal-Memory-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyasvp26/Temporal-Memory-Engine/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)
[![Architecture: Hexagonal](https://img.shields.io/badge/architecture-hexagonal-green.svg)](https://alistair.cockburn.us/hexagonal-architecture/)
[![PostgreSQL 16 + pgvector](https://img.shields.io/badge/postgres-16%20%7C%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)

</div>

---

## 📌 Executive Summary

Traditional LLM agent memory architectures suffer from three critical flaws:
1. **Silent Overwriting:** Updating a user's location (e.g., *"I moved from London to Tokyo"*) mutates or deletes the past, destroying historical context needed to answer questions like *"Where did I live during the 2024 Olympics?"*.
2. **Out-of-Order Fragility:** Messages arriving late, batch-ingested from notes, or synced across offline devices corrupt relational timelines if simple timestamps are used.
3. **Database Transaction Bleed:** Mixing network/LLM calls inside database write transactions exhausts connection pools and causes high lock contention.

**Temporal Memory Engine** is a high-assurance, bitemporal episodic and semantic memory engine governed by a pure Python domain core, hexagonal architecture boundaries, a **10-state pure conflict reconciliation machine (S1–S10)**, and a **5-lane hybrid retrieval fusion orchestrator** with exponential temporal decay and point-in-time replay.

---

## 🏛️ Core Architecture

```mermaid
flowchart TB
    subgraph Client ["Client Layer"]
        User["User / Agent"]
        Admin["Admin Dashboard"]
    end

    subgraph API ["FastAPI Service (apps/api)"]
        AuthMiddleware["JWT Auth + RLS Context Injection"]
        RateLimiter["Token Bucket Rate Limiter"]
        Routes["REST & SSE Streaming Endpoints"]
    end

    subgraph ReadPath ["5-Lane Retrieval Orchestrator"]
        Planner["Rules + LLM Intent Planner"]
        L1["Lane 1: Vector (HNSW Cosine)"]
        L2["Lane 2: BM25 (Postgres Full-Text)"]
        L3["Lane 3: 2-Hop Graph CTE (Degree <= 15)"]
        L4["Lane 4: Temporal Proximity"]
        L5["Lane 5: Verbatim Source Turns"]
        RRF["Reciprocal Rank Fusion (k=60)"]
        Decay["Exponential Temporal Decay (Half-Life)"]
        Synthesizer["Grounded Response Synthesizer"]
    end

    subgraph WritePath ["4-Phase Out-of-Tx Ingestion Pipeline"]
        IngestGate["Ingest Gate & Note Chunker"]
        OutboxQueue["Postgres Job Queue (Lease + Fencing)"]
        WorkerRunner["Worker Daemon (apps/worker)"]
        PhaseA["Phase A: LLM Fact & Time Extraction (No DB Lock)"]
        PhaseB["Phase B: Entity Resolution & Canonicalization"]
        PhaseC["Phase C: Pure S1-S10 Conflict State Machine"]
        PhaseD["Phase D: Atomic UnitOfWork Transaction Commit"]
    end

    subgraph Storage ["PostgreSQL 16 + pgvector"]
        Tables["sources, entities, facts, fact_sources, jobs"]
        RLS["Row Level Security: SET LOCAL app.user_id"]
        Roles["Dual Roles: app_api, app_worker"]
    end

    User --> AuthMiddleware --> Routes
    Routes --> ReadPath
    Planner --> L1 & L2 & L3 & L4 & L5 --> RRF --> Decay --> Synthesizer --> Routes
    Routes --> IngestGate --> OutboxQueue
    OutboxQueue --> WorkerRunner
    WorkerRunner --> PhaseA --> PhaseB --> PhaseC --> PhaseD --> Storage
    ReadPath -.-> Storage
```

---

## ⏱️ The Bitemporal Data Model

Every fact in the system tracks two independent time axes, creating a two-dimensional temporal plane:

| Axis | Timestamps | Description | Invariant Guarantees |
| :--- | :--- | :--- | :--- |
| **Valid Time** (Event Time) | `valid_from`<br>`valid_to` | The real-world interval during which the proposition was true in reality. | Open interval `valid_to IS NULL` denotes currently active truth. Point events have `valid_from == valid_to`. |
| **System Time** (Knowledge Time) | `recorded_at`<br>`invalidated_at` | The physical system timestamp when our system learned or superseded the fact. | Strictly monotonic. `invalidated_at IS NULL` means current belief. Never updated in-place; immutable audit log. |

### Why This Matters

Consider this sequence:
- **Jan 2024:** System records *"User lives in London"* (`valid_from = 2024-01-01`, `valid_to = NULL`).
- **Jun 2024:** User states *"I moved to Tokyo on May 1st"*.
- **The Engine Executes:**
  - Old fact London is closed: `valid_to = 2024-05-01`, `invalidated_at = 2024-06-01 10:00:00`.
  - New fact Tokyo is opened: `valid_from = 2024-05-01`, `valid_to = NULL`, `recorded_at = 2024-06-01 10:00:00`.
- **Query 1 (Current state):** *"Where do I live?"* $\rightarrow$ **Tokyo**.
- **Query 2 (Time-travel query):** *"Where did I live in March 2024?"* $\rightarrow$ Filters `valid_from <= 2024-03-01 < valid_to` $\rightarrow$ **London**.
- **Query 3 (System audit replay):** *"What did the system believe on May 15th before the June message?"* $\rightarrow$ Filters `recorded_at <= 2024-05-15 < invalidated_at` $\rightarrow$ **London**.

---

## 🔄 4-Phase Out-of-Transaction Ingestion Pipeline

To satisfy the non-negotiable architectural rule: **"Never run network/LLM/embedding calls inside a database write transaction"**, ingestion executes in 4 strictly decoupled phases:

```
[ Incoming Message / Note ]
           │
           ▼
[ Ingest Gate ] ──── (Filter greetings, trivial queries, out-of-scope turns)
           │
           ▼
[ Transactional Outbox ] ── (INSERT INTO jobs WITH coalesce_key & debounce window)
           │
  ═════════╪══════════════════════════════════════════════════════════════════════
           │  Worker polls with `UPDATE ... FOR UPDATE SKIP LOCKED`
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase A: LLM Fact Extraction (Zero DB Locks Held)                            │
│  - Extracts candidate facts, entity mentions, predicates, and time phrases.  │
│  - Deterministic Python TimeResolver parses relative dates ("two years ago"). │
├──────────────────────────────────────────────────────────────────────────────┤
│ Phase B: Entity Linking & Precomputations                                    │
│  - Embeds entity mentions and resolves aliases against existing entities.     │
│  - Canonicalizes objects (scalars, ISO dates, entity pointers).              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Phase C: Pure Conflict State Machine (S1–S10)                                │
│  - Fetches existing timeline for (subject, predicate).                       │
│  - Runs pure, side-effect-free resolution algorithm.                          │
│  - Produces an explicit atomic ResolutionPlan (Insert, Close, Bump, Review).  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Phase D: Atomic Database Write Commit                                        │
│  - Single short transaction: executes ResolutionPlan ops in pgsql.           │
│  - Updates outbox job state to 'completed', releases lease.                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 S1–S10 Pure Conflict State Machine

The single-valued timeline reconciler is modeled as an immutable, side-effect-free decision table tested with 100% branch coverage:

| Rule | Trigger Condition | Engine Action | Guarantee |
| :---: | :--- | :--- | :--- |
| **S1** | Explicit retraction targeting current fact | Close active fact (`valid_to = now`, `invalidated_at = now`) | Active fact ended; historical record retained |
| **S2** | Explicit retraction targeting past fact | Set `status = 'deleted'`, `invalidated_at = now` | Historical fact marked retracted |
| **S3** | Repeat affirmation (same subject, predicate, object) | `BumpConfidence` (+0.05), attach new source evidence | Zero duplicate facts created |
| **S4** | Future state update (`valid_from > now`) | Insert future fact with future `valid_from` | Current fact remains active until transition |
| **S5** | Current contradiction (`valid_from >= active.valid_from`) | Close old fact at new `valid_from`; insert new fact | Clean temporal boundary, no overlap |
| **S6** | Historical backfill (`valid_to` fits gap between facts) | Insert historical fact in gap; clamp bounds | Fills past timeline without corrupting present |
| **S7** | Historical fact overwriting open tail | Split/close existing historical fact at `valid_from` | No overlapping valid intervals |
| **S8** | Complete containment in historical interval | Split existing interval into $[T_0, T_1]$ and $[T_2, T_3]$ | Exact interval substitution |
| **S9** | Multi-interval boundary overlap | Truncate intersecting intervals to fit candidate | Pairwise disjoint valid intervals |
| **S10**| Irreconcilable ambiguity or confidence conflict | Route candidate to `status = 'needs_review'` | Zero silent corruption; human/judge review |

---

## 🔍 5-Lane Hybrid Retrieval Orchestrator

Queries are routed by an Intent Planner across 5 orthogonal retrieval lanes:

```
                  ┌─────────────────┐
                  │   User Query    │
                  └────────┬────────┘
                           │
                 [ Query Plan Parser ]
                           │
       ┌───────────┬───────┴───┬───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼
   [ Lane 1 ]  [ Lane 2 ]  [ Lane 3 ]  [ Lane 4 ]  [ Lane 5 ]
     Vector       BM25      2-Hop Graph  Temporal    Source
    (pgvector)  (Full-Text)    (CTE)     (Interval)  (Turns)
       │           │           │           │           │
       └───────────┼───────────┴───────────┼───────────┘
                   ▼                       ▼
       [ Reciprocal Rank Fusion ] ── (k = 60 rank fusion)
                   │
                   ▼
     [ Exponential Decay Scorer ] ── (S * 2^(-Δt / τ), floor = 0.3)
                   │
                   ▼
    [ Bounded Context Assembler ] ── (Strict 3,000 token budget)
                   │
                   ▼
     [ LLM Synthesis / Stream ]  ── (SSE streaming response)
```

1. **Lane 1 (Vector):** HNSW cosine index over 768-dim `nomic-ai/nomic-embed-text-v1.5` embeddings.
2. **Lane 2 (Full-Text):** PostgreSQL `tsvector` with English stemming and dictionary match.
3. **Lane 3 (Graph CTE):** 2-hop recursive graph traversal over `fact_relations` with per-hop degree cap $\le 15$ to avoid super-node explosion.
4. **Lane 4 (Temporal Proximity):** Timeline interval overlap filter prioritizing facts valid at target inquiry time.
5. **Lane 5 (Source Turns):** Verbatim conversation turn retrieval for contextual grounding.
6. **RRF & Decay:** Ranks are fused via $RRF(d) = \sum \frac{w_i}{k + r_i(d)}$ followed by half-life decay $2^{-\Delta t / \tau}$ (with 14, 90, or 365-day half-lives depending on predicate volatility). Time-scoped queries are immune to decay.

---

## 🛡️ Engineering Guarantees & Correctness Invariants

The engine enforces 9 constitutional domain invariants (**I-1** to **I-9**) and property-based verification (**G-1**):

- **I-1 (Strict Bitemporality):** Every fact possesses valid and system intervals; `valid_from` is always set.
- **I-2 (No Active Gaps):** Single-valued predicates never have more than one open (`valid_to IS NULL`) active fact.
- **I-3 (Pairwise Disjoint Intervals):** Active single-valued facts for the same subject-predicate have non-overlapping validity windows.
- **I-4 (Timezone-Aware UTC):** All timestamps are strictly timezone-aware UTC (`tzinfo is not None`).
- **I-5 (Fencing & Lease Protection):** Background worker claims use atomic leases and fencing tokens; lost leases abort writes immediately.
- **I-6 (Row-Level Security):** Multi-tenant isolation is enforced at the PostgreSQL kernel level via `SET LOCAL app.user_id` and dual roles (`app_api` and `app_worker`).
- **I-7 (Zero Tx Network Calls):** Network, embedding, and LLM calls never occur inside a database write transaction (`UnitOfWork`).
- **I-8 (Idempotent Ingestion):** Re-submitting the same conversation turn produces identical state via content hashing and debouncing.
- **I-9 (Hermetic Unit Testing):** Unit tests execute with network sockets disabled (`pytest --disable-socket`), ensuring complete isolation from external services.
- **G-1 (Hypothesis Property Testing):** Timeline permutation test generates shuffled lists of facts with random validity dates and verifies that out-of-order ingestion yields identical disjoint timeline invariants.

---

## 📂 Repository Structure

```
Temporal Memory Engine/
├── apps/
│   ├── api/                    # FastAPI web application, routes, auth, rate limiting
│   │   ├── auth.py             # JWT verification (JWKS / HS256) & RLS injection
│   │   ├── deps.py             # Dependency injection container (AppContainer)
│   │   ├── main.py             # FastAPI app factory & lifespan connection pool
│   │   ├── routes.py           # REST endpoints & SSE streaming /v1/chat
│   │   └── schemas.py          # Strict Pydantic v2 boundary DTO schemas
│   └── worker/                 # Background daemon processing outbox extract jobs
│       └── main.py             # WorkerRunner polling loop, heartbeats, and reaper
├── packages/
│   └── memory_engine/          # Hexagonal core library
│       └── src/memory_engine/
│           ├── domain/         # Pure Python models, primitives, errors (ZERO dependencies)
│           ├── ports/          # Abstract repository and adapter interfaces
│           ├── db/             # asyncpg connection pools, UoW, SQL catalog, migrations
│           ├── ingest/         # Gate, Chunker, Extractor, Linker, S1-S10 ConflictResolver
│           ├── retrieve/       # 5 retrieval lanes, RRF fusion, decay scorer, ChatService
│           ├── lifecycle/      # Memory review, deletion, restore, and profile maintenance
│           ├── config.py       # pydantic-settings (SB_ prefix configuration)
│           └── obs/            # Structured logging and OpenTelemetry tracing
├── eval/                       # LongMemEval evaluation runner, metrics & baselines
│   ├── baselines.py            # Full-context, vector-only, BM25-only baselines
│   ├── cli.py                  # Evaluation CLI runner
│   ├── longmemeval.py          # Dataset adapter, bootstrap stats, Markdown reporter
│   └── metrics.py              # NDCG@K, Recall@K, Accuracy, Bitemporal Accuracy
├── tests/
│   ├── unit/                   # 450+ hermetic unit tests (sockets disabled)
│   └── integration/            # Postgres 16 + pgvector RLS and bitemporal E2E tests
├── ops/
│   ├── entrypoint.sh           # Container entrypoint script
│   └── init-db.sh              # Postgres initialization and role provisioning
├── Dockerfile                  # Multi-stage Python 3.12 + uv build
├── docker-compose.yml          # Production-like Postgres 16, API, and Worker stack
├── Makefile                    # Standard engineering task runner
├── pyproject.toml              # Unified project dependencies, ruff, mypy, import-linter
└── README.md                   # System documentation
```

---

## 🚀 Quickstart Guide

### Option 1: Docker Compose (Fastest)

Clone the repository and spin up the complete stack (Postgres with pgvector, API, and Background Worker):

```bash
# Clone the repository
git clone https://github.com/shreyasvp26/Temporal-Memory-Engine.git
cd Temporal-Memory-Engine

# Launch Postgres 16 + pgvector, FastAPI, and Background Worker
docker compose up -d

# Verify services are healthy
docker compose ps
```

The API will be available at `http://localhost:8000`:
- **Health check:** `curl http://localhost:8000/healthz`
- **Interactive OpenAPI documentation:** `http://localhost:8000/docs`

---

### Option 2: Local Development Setup

Prerequisites: Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install all dependencies including dev tools
uv sync --all-extras

# 2. Run the complete test and linting verification pipeline
make check
```

`make check` executes:
1. `uv run ruff check .` (linter)
2. `uv run ruff format --check .` (code formatting)
3. `uv run mypy packages/memory_engine/src apps eval tests` (strict type checking)
4. `uv run lint-imports` (architectural boundary contract verification)
5. `uv run pytest tests/unit --disable-socket -m "not integration"` (hermetic unit tests)

---

### Running Integration Tests

Integration tests run against a real PostgreSQL 16 + pgvector instance to verify Row Level Security (RLS) tenant isolation and bitemporal outbox execution:

```bash
# Spin up test Postgres container
docker compose up -d postgres

# Run the integration test suite
SB_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/second_brain" make test-int
```

---

## 📊 LongMemEval Benchmark & Evaluation CLI

The engine includes a full evaluation harness supporting the **LongMemEval** benchmark, custom memory suites, and baseline comparisons (Full Context, Vector-only, BM25-only):

```bash
# Evaluate Temporal Memory Engine on LongMemEval
uv run python -m eval.cli --dataset data/longmemeval.json --output-dir results

# Evaluate a baseline model
uv run python -m eval.cli --dataset data/longmemeval.json --baseline full_context --output-dir results/full_context

# Run with custom sample limits and random seed
uv run python -m eval.cli run --dataset data/longmemeval.json --limit 50 --seed 42
```

### Metrics Tracked
- **NDCG@K & Recall@K:** Ranking relevance of retrieved memory facts.
- **Answer Accuracy:** Synthesized answer precision judged by an LLM-as-a-Judge against ground truth.
- **Bitemporal Accuracy:** Correctness on temporal reasoning questions involving superseded facts, historical state, and event chronology.
- **Abstention Precision & Recall:** System ability to correctly abstain when memories do not contain the answer.

---

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
