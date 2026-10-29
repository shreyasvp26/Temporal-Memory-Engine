"""Unit tests for Second-Brain Bitemporal Memory & Observability Workbench UI.

Tests that GET / and GET /ui return HTTP 200 with text/html and contain all
mandated UI components, styling tokens, and interactive controllers.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from memory_engine.config import Settings
from memory_engine.domain.clock import FixedClock
from memory_engine.ingest.canonicalizer import Canonicalizer
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.repositories import EvidenceSnippet, FactListQuery, Page

from apps.api import AppContainer, JwtVerifier, RateLimiter, create_app

JWT_TEST_SECRET = 'workbench-ui-test-key-32-bytes-long!'


class UiTestFakeFactsRepo:
    async def list_facts(self, user_id: Any, query: FactListQuery) -> Page[Any]:
        return Page(items=[], next_cursor=None)

    async def get(self, user_id: Any, fact_id: Any) -> Any:
        return None

    async def get_evidence(
        self, user_id: Any, fact_ids: Any, per_fact: int
    ) -> dict[Any, list[EvidenceSnippet]]:
        return {}

    async def chain(self, user_id: Any, fact_id: Any) -> list[Any]:
        return []


class UiTestFakeEntitiesRepo:
    async def ensure_self(self, user_id: Any) -> Any:
        return None


class UiTestFakeProfilesRepo:
    async def purge_user(self, user_id: Any) -> int:
        return 0

    async def export_user(self, user_id: Any) -> dict[str, Any]:
        return {'user_id': str(user_id)}


class UiTestFakeAuditRepo:
    async def append(self, **kwargs: Any) -> None:
        pass

    async def record(self, **kwargs: Any) -> None:
        pass


class UiTestFakeUow:
    def __init__(self) -> None:
        self.facts = UiTestFakeFactsRepo()
        self.entities = UiTestFakeEntitiesRepo()
        self.profiles = UiTestFakeProfilesRepo()
        self.audit = UiTestFakeAuditRepo()

    async def lock_user(self, user_id: Any = None) -> None:
        pass

    async def __aenter__(self) -> 'UiTestFakeUow':
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class UiTestFakeUowFactory:
    def __init__(self, uow: UiTestFakeUow) -> None:
        self._uow = uow

    def for_user(self, user_id: Any, *, readonly: bool = False) -> Any:
        return self._uow

    def begin(self, user_id: Any = None, *, readonly: bool = False) -> Any:
        return self._uow

    def system(self, *, readonly: bool = False) -> Any:
        return self._uow


@pytest.fixture
def ui_container() -> AppContainer:
    clock = FixedClock(datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC))
    uow_factory = UiTestFakeUowFactory(UiTestFakeUow())
    settings = Settings(
        env='test',
        jwt_secret=JWT_TEST_SECRET,
        jwt_audience='authenticated',
    )
    lifecycle = MemoryLifecycleService(
        uows=uow_factory,
        clock=clock,
        canonicalizer=Canonicalizer(),
    )
    jwt_verifier = JwtVerifier(settings.jwt, env='test')
    limiter = RateLimiter(
        clock=clock,
        bucket_configs={
            'chat': (30.0, 10.0),
            'ingest': (30.0, 10.0),
            'default': (120.0, 30.0),
        },
    )

    return AppContainer(
        settings=settings,
        clock=clock,
        uows=uow_factory,
        ingest=None,  # type: ignore[arg-type]
        retrieval=None,  # type: ignore[arg-type]
        chat=None,  # type: ignore[arg-type]
        lifecycle=lifecycle,
        rate_limiter=limiter,
        jwt_verifier=jwt_verifier,
        embedder=None,
    )


@pytest.fixture
async def client(ui_container: AppContainer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container=ui_container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as ac:
        yield ac


@pytest.mark.asyncio
async def test_workbench_ui_root_endpoint_returns_html(client: httpx.AsyncClient) -> None:
    """GET / serves 200 OK with text/html content-type and full Workbench UI markup."""
    resp = await client.get('/')
    assert resp.status_code == 200
    assert 'text/html' in resp.headers.get('content-type', '')

    html = resp.text

    # 1. Branding & Precision Dark styling tokens
    assert 'SECOND-BRAIN' in html
    assert 'BITEMPORAL ENGINE v1.0' in html
    assert '#09090B' in html
    assert '#121215' in html
    assert '#18181B' in html
    assert '#27272A' in html

    # 2. Typography
    assert 'Inter' in html
    assert 'JetBrains Mono' in html

    # 3. Telemetry & Top Bar Controls
    assert 'ready-pill' in html
    assert 'PG16 + HNSW READY' in html
    assert '00000000-0000-0000-0000-000000000001' in html
    assert '00000000-0000-0000-0000-000000000002' in html
    assert 'Seed Scenario (S5 + S10)' in html
    assert 'Flush Outbox (run-once)' in html
    assert 'GDPR Export' in html
    assert 'Purge Tenant' in html


@pytest.mark.asyncio
async def test_workbench_ui_explicit_endpoint_and_interactive_components(
    client: httpx.AsyncClient,
) -> None:
    """GET /ui serves 200 OK and contains Copilot, Ingestion, Timeline, Ledger, and Drawer."""
    resp = await client.get('/ui')
    assert resp.status_code == 200
    assert 'text/html' in resp.headers.get('content-type', '')

    html = resp.text

    # Left Pane: Mode Switcher & Quick Prompts
    assert 'Grounded Query (SSE)' in html
    assert 'Ingest Raw Note' in html
    assert 'temporal-anchor' in html
    assert 'Where do I live now vs. where did I live in 2024?' in html
    assert 'Where was I living and working in June 2025?' in html
    assert 'What tech stack do I use and what is my diet?' in html
    assert 'What is my passport number?' in html

    # 5-Lane Breakdown & X-Ray Telemetry
    assert 'Vector Facts' in html
    assert 'FTS Facts' in html
    assert 'Graph Expand' in html
    assert 'Temporal Facts' in html
    assert 'Vector Sources' in html
    assert 'View Assembled Prompt Context' in html

    # Right Pane: 3 Tabs & Bitemporal Timeline
    assert 'Bitemporal Timeline (Swimlanes)' in html
    assert 'Fact Ledger (Cards)' in html
    assert 'Context Assembler Output' in html
    assert 'VALID (World) :' in html
    assert 'TX    (System):' in html

    # Status Semantics
    assert 'pill-active' in html
    assert 'pill-superseded' in html
    assert 'pill-retracted' in html

    # Drawer for Provenance (Q-14) & Lineage (Q-15)
    assert 'FACT PROVENANCE' in html
    assert 'EVIDENCE QUOTES (Q-14)' in html
    assert 'SUPERSESSION LINEAGE CHAIN (Q-15)' in html
    assert 'Patch Fact Value (User Edit)' in html
