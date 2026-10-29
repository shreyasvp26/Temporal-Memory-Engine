"""Dependency injection container and FastAPI dependencies for API endpoints."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from memory_engine.config import Settings
from memory_engine.domain.clock import Clock
from memory_engine.domain.errors import InvariantViolation
from memory_engine.ingest.service import IngestService
from memory_engine.lifecycle.service import MemoryLifecycleService
from memory_engine.ports.embed import Embedder
from memory_engine.ports.uow import UnitOfWorkFactory
from memory_engine.retrieve.chat_service import ChatService
from memory_engine.retrieve.orchestrator import RetrievalOrchestrator

from .auth import JwtVerifier
from .rate_limit import RateLimiter


@dataclass
class AppContainer:
    """Composition root container holding wired services and adapters."""

    settings: Settings
    clock: Clock
    uows: UnitOfWorkFactory
    ingest: IngestService
    retrieval: RetrievalOrchestrator
    chat: ChatService
    lifecycle: MemoryLifecycleService
    rate_limiter: RateLimiter | None = None
    jwt_verifier: JwtVerifier | None = None
    embedder: Embedder | None = None


def get_container(request: Request) -> AppContainer:
    """Retrieve AppContainer from request.app.state."""
    container = getattr(request.app.state, 'container', None)
    if container is None:
        raise InvariantViolation('AppContainer is not initialized on app.state.container')
    return container  # type: ignore[no-any-return]


ContainerDep = Annotated[AppContainer, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    """FastAPI dependency for Settings."""
    return container.settings


def get_clock(container: ContainerDep) -> Clock:
    """FastAPI dependency for Clock."""
    return container.clock


def get_uow_factory(container: ContainerDep) -> UnitOfWorkFactory:
    """FastAPI dependency for UnitOfWorkFactory."""
    return container.uows


def get_ingest_service(container: ContainerDep) -> IngestService:
    """FastAPI dependency for IngestService."""
    return container.ingest


def get_retrieval_orchestrator(
    container: ContainerDep,
) -> RetrievalOrchestrator:
    """FastAPI dependency for RetrievalOrchestrator."""
    return container.retrieval


def get_chat_service(container: ContainerDep) -> ChatService:
    """FastAPI dependency for ChatService."""
    return container.chat


def get_lifecycle_service(
    container: ContainerDep,
) -> MemoryLifecycleService:
    """FastAPI dependency for MemoryLifecycleService."""
    return container.lifecycle


def get_embedder(container: ContainerDep) -> Embedder | None:
    """FastAPI dependency for Embedder."""
    return container.embedder
