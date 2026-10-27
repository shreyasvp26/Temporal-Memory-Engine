"""Unit of Work ports."""

from types import TracebackType
from typing import Protocol

from memory_engine.domain.ids import UserId
from memory_engine.ports.repositories import (
    AuditRepository,
    EntityRepository,
    FactRepository,
    JobRepository,
    MessageRepository,
    ProfileRepository,
    SearchRepository,
    SourceRepository,
)


class UnitOfWork(Protocol):
    sources: SourceRepository
    entities: EntityRepository
    facts: FactRepository
    search: SearchRepository
    jobs: JobRepository
    messages: MessageRepository
    audit: AuditRepository
    profiles: ProfileRepository

    async def lock_user(self, user_id: UserId | None = None) -> None: ...

    async def __aenter__(self) -> 'UnitOfWork': ...

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class UnitOfWorkFactory(Protocol):
    def begin(self, user_id: UserId | None, *, readonly: bool = False) -> UnitOfWork: ...

    def for_user(self, user_id: UserId, *, readonly: bool = False) -> UnitOfWork: ...

    def system(self, *, readonly: bool = False) -> UnitOfWork: ...
