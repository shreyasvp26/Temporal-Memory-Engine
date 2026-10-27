"""PostgreSQL Unit of Work and Factory implementation."""

from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any

from memory_engine.db.pool import wrap_db_errors
from memory_engine.db.tx import acquire_user_advisory_lock, set_transaction_user
from memory_engine.domain.errors import InvariantViolation
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
from memory_engine.ports.uow import UnitOfWork, UnitOfWorkFactory

_active_uow: ContextVar['PostgresUnitOfWork | None'] = ContextVar('sb_active_uow', default=None)

RepoFactory = Callable[['PostgresUnitOfWork'], Any]
_global_repo_factories: dict[str, RepoFactory] = {}


def register_repo_factory(name: str, factory: RepoFactory) -> None:
    """Register a repository adapter constructor by name ('sources', 'facts', etc.)."""
    _global_repo_factories[name] = factory


def clear_repo_factories() -> None:
    """Clear registered global repository factories (useful for test resets)."""
    _global_repo_factories.clear()


def get_active_uow() -> 'PostgresUnitOfWork | None':
    """Return the active UnitOfWork in the current task context, or None."""
    return _active_uow.get()


class PostgresUnitOfWork(UnitOfWork):
    """PostgreSQL implementation of UnitOfWork managing a single connection and transaction."""

    def __init__(
        self,
        pool: Any,
        user_id: UserId | None = None,
        *,
        readonly: bool = False,
        repo_factories: dict[str, RepoFactory] | None = None,
        sources: SourceRepository | None = None,
        entities: EntityRepository | None = None,
        facts: FactRepository | None = None,
        search: SearchRepository | None = None,
        jobs: JobRepository | None = None,
        messages: MessageRepository | None = None,
        audit: AuditRepository | None = None,
        profiles: ProfileRepository | None = None,
    ) -> None:
        self._pool = pool
        self._user_id = user_id
        self._readonly = readonly
        self._repo_factories = repo_factories or {}

        self._conn: Any | None = None
        self._tx: Any | None = None
        self._token: Token[PostgresUnitOfWork | None] | None = None
        self._locked_user_id: UserId | None = None

        self._sources = sources
        self._entities = entities
        self._facts = facts
        self._search = search
        self._jobs = jobs
        self._messages = messages
        self._audit = audit
        self._profiles = profiles

    @property
    def user_id(self) -> UserId | None:
        """Bound user ID, or None for system/worker tasks."""
        return self._user_id

    @property
    def readonly(self) -> bool:
        """Whether this transaction is read-only."""
        return self._readonly

    @property
    def conn(self) -> Any:
        """Active database connection within this transaction."""
        if self._conn is None:
            raise InvariantViolation('UnitOfWork is not open (no active connection)')
        return self._conn

    async def lock_user(self, user_id: UserId | None = None) -> None:
        """Take per-user advisory xact lock on the bound user.

        Serializes timeline fact updates for this user.
        """
        if self._conn is None:
            raise InvariantViolation('UnitOfWork is not open')

        target_user_id = user_id if user_id is not None else self._user_id
        if target_user_id is None:
            raise InvariantViolation(
                'Cannot take user advisory lock on a system UnitOfWork without specifying user_id'
            )

        if self._user_id is not None and user_id is not None and self._user_id != user_id:
            raise InvariantViolation(
                f'UnitOfWork bound to user {self._user_id} cannot lock different user {user_id}'
            )

        if self._locked_user_id is not None:
            if self._locked_user_id == target_user_id:
                return
            raise InvariantViolation(
                f'Transaction already holds advisory lock for user {self._locked_user_id}, '
                f'cannot lock {target_user_id}'
            )

        async with wrap_db_errors():
            await acquire_user_advisory_lock(self._conn, target_user_id)
        self._locked_user_id = target_user_id

    async def __aenter__(self) -> 'PostgresUnitOfWork':
        if self._conn is not None:
            raise InvariantViolation('UnitOfWork is already open')

        if _active_uow.get() is not None:
            raise InvariantViolation(
                'Nested UnitOfWork is forbidden: an active UoW is already open in this task context'
            )

        conn = None
        tx = None
        try:
            async with wrap_db_errors():
                conn = await self._pool.acquire()
                tx = conn.transaction(readonly=self._readonly)
                await tx.start()

                if self._user_id is not None:
                    await set_transaction_user(conn, self._user_id)
        except Exception:
            if tx is not None:
                with suppress(Exception):
                    await tx.rollback()
            if conn is not None:
                with suppress(Exception):
                    await self._pool.release(conn)
            raise

        self._conn = conn
        self._tx = tx
        self._token = _active_uow.set(self)
        return self

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._conn is None or self._tx is None:
            return

        conn = self._conn
        tx = self._tx
        token = self._token

        self._conn = None
        self._tx = None
        self._token = None
        self._locked_user_id = None

        try:
            async with wrap_db_errors():
                if et is not None:
                    await tx.rollback()
                else:
                    await tx.commit()
        finally:
            if token is not None:
                _active_uow.reset(token)
            with suppress(Exception):
                await self._pool.release(conn)

    def _get_or_create_repo(self, name: str) -> Any:
        if name in self._repo_factories:
            return self._repo_factories[name](self)
        if name in _global_repo_factories:
            return _global_repo_factories[name](self)
        if name == 'sources':
            from memory_engine.db.repos.sources import PostgresSourceRepository

            return PostgresSourceRepository(self)
        if name == 'entities':
            from memory_engine.db.repos.entities import PostgresEntityRepository

            return PostgresEntityRepository(self)
        if name == 'facts':
            from memory_engine.db.repos.facts import PostgresFactRepository

            return PostgresFactRepository(self)
        if name == 'search':
            from memory_engine.db.repos.search import PostgresSearchRepository

            return PostgresSearchRepository(self)
        if name == 'jobs':
            from memory_engine.db.repos.jobs import PostgresJobRepository

            return PostgresJobRepository(self)
        if name == 'messages':
            from memory_engine.db.repos.messages import PostgresMessageRepository

            return PostgresMessageRepository(self)
        if name == 'audit':
            from memory_engine.db.repos.audit import PostgresAuditRepository

            return PostgresAuditRepository(self)
        if name == 'profiles':
            from memory_engine.db.repos.profile import PostgresProfileRepository

            return PostgresProfileRepository(self)
        raise NotImplementedError(f'{name.capitalize()} repository adapter not wired yet')

    @property
    def sources(self) -> SourceRepository:
        if self._sources is None:
            self._sources = self._get_or_create_repo('sources')
        return self._sources

    @sources.setter
    def sources(self, value: SourceRepository) -> None:
        self._sources = value

    @property
    def entities(self) -> EntityRepository:
        if self._entities is None:
            self._entities = self._get_or_create_repo('entities')
        return self._entities

    @entities.setter
    def entities(self, value: EntityRepository) -> None:
        self._entities = value

    @property
    def facts(self) -> FactRepository:
        if self._facts is None:
            self._facts = self._get_or_create_repo('facts')
        return self._facts

    @facts.setter
    def facts(self, value: FactRepository) -> None:
        self._facts = value

    @property
    def search(self) -> SearchRepository:
        if self._search is None:
            self._search = self._get_or_create_repo('search')
        return self._search

    @search.setter
    def search(self, value: SearchRepository) -> None:
        self._search = value

    @property
    def jobs(self) -> JobRepository:
        if self._jobs is None:
            self._jobs = self._get_or_create_repo('jobs')
        return self._jobs

    @jobs.setter
    def jobs(self, value: JobRepository) -> None:
        self._jobs = value

    @property
    def messages(self) -> MessageRepository:
        if self._messages is None:
            self._messages = self._get_or_create_repo('messages')
        return self._messages

    @messages.setter
    def messages(self, value: MessageRepository) -> None:
        self._messages = value

    @property
    def audit(self) -> AuditRepository:
        if self._audit is None:
            self._audit = self._get_or_create_repo('audit')
        return self._audit

    @audit.setter
    def audit(self, value: AuditRepository) -> None:
        self._audit = value

    @property
    def profiles(self) -> ProfileRepository:
        if self._profiles is None:
            self._profiles = self._get_or_create_repo('profiles')
        return self._profiles

    @profiles.setter
    def profiles(self, value: ProfileRepository) -> None:
        self._profiles = value


class PostgresUnitOfWorkFactory(UnitOfWorkFactory):
    """Factory creating PostgresUnitOfWork instances for users or system operations."""

    def __init__(
        self,
        pool: Any,
        *,
        default_readonly: bool = False,
        repo_factories: dict[str, RepoFactory] | None = None,
    ) -> None:
        self._pool = pool
        self._default_readonly = default_readonly
        self._repo_factories = dict(repo_factories) if repo_factories else None

    def begin(
        self,
        user_id: UserId | None = None,
        *,
        readonly: bool = False,
    ) -> PostgresUnitOfWork:
        """Begin a new UnitOfWork. Raises InvariantViolation if a UoW is already active."""
        if _active_uow.get() is not None:
            raise InvariantViolation(
                'Nested UnitOfWork is forbidden: an active UoW is already open in this task context'
            )
        ro = self._default_readonly if readonly is False else readonly
        return PostgresUnitOfWork(
            pool=self._pool,
            user_id=user_id,
            readonly=ro,
            repo_factories=self._repo_factories,
        )

    def for_user(
        self,
        user_id: UserId,
        *,
        readonly: bool = False,
    ) -> PostgresUnitOfWork:
        """Create a UnitOfWork bound to a specific user (sets app.user_id)."""
        return self.begin(user_id, readonly=readonly)

    def system(
        self,
        *,
        readonly: bool = False,
    ) -> PostgresUnitOfWork:
        """Create a system UnitOfWork without user context (for jobs/worker operations)."""
        return self.begin(None, readonly=readonly)
