"""Database adapters, connection pooling, and Unit of Work."""

from memory_engine.db.migrate import (
    ADVISORY_LOCK_ID,
    EXPECTED_SCHEMA_VERSION,
    AsyncConnection,
    MigrationFile,
    apply_migrations,
    compute_checksum,
    load_migration_files,
    verify_applied_checksums,
    verify_startup_guard,
)
from memory_engine.db.pool import (
    create_api_pool,
    create_db_pool,
    create_eval_pool,
    create_pool,
    create_worker_pool,
    init_connection,
    translate_pg_error,
    wrap_db_errors,
)
from memory_engine.db.tx import (
    LOCK_USER_ADVISORY_SQL,
    SET_APP_USER_ID_SQL,
    SET_LOCK_TIMEOUT_SQL,
    acquire_user_advisory_lock,
    set_transaction_user,
    transaction_scope,
)
from memory_engine.db.uow import (
    PostgresUnitOfWork,
    PostgresUnitOfWorkFactory,
    clear_repo_factories,
    get_active_uow,
    register_repo_factory,
)

__all__ = [
    'ADVISORY_LOCK_ID',
    'EXPECTED_SCHEMA_VERSION',
    'AsyncConnection',
    'LOCK_USER_ADVISORY_SQL',
    'MigrationFile',
    'PostgresUnitOfWork',
    'PostgresUnitOfWorkFactory',
    'SET_APP_USER_ID_SQL',
    'SET_LOCK_TIMEOUT_SQL',
    'acquire_user_advisory_lock',
    'apply_migrations',
    'clear_repo_factories',
    'compute_checksum',
    'create_api_pool',
    'create_db_pool',
    'create_eval_pool',
    'create_pool',
    'create_worker_pool',
    'get_active_uow',
    'init_connection',
    'load_migration_files',
    'register_repo_factory',
    'set_transaction_user',
    'transaction_scope',
    'translate_pg_error',
    'verify_applied_checksums',
    'verify_startup_guard',
    'wrap_db_errors',
]
