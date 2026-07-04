"""Database engine, session factory and declarative base.

We expose a synchronous SQLAlchemy 2.0 setup. Synchronous is a deliberate
choice: dataset building and training are batch/CLI workloads, and the FastAPI
layer runs blocking DB calls inside threadpool-backed dependencies (see
``app.api.deps``). This keeps a single ORM mental model across CLI, trainer and
API rather than maintaining parallel sync/async stacks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Explicit, DB-agnostic naming convention so Alembic autogenerate emits stable,
# named constraints (required for clean SQLite batch migrations).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` columns to a model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


def _make_engine(url: str):
    """Create an engine with SQLite-friendly defaults."""
    connect_args = {}
    if url.startswith("sqlite"):
        # Allow cross-thread use (FastAPI threadpool) and reasonable timeouts.
        connect_args = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True)
    if url.startswith("sqlite"):
        _enable_sqlite_pragmas(engine)
    return engine


def _enable_sqlite_pragmas(engine) -> None:
    """Enable WAL + foreign keys so SQLite behaves well under concurrency."""
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


# Lazily-initialised process-wide engine + session factory.
_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine():
    """Return (creating on first call) the process-wide engine."""
    global _engine, _SessionFactory
    if _engine is None:
        url = get_settings().database_url
        logger.debug("Creating database engine for %s", url.split("://")[0])
        _engine = _make_engine(url)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commit on success, rollback on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create all tables directly (used in tests; prod uses Alembic)."""
    import app.db.models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=get_engine())
