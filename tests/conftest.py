"""Shared pytest fixtures.

Tests run without a GPU or the `tribev2` weights by default; cases that need
them are marked ``@pytest.mark.gpu`` / ``@pytest.mark.tribe`` and skipped in CI.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Force a hermetic, temp-backed environment before app.core.config is imported.
os.environ.setdefault("YSP_ENV", "test")


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Provide a Settings instance whose storage points at a temp directory."""
    from app.core.config import Settings, get_settings

    monkeypatch.setenv("YSP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YSP_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = Settings()
    settings.ensure_directories()
    yield settings
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield a Session backed by a fresh temp SQLite DB with the schema created."""
    import app.db.base as db_base
    from app.core.config import get_settings

    monkeypatch.setenv("YSP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YSP_PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("YSP_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    # Reset the module-level engine so it rebinds to the temp database.
    db_base._engine = None
    db_base._SessionFactory = None
    db_base.create_all()

    session = db_base.get_session_factory()()
    try:
        yield session
    finally:
        session.close()
        db_base._engine = None
        db_base._SessionFactory = None
        get_settings.cache_clear()  # type: ignore[attr-defined]
