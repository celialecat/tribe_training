"""Cached embedding/tensor inspection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models.embedding import Embedding
from app.db.models.enums import EmbeddingKind


class TensorService:
    """List cached brain-activity artefacts from the database and disk."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def list(self, session: Session, *, video_id: int | None = None) -> dict[str, Any]:
        try:
            stmt = select(Embedding).where(Embedding.kind == EmbeddingKind.brain_activity)
            if video_id is not None:
                stmt = stmt.where(Embedding.video_id == video_id)
            rows = list(session.scalars(stmt))
        except SQLAlchemyError:
            rows = []
        items = [self._row(item) for item in rows]
        total_bytes = sum(item["file_size_bytes"] for item in items)
        return {"items": items, "count": len(items), "storage_bytes": total_bytes}

    def get(self, session: Session, video_id: int) -> dict[str, Any]:
        return self.list(session, video_id=video_id)

    def _row(self, item: Embedding) -> dict[str, Any]:
        path = self._abs(item.path)
        size = path.stat().st_size if path.exists() else 0
        return {
            "video_id": item.video_id,
            "shape": item.shape,
            "dtype": item.dtype,
            "num_timesteps": item.num_timesteps,
            "num_features": item.num_features,
            "file_size_bytes": size,
            "path": str(path),
        }

    def _abs(self, stored: str) -> Path:
        path = Path(stored)
        return path if path.is_absolute() else self.settings.data_dir / path
