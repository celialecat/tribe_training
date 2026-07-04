"""Embedding repository."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models.embedding import Embedding
from app.db.models.enums import EmbeddingKind
from app.db.repositories.base import BaseRepository


class EmbeddingRepository(BaseRepository[Embedding]):
    model = Embedding

    def find(
        self, video_id: int, kind: EmbeddingKind, producer: str
    ) -> Embedding | None:
        stmt = select(Embedding).where(
            Embedding.video_id == video_id,
            Embedding.kind == kind,
            Embedding.producer == producer,
        )
        return self.session.scalar(stmt)

    def upsert(self, embedding: Embedding) -> Embedding:
        """Insert or replace the artefact for (video, kind, producer)."""
        existing = self.find(embedding.video_id, embedding.kind, embedding.producer)
        if existing is not None:
            existing.path = embedding.path
            existing.shape = embedding.shape
            existing.dtype = embedding.dtype
            existing.num_timesteps = embedding.num_timesteps
            existing.num_features = embedding.num_features
            existing.content_hash = embedding.content_hash
            self.session.flush()
            return existing
        return self.add(embedding)
