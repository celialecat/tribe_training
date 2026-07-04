"""Embedding model.

Records TRIBE-derived artefacts for a video. The heavy tensors live on disk
(``.npy`` under the cache directory); the row stores the path, shape, dtype and
a content hash so we can detect staleness and avoid recomputation. Two kinds are
tracked (see :class:`EmbeddingKind`): raw brain activity (T x V) and the 512-d
brain latent produced by a specific encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.models.enums import EmbeddingKind

if TYPE_CHECKING:
    from app.db.models.video import Video


class Embedding(Base, TimestampMixin):
    __tablename__ = "embeddings"
    __table_args__ = (
        # One artefact of each kind per (video, producer) combination.
        UniqueConstraint("video_id", "kind", "producer", name="video_kind_producer"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), index=True)

    kind: Mapped[EmbeddingKind] = mapped_column(String(32))
    # Which component produced this artefact: e.g. "tribev2" for brain activity,
    # or an encoder model name/version for a latent.
    producer: Mapped[str] = mapped_column(String(128))

    # On-disk artefact.
    path: Mapped[str] = mapped_column(String(512))
    shape: Mapped[list[int]] = mapped_column(JSON)      # e.g. [T, V] or [512]
    dtype: Mapped[str] = mapped_column(String(16), default="float32")
    num_timesteps: Mapped[int | None] = mapped_column(Integer, default=None)
    num_features: Mapped[int | None] = mapped_column(Integer, default=None)
    # Hash of the source inputs so we can invalidate on change.
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)

    video: Mapped[Video] = relationship(back_populates="embeddings")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Embedding video={self.video_id} kind={self.kind} shape={self.shape}>"
