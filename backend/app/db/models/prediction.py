"""Prediction model.

A single inference result for a (video, model) pair. Target point estimates and
their confidence intervals are stored as JSON keyed by the canonical target
names so the schema is stable even if the target set evolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.model import ModelArtifact
    from app.db.models.video import Video


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id"), index=True)

    # {target_name: value} point estimates (already de-normalised, human units).
    targets: Mapped[dict] = mapped_column(JSON, default=dict)
    # {target_name: [lower, upper]} confidence intervals.
    intervals: Mapped[dict] = mapped_column(JSON, default=dict)
    # {target_name: sigma} predicted aleatoric std-dev, for calibration analysis.
    uncertainty: Mapped[dict] = mapped_column(JSON, default=dict)

    # Convenience denormalised virality score for cheap sorting/filtering.
    virality_score: Mapped[float | None] = mapped_column(Float, default=None, index=True)
    confidence_level: Mapped[float] = mapped_column(Float, default=0.95)

    video: Mapped[Video] = relationship(back_populates="predictions")
    model: Mapped[ModelArtifact | None] = relationship(back_populates="predictions")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Prediction video={self.video_id} model={self.model_id}>"
