"""Experiment model.

One training run. Captures the composed Hydra config, the Hydra/TensorBoard run
directory, live/final metrics, and lifecycle status so the dashboard can render
training history without parsing event files.
"""

from __future__ import annotations

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.models.enums import ExperimentStatus


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[ExperimentStatus] = mapped_column(String(16), default=ExperimentStatus.created)

    run_dir: Mapped[str | None] = mapped_column(String(512), default=None)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Progress + results.
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int | None] = mapped_column(Integer, default=None)
    best_metric: Mapped[float | None] = mapped_column(default=None)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)     # latest scalar snapshot
    error: Mapped[str | None] = mapped_column(Text, default=None)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Experiment {self.name} status={self.status} epoch={self.epoch}>"
