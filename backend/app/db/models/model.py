"""Model registry entry.

A trained artefact (Brain Encoder + multitask head) promoted for inference.
Stores the composed Hydra config, the target normalisation statistics needed to
de-normalise predictions, evaluation metrics, and the checkpoint path. Exactly
one row may be ``is_active`` at a time (enforced in the repository layer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.prediction import Prediction


class ModelArtifact(Base, TimestampMixin):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "version", name="name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(32), default="v1")

    checkpoint_path: Mapped[str] = mapped_column(String(512))
    onnx_path: Mapped[str | None] = mapped_column(String(512), default=None)
    torchscript_path: Mapped[str | None] = mapped_column(String(512), default=None)

    # Full composed config (model + training) for exact reproducibility.
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    # Per-target mean/std used to de-normalise regression outputs.
    target_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    # Held-out metrics captured at promotion time.
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="model")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        flag = " *active" if self.is_active else ""
        return f"<ModelArtifact {self.name}:{self.version}{flag}>"
