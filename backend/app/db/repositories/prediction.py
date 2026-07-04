"""Prediction & experiment repositories."""

from __future__ import annotations

from sqlalchemy import desc, select

from app.db.models.experiment import Experiment
from app.db.models.prediction import Prediction
from app.db.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    def latest_for_video(self, video_id: int) -> Prediction | None:
        stmt = (
            select(Prediction)
            .where(Prediction.video_id == video_id)
            .order_by(desc(Prediction.created_at))
            .limit(1)
        )
        return self.session.scalar(stmt)

    def history(self, *, limit: int = 50, offset: int = 0) -> list[Prediction]:
        stmt = (
            select(Prediction)
            .order_by(desc(Prediction.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))


class ExperimentRepository(BaseRepository[Experiment]):
    model = Experiment

    def by_name(self, name: str) -> Experiment | None:
        return self.session.scalar(select(Experiment).where(Experiment.name == name))

    def recent(self, *, limit: int = 20) -> list[Experiment]:
        stmt = select(Experiment).order_by(desc(Experiment.created_at)).limit(limit)
        return list(self.session.scalars(stmt))
