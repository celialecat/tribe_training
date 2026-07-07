"""Sequential end-to-end pipeline orchestration."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.services.dataset_service import DatasetFilter, DatasetService
from app.services.evaluation_service import EvaluationService
from app.services.jobs import JobContext
from app.services.training_service import TrainingService
from app.services.tribe_service import TribeService


class PipelineService:
    """Run download -> validate -> tribe -> train -> evaluate in sequence."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dataset = DatasetService(settings=self.settings)
        self.tribe = TribeService(settings=self.settings)
        self.training = TrainingService(settings=self.settings)
        self.evaluation = EvaluationService(settings=self.settings)

    def run(
        self,
        *,
        mode: str,
        n_videos: int,
        horizon_days: int | None = None,
        job: JobContext | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": mode, "n_videos": n_videos}
        result["download"] = self.dataset.build(
            mode, count=n_videos, filter=DatasetFilter(), job=job
        )
        if job is not None:
            job.set_progress(stage="validate", current=0, total=1, message="validate")
        result["validate"] = self.dataset.validate(count=n_videos, job=job)
        if job is not None:
            job.set_progress(stage="tribe", current=0, total=1, message="tribe")
        result["tribe"] = self.tribe.run(job=job, limit=n_videos)
        if job is not None:
            job.set_progress(stage="train", current=0, total=1, message="train")
        result["train"] = self.training.run(
            overrides={"prediction_horizon_days": horizon_days} if horizon_days else None, job=job
        )
        if job is not None:
            job.set_progress(stage="evaluate", current=0, total=1, message="evaluate")
        result["evaluate"] = self.evaluation.run(job=job)
        return result
