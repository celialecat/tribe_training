"""Single-video prediction workflow for dashboard use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.constants import PRIMARY_TARGET, Target
from app.core.device import resolve_device
from app.core.logging import get_logger
from app.dataset.builder import DatasetBuilder
from app.dataset.features import build_feature_vector
from app.db.base import session_scope
from app.db.models.video import Video
from app.db.repositories.model import ModelRepository
from app.db.repositories.prediction import ExperimentRepository
from app.explainability.explainer import Explainer
from app.models import TargetNormalizer, build_model
from app.models.model import Prediction, SuccessPredictor
from app.services.jobs import JobContext
from app.services.tribe_service import TribeService
from app.training.checkpoint import CheckpointManager
from app.tribe.types import BrainActivity

logger = get_logger(__name__)


class PredictionService:
    """Orchestrate ingest -> TRIBE -> model inference -> explanation."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        tribe_service: TribeService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.tribe_service = tribe_service or TribeService(settings=self.settings)

    def run(
        self,
        *,
        url: str | None = None,
        file_path: str | Path | None = None,
        confidence_level: float = 0.95,
        horizon_days: int | None = None,
        job: JobContext | None = None,
    ) -> dict[str, Any]:
        if (url is None) == (file_path is None):
            raise ValueError("provide exactly one of url or file_path")

        with session_scope() as session:
            builder = DatasetBuilder(session, settings=self.settings)
            if url is not None:
                video = builder.ingest_url(url)
            else:
                video = builder.ingest_local(file_path or "")

            if job is not None:
                job.set_progress(stage="predict", current=0, total=1, message="ingested")
                job.check_control()

            activity = self._ensure_brain(session, video, job=job)
            model, normalizer, model_horizon = self._load_model(session)
            horizon = horizon_days or model_horizon or 30

            device = resolve_device(self.settings.device)
            brain = torch.from_numpy(activity.array).unsqueeze(0)
            features = torch.from_numpy(build_feature_vector(video)).unsqueeze(0)
            pad_mask = torch.zeros(brain.size(0), brain.size(1), dtype=torch.bool)
            prediction = model.predict(
                brain.to(device),
                features.to(device),
                normalizer,
                pad_mask.to(device),
                confidence_level=confidence_level,
            )[0]
            explainer = Explainer(model, normalizer, device=device)
            explanation = explainer.explain_sample(
                brain.to(device),
                features.to(device),
                pad_mask=pad_mask.to(device),
                target=PRIMARY_TARGET,
                tr_seconds=activity.tr_seconds,
                hemodynamic_offset_seconds=activity.hemodynamic_offset_seconds,
            )

        return {
            "video_id": video.id,
            "youtube_id": video.youtube_id,
            "horizon_days": horizon,
            "horizon_label": f"This prediction corresponds to likes after {horizon} days.",
            "prediction": self._format_prediction(prediction, horizon),
            "explanation": explanation.as_dict(),
        }

    def _ensure_brain(
        self,
        session: Session,
        video: Video,
        *,
        job: JobContext | None,
    ) -> BrainActivity:
        from app.tribe.service import BrainActivityService

        service = BrainActivityService(session, extractor=self.tribe_service.extractor)
        if job is not None:
            job.set_progress(stage="tribe", current=0, total=1, message="brain activity")
        return service.get_or_compute(video, force=False)

    def _load_model(self, session: Session) -> tuple[SuccessPredictor, TargetNormalizer, int]:
        artifact = ModelRepository(session).get_active()
        horizon = 30
        if artifact is not None and artifact.checkpoint_path:
            cfg = artifact.config
            if isinstance(cfg, dict):
                horizon = int(
                    cfg.get(
                        "prediction_horizon_days",
                        cfg.get("model", {}).get("prediction_horizon_days", 30),
                    )
                )
                model_cfg = cfg.get("model", cfg)
            else:
                model_cfg = {}
            checkpoint_path = Path(artifact.checkpoint_path)
            if not checkpoint_path.exists():
                artifact = None
            else:
                model = build_model(model_cfg)
                payload = CheckpointManager(checkpoint_path.parent).load(
                    checkpoint_path, map_location="cpu"
                )
                model.load_state_dict(payload["model"])
                normalizer = TargetNormalizer.from_dict(
                    artifact.target_stats or payload["normalizer"]
                )
                return model, normalizer, horizon

        if artifact is not None:
            # Artifact exists but has no usable checkpoint; fall through to latest experiment.
            pass

        recent = ExperimentRepository(session).recent(limit=1)
        if not recent:
            raise RuntimeError("no trained model available")
        exp = recent[0]
        cfg = exp.config or {}
        horizon = int(cfg.get("prediction_horizon_days", 30))
        model_cfg = cfg.get("model", cfg)
        model = build_model(model_cfg)
        run_dir = Path(exp.run_dir or "")
        checkpoint = (
            CheckpointManager(run_dir / "checkpoints").best()
            or CheckpointManager(run_dir / "checkpoints").latest()
        )
        if checkpoint is None:
            raise RuntimeError("no checkpoint found for latest experiment")
        payload = CheckpointManager(checkpoint.parent).load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
        normalizer = TargetNormalizer.from_dict(payload["normalizer"])
        return model, normalizer, horizon

    @staticmethod
    def _format_prediction(prediction: Prediction, horizon_days: int) -> dict[str, Any]:
        values = dict(prediction.values)
        log_likes = float(values.get(Target.log_likes.value, 0.0))
        log_views = float(values.get(Target.log_views_30d.value, 0.0))
        return {
            "likes": max(0.0, 10**log_likes - 1.0),
            "views": max(0.0, 10**log_views - 1.0),
            "engagement": values.get(Target.engagement.value),
            "virality": values.get(Target.virality.value),
            "confidence_level": prediction.confidence_level,
            "point_estimates": values,
            "intervals": prediction.intervals,
            "uncertainty": prediction.uncertainty,
            "horizon_days": horizon_days,
            "horizon_label": f"This prediction corresponds to likes after {horizon_days} days.",
        }
