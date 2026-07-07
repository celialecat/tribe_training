"""Model evaluation orchestration for dashboard use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from app.core.config import Settings, get_settings
from app.core.device import resolve_device
from app.core.logging import get_logger
from app.db.base import session_scope
from app.evaluation.evaluator import Evaluator
from app.services.jobs import JobContext
from app.services.prediction_service import PredictionService
from app.training.data import build_dataloaders

logger = get_logger(__name__)


class EvaluationService:
    """Evaluate the latest trained checkpoint on the test split."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.prediction_service = PredictionService(settings=self.settings)

    def run(self, *, job: JobContext | None = None) -> dict[str, Any]:
        with session_scope() as session:
            model, normalizer, _horizon = self.prediction_service._load_model(session)
            bundle = build_dataloaders(
                session,
                data_cfg=self._data_cfg(),
                tribe_model_id=str(self.settings.tribe_model_id),
            )
            if bundle.test_loader is None:
                return {"metrics": {}, "residuals": {}, "calibration": {}, "n_samples": 0}
            evaluator = Evaluator(model, normalizer, device=resolve_device(self.settings.device))
            report = evaluator.evaluate(bundle.test_loader)
            if job is not None:
                job.set_progress(stage="evaluate", current=1, total=1, message="complete")
            return report

    def _data_cfg(self) -> dict[str, Any]:
        from omegaconf import OmegaConf

        return cast(
            dict[str, Any],
            OmegaConf.to_container(
                OmegaConf.load(
                    Path(__file__).resolve().parents[3] / "configs" / "data" / "default.yaml"
                ),
                resolve=True,
            ),
        )
