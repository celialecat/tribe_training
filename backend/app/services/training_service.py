"""Programmatic training orchestration used by the dashboard backend."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from sqlalchemy.orm import Session

from app.core.config import REPO_ROOT, Settings, get_settings
from app.core.device import describe_device, resolve_device
from app.core.logging import get_logger
from app.db.base import session_scope
from app.db.models.enums import ExperimentStatus
from app.db.models.experiment import Experiment
from app.db.models.model import ModelArtifact
from app.db.repositories.model import ModelRepository
from app.db.repositories.prediction import ExperimentRepository
from app.models import MaskedMultitaskLoss, build_model
from app.services.jobs import JobCancelled, JobContext
from app.training.cli import ExperimentTracker
from app.training.data import build_dataloaders
from app.training.export import export_onnx, export_torchscript
from app.training.optim import build_optimizer
from app.training.scheduler import build_scheduler
from app.training.trainer import TrainConfig, Trainer, TrainerCallbacks
from app.training.utils import detect_distributed, seed_everything

logger = get_logger(__name__)


class JobTrainerCallbacks(TrainerCallbacks):
    """Bridge training metrics into a job record and coop pause/stop."""

    def __init__(
        self,
        job: JobContext,
        *,
        experiment_tracker: ExperimentTracker | None = None,
        total_epochs: int = 0,
    ) -> None:
        self.job = job
        self.experiment_tracker = experiment_tracker
        self.total_epochs = total_epochs
        self.last_metrics: dict[str, float] = {}

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        self.job.check_control()
        self.last_metrics = {k: float(v) for k, v in metrics.items()}
        history = self.job.job.result.setdefault("history", []) if self.job.job.result else []
        if self.job.job.result is None:
            self.job.job.result = {"history": history}
        snapshot = {"epoch": epoch, **self.last_metrics}
        history.append(snapshot)
        self.job.job.result["history"] = history
        self.job.set_progress(
            stage="train",
            current=epoch + 1,
            total=self.total_epochs,
            message=f"epoch={epoch}",
        )
        if self.experiment_tracker is not None:
            self.experiment_tracker.on_epoch_end(epoch, self.last_metrics)
        if self.job.is_cancelled():
            raise JobCancelled(self.job.job.id)


class TrainingService:
    """Compose the existing training stack without Hydra entrypoint plumbing."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

    def load_config(self, overrides: dict[str, Any] | None = None) -> Any:
        base = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
        data = OmegaConf.load(REPO_ROOT / "configs" / "data" / "default.yaml")
        model = OmegaConf.load(REPO_ROOT / "configs" / "model" / "default.yaml")
        training = OmegaConf.load(REPO_ROOT / "configs" / "training" / "default.yaml")
        tribe = OmegaConf.load(REPO_ROOT / "configs" / "tribe" / "default.yaml")
        cfg = OmegaConf.merge(
            base, {"data": data, "model": model, "training": training, "tribe": tribe}
        )
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        return cfg

    def run(
        self,
        *,
        overrides: dict[str, Any] | None = None,
        job: JobContext | None = None,
    ) -> dict[str, Any]:
        cfg = self.load_config(overrides)
        seed_everything(int(cfg.seed))
        device = resolve_device(str(cfg.device))
        dist = detect_distributed(bool(cfg.training.get("ddp", False)))
        run_name = str(cfg.run_name)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = cfg.get("output_dir")
        if isinstance(output_dir, str) and not output_dir.startswith("${"):
            run_dir = Path(output_dir)
        else:
            run_dir = REPO_ROOT / "outputs" / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        full_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[assignment]
        logger.info("Training '%s' on %s", run_name, describe_device(device))

        with session_scope() as session:
            bundle = build_dataloaders(
                session,
                data_cfg=cfg.data,
                tribe_model_id=str(cfg.tribe.model_id),
                seed=int(cfg.seed),
                dist=dist,
            )
            OmegaConf.update(cfg, "model.brain_encoder.input_vertices", bundle.n_vertices)
            model = build_model(cfg.model)
            loss_fn = MaskedMultitaskLoss(
                target_weights=cast(
                    dict[str, float],
                    OmegaConf.to_container(cfg.training.target_loss_weights, resolve=True),
                ),
                heteroscedastic=bool(cfg.model.predictor.heteroscedastic),
            )
            optimizer = build_optimizer(model, cfg=cfg.training)
            accum = max(1, int(cfg.training.grad_accum_steps))
            steps_per_epoch = max(1, len(bundle.train_loader) // accum)
            scheduler = build_scheduler(
                optimizer,
                kind=str(cfg.training.scheduler),
                total_steps=int(cfg.training.epochs) * steps_per_epoch,
                warmup_steps=int(cfg.training.get("warmup_steps", 0)),
                min_lr=float(cfg.training.get("min_lr", 1e-6)),
                base_lr=float(cfg.training.lr),
            )

            experiment_id = None
            tracker = None
            if dist.is_main:
                experiment = Experiment(
                    name=run_name,
                    status=ExperimentStatus.running,
                    run_dir=str(run_dir),
                    config=full_config,
                    total_epochs=int(cfg.training.epochs),
                )
                exp = ExperimentRepository(session).add(experiment)
                session.flush()
                experiment_id = exp.id
                tracker = ExperimentTracker(experiment_id)

            callbacks = (
                JobTrainerCallbacks(
                    job,
                    experiment_tracker=tracker,
                    total_epochs=int(cfg.training.epochs),
                )
                if job is not None
                else None
            )
            trainer = Trainer(
                model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=bundle.normalizer,
                config=TrainConfig.from_mapping(cfg.training),
                run_dir=run_dir,
                device=device,
                dist=dist,
                full_config=full_config,
                callbacks=callbacks,
            )

            resume_path = self._resolve_resume(cfg.training.get("resume"), run_dir)
            if resume_path is not None:
                trainer.resume(resume_path)

            final_metrics: dict[str, Any] = {}
            try:
                final_metrics = trainer.fit(
                    bundle.train_loader,
                    bundle.val_loader,
                    train_sampler=bundle.train_sampler,
                )
            except JobCancelled:
                if callbacks is not None:
                    trainer._save(callbacks.last_metrics, is_best=False)
                raise

            if dist.is_main:
                final_metrics = self._finalise(
                    session, trainer, bundle, cfg, run_dir, experiment_id, final_metrics
                )

        return {
            "run_dir": str(run_dir),
            "device": device,
            "metrics": final_metrics,
        }

    def _resolve_resume(self, resume: object, run_dir: Path) -> Path | None:
        if not resume:
            return None
        if str(resume) != "auto":
            return Path(str(resume))
        candidates = [
            ckpt
            for ckpt in run_dir.parent.glob("*/checkpoints/last.pt")
            if ckpt.is_file() and run_dir not in ckpt.parents
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _finalise(
        self,
        session: Session,
        trainer: Trainer,
        bundle: Any,
        cfg: Any,
        run_dir: Path,
        experiment_id: int | None,
        final_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        test_metrics = (
            trainer._evaluate(bundle.test_loader, prefix="test") if bundle.test_loader else {}
        )
        best = trainer.ckpt.best() or trainer.ckpt.latest()
        if best is not None:
            payload = trainer.ckpt.load(best, map_location="cpu")
            trainer.core_model.load_state_dict(payload["model"])

        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts_path = export_torchscript(trainer.core_model.cpu(), export_dir / "model.torchscript")
        onnx_path = export_onnx(trainer.core_model.cpu(), export_dir / "model.onnx")

        artifact = ModelArtifact(
            name=str(cfg.run_name),
            version=f"e{experiment_id or 0}",
            checkpoint_path=str(best) if best else "",
            torchscript_path=str(ts_path) if ts_path else None,
            onnx_path=str(onnx_path) if onnx_path else None,
            config=OmegaConf.to_container(cfg, resolve=False),
            target_stats=bundle.normalizer.to_dict(),
            metrics={**final_metrics, **test_metrics},
        )
        repo = ModelRepository(session)
        repo.add(artifact)
        repo.set_active(artifact)

        if experiment_id is not None:
            exp = session.get(Experiment, experiment_id)
            if exp is not None:
                exp.status = ExperimentStatus.completed
                exp.metrics = {**exp.metrics, **test_metrics}
        return {**final_metrics, **test_metrics}
