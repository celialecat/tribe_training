"""``ysp-train`` — Hydra entry point for training the success predictor.

Composes the config (``configs/``), builds dataloaders from the cached TRIBE
brain activity, trains a :class:`SuccessPredictor`, evaluates on the held-out
test split, exports TorchScript/ONNX, and registers the result in the model
registry. Training runs are tracked in the Experiment table so the dashboard can
render history live.

Examples
--------
    ysp-train                                  # defaults
    ysp-train +experiment=baseline
    ysp-train training.epochs=200 data.batch_size=32 model.brain_encoder.n_layers=6
    torchrun --nproc_per_node=4 -m app.training.cli training.ddp=true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from app.core.config import REPO_ROOT, get_settings
from app.core.device import describe_device, resolve_device
from app.core.logging import get_logger, setup_logging
from app.db.base import session_scope
from app.db.models.enums import ExperimentStatus
from app.db.models.experiment import Experiment
from app.db.models.model import ModelArtifact
from app.db.repositories.prediction import ExperimentRepository
from app.models import MaskedMultitaskLoss, build_model
from app.training.data import build_dataloaders
from app.training.export import export_onnx, export_torchscript
from app.training.optim import build_optimizer
from app.training.scheduler import build_scheduler
from app.training.trainer import TrainConfig, Trainer
from app.training.utils import detect_distributed, seed_everything

logger = get_logger(__name__)


class ExperimentTracker:
    """Callback mirroring epoch metrics into the Experiment table."""

    def __init__(self, experiment_id: int) -> None:
        self.experiment_id = experiment_id

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        with session_scope() as session:
            exp = session.get(Experiment, self.experiment_id)
            if exp is None:
                return
            exp.epoch = epoch
            exp.metrics = {k: float(v) for k, v in metrics.items()}
            monitored = metrics.get("val/loss")
            if monitored is not None and (exp.best_metric is None or monitored < exp.best_metric):
                exp.best_metric = float(monitored)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(level=get_settings().log_level)
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    dist = detect_distributed(bool(cfg.training.get("ddp", False)))
    logger.info("Run '%s' on %s", cfg.run_name, describe_device(device))

    run_dir = Path(cfg.output_dir)
    full_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    with session_scope() as session:
        bundle = build_dataloaders(
            session,
            data_cfg=cfg.data,
            tribe_model_id=str(cfg.tribe.model_id),
            seed=int(cfg.seed),
            dist=dist,
        )
        # Adapt the encoder's vertex count to the real TRIBE geometry.
        OmegaConf.update(cfg, "model.brain_encoder.input_vertices", bundle.n_vertices)

        model = build_model(cfg.model)
        logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

        loss_fn = MaskedMultitaskLoss(
            target_weights=OmegaConf.to_container(cfg.training.target_loss_weights, resolve=True),
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

        tracker = None
        experiment_id = None
        if dist.is_main:
            exp = ExperimentRepository(session).add(
                Experiment(
                    name=str(cfg.run_name),
                    status=ExperimentStatus.running,
                    run_dir=str(run_dir),
                    config=full_config,
                    total_epochs=int(cfg.training.epochs),
                )
            )
            session.flush()
            experiment_id = exp.id
            tracker = ExperimentTracker(experiment_id)

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
            callbacks=tracker,
        )
        if cfg.training.get("resume"):
            trainer.resume(str(cfg.training.resume))

        trainer.fit(bundle.train_loader, bundle.val_loader, train_sampler=bundle.train_sampler)

        if dist.is_main:
            _finalise(session, trainer, bundle, cfg, run_dir, experiment_id)


def _finalise(session, trainer, bundle, cfg, run_dir: Path, experiment_id: int | None) -> None:
    """Evaluate on test, export artefacts, register the model, close experiment."""
    test_metrics = (
        trainer._evaluate(bundle.test_loader, prefix="test") if bundle.test_loader else {}
    )
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in test_metrics.items()})

    # Load best weights for export/registration.
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
        config=OmegaConf.to_container(cfg.model, resolve=True),
        target_stats=bundle.normalizer.to_dict(),
        metrics=test_metrics,
    )
    from app.db.repositories.model import ModelRepository

    repo = ModelRepository(session)
    repo.add(artifact)
    if repo.get_active() is None:  # auto-activate the first trained model
        repo.set_active(artifact)

    if experiment_id is not None:
        exp = session.get(Experiment, experiment_id)
        if exp is not None:
            exp.status = ExperimentStatus.completed
            exp.metrics = {**exp.metrics, **test_metrics}


if __name__ == "__main__":  # pragma: no cover
    main()
