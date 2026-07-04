"""Trainer — the training loop with all production ergonomics.

Features: automatic mixed precision (bf16/fp16), gradient accumulation, gradient
clipping, cosine/plateau LR scheduling, checkpointing + full-state resume,
TensorBoard logging, early stopping and optional multi-GPU DistributedDataParallel.

The trainer is UI-agnostic: it emits progress through an optional ``callbacks``
object so the CLI can mirror epoch metrics into the Experiment table without the
trainer importing the database layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app.core.logging import get_logger
from app.models.losses import MaskedMultitaskLoss
from app.models.model import SuccessPredictor
from app.models.normalizer import TargetNormalizer
from app.training.checkpoint import CheckpointManager, CheckpointState
from app.training.early_stopping import EarlyStopping
from app.training.scheduler import is_step_based
from app.training.utils import AverageMeter, DistributedContext

logger = get_logger(__name__)


class TrainerCallbacks(Protocol):
    """Hooks the trainer invokes on the main process only."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...


@dataclass(slots=True)
class TrainConfig:
    """Resolved training hyper-parameters (from configs/training)."""

    epochs: int = 100
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    amp: bool = True
    amp_dtype: str = "bf16"
    log_every: int = 20
    checkpoint_every: int = 1
    keep_last_k: int = 3
    early_stopping: bool = True
    early_stopping_metric: str = "val/loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 12
    tensorboard: bool = True

    @classmethod
    def from_mapping(cls, cfg: Any) -> TrainConfig:
        return cls(
            epochs=int(cfg.get("epochs", 100)),
            grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
            grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
            amp=bool(cfg.get("amp", True)),
            amp_dtype=str(cfg.get("amp_dtype", "bf16")),
            log_every=int(cfg.get("log_every", 20)),
            checkpoint_every=int(cfg.get("checkpoint_every", 1)),
            keep_last_k=int(cfg.get("keep_last_k", 3)),
            early_stopping=bool(cfg.get("early_stopping", True)),
            early_stopping_metric=str(cfg.get("early_stopping_metric", "val/loss")),
            early_stopping_mode=str(cfg.get("early_stopping_mode", "min")),
            early_stopping_patience=int(cfg.get("early_stopping_patience", 12)),
            tensorboard=bool(cfg.get("tensorboard", True)),
        )


class Trainer:
    """Own the optimisation of a :class:`SuccessPredictor`."""

    def __init__(
        self,
        model: SuccessPredictor,
        *,
        loss_fn: MaskedMultitaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        normalizer: TargetNormalizer,
        config: TrainConfig,
        run_dir: str | Path,
        device: str = "cpu",
        dist: DistributedContext | None = None,
        full_config: dict[str, Any] | None = None,
        callbacks: TrainerCallbacks | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.core_model = model.to(self.device)
        self.dist = dist or DistributedContext()
        self.model: nn.Module = self._maybe_wrap_ddp(self.core_model)

        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.config = config
        self.full_config = full_config or {}
        self.callbacks = callbacks

        self.run_dir = Path(run_dir)
        self.ckpt = CheckpointManager(self.run_dir / "checkpoints", keep_last_k=config.keep_last_k)
        self.early = EarlyStopping(
            patience=config.early_stopping_patience, mode=config.early_stopping_mode
        )

        use_fp16 = config.amp and config.amp_dtype == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=use_fp16)
        self._autocast_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
        self._amp_enabled = config.amp and self.device.type in {"cuda", "cpu"}

        self.epoch = 0
        self.global_step = 0
        self.writer: SummaryWriter | None = (
            SummaryWriter(str(self.run_dir / "tb"))
            if config.tensorboard and self.dist.is_main
            else None
        )

    # ------------------------------------------------------------------ DDP --
    def _maybe_wrap_ddp(self, model: nn.Module) -> nn.Module:
        if not self.dist.enabled:
            return model
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self.dist.backend)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.dist.local_rank)
        device_ids = [self.dist.local_rank] if self.device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    # --------------------------------------------------------------- resume --
    def resume(self, checkpoint_path: str | Path) -> None:
        payload = self.ckpt.load(checkpoint_path, map_location=str(self.device))
        self.core_model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler"):
            self.scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))
        if payload.get("best_metric") is not None:
            self.early.best = float(payload["best_metric"])
        logger.info("Resumed at epoch %d (global_step %d)", self.epoch, self.global_step)

    # ------------------------------------------------------------------ fit --
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        train_sampler: Any | None = None,
    ) -> dict[str, float]:
        logger.info(
            "Training on %s for %d epochs (accum=%d, amp=%s/%s)",
            self.device, self.config.epochs, self.config.grad_accum_steps,
            self.config.amp, self.config.amp_dtype,
        )
        final_metrics: dict[str, float] = {}
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._evaluate(val_loader, prefix="val") if val_loader else {}
            metrics = {**train_metrics, **val_metrics}
            final_metrics = metrics

            self._epoch_scheduler_step(metrics)
            self._log_epoch(epoch, metrics)

            # Track "best" on the configured metric, falling back to train loss
            # when no validation split is available (small/bootstrap datasets).
            monitored = metrics.get(self.config.early_stopping_metric)
            if monitored is None:
                monitored = metrics.get("train/loss")
            is_best = monitored is not None and self.early.step(monitored)
            if self.dist.is_main and (epoch % self.config.checkpoint_every == 0 or is_best):
                self._save(metrics, is_best=is_best)
            if self.callbacks and self.dist.is_main:
                self.callbacks.on_epoch_end(epoch, metrics)

            if self.config.early_stopping and self.early.should_stop:
                logger.info("Early stopping at epoch %d (no improvement).", epoch)
                break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        return final_metrics

    # ----------------------------------------------------------- train step --
    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        meter = AverageMeter()
        accum = max(1, self.config.grad_accum_steps)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            batch = self._to_device(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._autocast_dtype,
                enabled=self._amp_enabled,
            ):
                mean, log_var = self.model(batch["brain"], batch["features"], batch["pad_mask"])
                target_z = self.normalizer.normalize(batch["targets"])
                loss, _ = self.loss_fn(mean, log_var, target_z, batch["target_mask"])
                loss = loss / accum

            self.scaler.scale(loss).backward()
            meter.update(float(loss.item()) * accum, n=batch["brain"].size(0))

            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.scheduler is not None and is_step_based(self.scheduler):
                    self.scheduler.step()
                if self.global_step % self.config.log_every == 0:
                    self._log_step(meter.average)

        return {"train/loss": meter.average, "lr": self._current_lr()}

    # ------------------------------------------------------------- evaluate --
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, *, prefix: str) -> dict[str, float]:
        self.model.eval()
        meter = AverageMeter()
        per_target: dict[str, AverageMeter] = {}
        for batch in loader:
            batch = self._to_device(batch)
            mean, log_var = self.model(batch["brain"], batch["features"], batch["pad_mask"])
            target_z = self.normalizer.normalize(batch["targets"])
            loss, logs = self.loss_fn(mean, log_var, target_z, batch["target_mask"])
            n = batch["brain"].size(0)
            meter.update(float(loss.item()), n=n)
            for name, value in logs.items():
                per_target.setdefault(name, AverageMeter()).update(float(value), n=n)

        metrics = {f"{prefix}/loss": meter.average}
        for name, m in per_target.items():
            metrics[f"{prefix}/mse/{name}"] = m.average
        return metrics

    # ------------------------------------------------------------- helpers --
    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _epoch_scheduler_step(self, metrics: dict[str, float]) -> None:
        if self.scheduler is None or is_step_based(self.scheduler):
            return
        # Plateau scheduler steps on the monitored metric.
        monitored = metrics.get(self.config.early_stopping_metric, metrics.get("train/loss", 0.0))
        self.scheduler.step(monitored)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_step(self, loss: float) -> None:
        logger.info(
            "step %d | train/loss %.4f | lr %.2e", self.global_step, loss, self._current_lr()
        )
        if self.writer is not None:
            self.writer.add_scalar("train/loss_step", loss, self.global_step)
            self.writer.add_scalar("train/lr", self._current_lr(), self.global_step)

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        summary = " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
        logger.info("epoch %d | %s", epoch, summary)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def _save(self, metrics: dict[str, float], *, is_best: bool) -> None:
        state = CheckpointState(
            model=self.core_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            scaler=self.scaler.state_dict() if self.scaler.is_enabled() else None,
            epoch=self.epoch,
            global_step=self.global_step,
            best_metric=self.early.best,
            normalizer=self.normalizer.to_dict(),
            config=self.full_config,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        self.ckpt.save(state, is_best=is_best)

    def load_iterable(self, batches: Iterable[dict[str, torch.Tensor]]) -> None:  # pragma: no cover
        """Reserved hook for streaming datasets; unused by the default loop."""
        raise NotImplementedError
