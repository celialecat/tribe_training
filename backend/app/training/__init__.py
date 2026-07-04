"""Training engine: loops, schedulers, checkpoints, export.

Public surface:
    - :class:`Trainer` / :class:`TrainConfig`   the training loop
    - :func:`build_dataloaders`                 data assembly + normaliser fit
    - :func:`build_optimizer` / :func:`build_scheduler`
    - :class:`CheckpointManager`                save / resume
    - :func:`export_torchscript` / :func:`export_onnx`
"""

from app.training.checkpoint import CheckpointManager, CheckpointState
from app.training.data import DataBundle, build_dataloaders
from app.training.early_stopping import EarlyStopping
from app.training.export import export_onnx, export_torchscript
from app.training.optim import build_optimizer
from app.training.scheduler import build_scheduler
from app.training.trainer import TrainConfig, Trainer
from app.training.utils import DistributedContext, detect_distributed, seed_everything

__all__ = [
    "CheckpointManager",
    "CheckpointState",
    "DataBundle",
    "DistributedContext",
    "EarlyStopping",
    "TrainConfig",
    "Trainer",
    "build_dataloaders",
    "build_optimizer",
    "build_scheduler",
    "detect_distributed",
    "export_onnx",
    "export_torchscript",
    "seed_everything",
]
