"""Assemble train/val/test dataloaders from cached brain activity.

Builds :class:`BrainDataset` splits, fits the :class:`TargetNormalizer` on the
*training* targets only (respecting the mask), and wires up dataloaders —
including a ``DistributedSampler`` for the training split under DDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from app.core.logging import get_logger
from app.models.normalizer import TargetNormalizer
from app.training.utils import DistributedContext
from app.tribe.dataset import BrainDataset, make_splits

logger = get_logger(__name__)


@dataclass(slots=True)
class DataBundle:
    """Everything the trainer needs about the data."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None
    normalizer: TargetNormalizer
    train_sampler: DistributedSampler | None
    n_vertices: int
    sizes: dict[str, int]


def _fit_normalizer(dataset: BrainDataset) -> TargetNormalizer:
    if len(dataset) == 0:
        return TargetNormalizer.identity()
    values = np.stack([s.targets for s in dataset.samples])
    mask = np.stack([s.mask for s in dataset.samples])
    return TargetNormalizer.fit(values, mask)


def build_dataloaders(
    session: Session,
    *,
    data_cfg: Any,
    tribe_model_id: str,
    seed: int = 42,
    dist: DistributedContext | None = None,
) -> DataBundle:
    """Construct dataloaders + normalizer from the database."""
    dist = dist or DistributedContext()
    assignment = make_splits(
        session,
        val_fraction=float(data_cfg.get("val_fraction", 0.15)),
        test_fraction=float(data_cfg.get("test_fraction", 0.15)),
        split_by=str(data_cfg.get("split_by", "channel")),
        seed=seed,
    )

    def make(split: str) -> BrainDataset:
        return BrainDataset.from_database(
            session,
            split=split,  # type: ignore[arg-type]
            assignment=assignment,
            model_id=tribe_model_id,
            approximate_view_curve=bool(data_cfg.get("approximate_view_curve", True)),
            max_time_steps=int(data_cfg.get("max_time_steps", 256)),
            pad_value=float(data_cfg.get("pad_value", 0.0)),
        )

    train_ds, val_ds, test_ds = make("train"), make("val"), make("test")
    if len(train_ds) == 0:
        raise RuntimeError(
            "No training samples found. Build a dataset and compute TRIBE brain "
            "activity first (see docs/training.md)."
        )
    normalizer = _fit_normalizer(train_ds)

    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
        if dist.enabled
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = _eval_loader(val_ds, batch_size, num_workers, pin_memory)
    test_loader = _eval_loader(test_ds, batch_size, num_workers, pin_memory)

    sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    logger.info("Dataloaders ready: %s | vertices=%d", sizes, train_ds.n_vertices)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        train_sampler=train_sampler,
        n_vertices=train_ds.n_vertices,
        sizes=sizes,
    )


def _eval_loader(
    dataset: BrainDataset, batch_size: int, num_workers: int, pin_memory: bool
) -> DataLoader | None:
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
