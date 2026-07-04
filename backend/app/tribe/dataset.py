"""BrainDataset — a PyTorch dataset over cached TRIBE brain activity.

Each sample couples a video's brain-activity tensor ``(T, V)`` with its causal
metadata features and its (masked) success targets. Brain arrays are memory-
mapped from disk and loaded lazily per item so datasets far larger than RAM are
trainable. Variable-length sequences are right-padded to a fixed horizon with an
accompanying key-padding mask so the encoder can ignore padding.

Splits are made at the *channel* level by default: all videos from one channel
land in the same split, preventing a channel's style/audience leaking between
train and test and inflating metrics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sqlalchemy.orm import Session
from torch.utils.data import Dataset

from app.core.config import Settings, get_settings
from app.core.constants import NUM_TARGETS
from app.core.logging import get_logger
from app.dataset.features import FEATURE_DIM, build_feature_vector
from app.dataset.targets import compute_targets
from app.db.models.embedding import Embedding
from app.db.models.enums import EmbeddingKind
from app.db.models.video import Video
from app.db.repositories.video import VideoRepository

logger = get_logger(__name__)

Split = Literal["train", "val", "test"]


@dataclass(slots=True)
class Sample:
    """A precomputed, disk-backed training example."""

    video_id: int
    brain_path: Path
    features: np.ndarray   # (FEATURE_DIM,)
    targets: np.ndarray    # (NUM_TARGETS,)
    mask: np.ndarray       # (NUM_TARGETS,)


def _stable_fraction(key: str, seed: int) -> float:
    """Deterministic hash of a key -> float in [0, 1) for reproducible splits."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def make_splits(
    session: Session,
    *,
    val_fraction: float,
    test_fraction: float,
    split_by: str = "channel",
    seed: int = 42,
) -> dict[int, Split]:
    """Assign each training-ready video to a split, deterministically.

    When ``split_by == "channel"`` the split key is the channel id (falling back
    to per-video when a channel is unknown), so channels never straddle splits.
    """
    videos = VideoRepository(session).list_ready_for_training()
    assignment: dict[int, Split] = {}
    for video in videos:
        if split_by == "channel" and video.channel_id:
            key = f"chan:{video.channel_id}"
        else:
            key = f"vid:{video.id}"
        frac = _stable_fraction(key, seed)
        if frac < test_fraction:
            assignment[video.id] = "test"
        elif frac < test_fraction + val_fraction:
            assignment[video.id] = "val"
        else:
            assignment[video.id] = "train"
    return assignment


class BrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset yielding brain activity + features + masked targets."""

    def __init__(
        self,
        samples: list[Sample],
        *,
        max_time_steps: int = 256,
        pad_value: float = 0.0,
        n_vertices: int | None = None,
    ) -> None:
        self.samples = samples
        self.max_time_steps = max_time_steps
        self.pad_value = pad_value
        self.n_vertices = n_vertices or self._infer_vertices(samples)

    # ---------------------------------------------------------- factories --
    @classmethod
    def from_database(
        cls,
        session: Session,
        *,
        split: Split,
        assignment: dict[int, Split],
        model_id: str,
        approximate_view_curve: bool = True,
        settings: Settings | None = None,
        max_time_steps: int = 256,
        pad_value: float = 0.0,
    ) -> BrainDataset:
        """Build a dataset for one split from cached brain activity + labels."""
        settings = settings or get_settings()
        producer = f"tribev2:{model_id}"
        video_ids = [vid for vid, sp in assignment.items() if sp == split]
        samples: list[Sample] = []
        for video_id in video_ids:
            video = session.get(Video, video_id)
            if video is None:
                continue
            record = _brain_record(session, video_id, producer)
            if record is None:
                continue  # brain activity not computed yet -> skip
            target = compute_targets(video, approximate_view_curve=approximate_view_curve)
            if float(target.mask.sum()) == 0.0:
                continue  # no supervised targets -> unusable
            samples.append(
                Sample(
                    video_id=video_id,
                    brain_path=Path(_abs(record.path, settings)),
                    features=build_feature_vector(video),
                    targets=target.values,
                    mask=target.mask,
                )
            )
        logger.info("Built %s dataset: %d samples", split, len(samples))
        return cls(samples, max_time_steps=max_time_steps, pad_value=pad_value)

    # ------------------------------------------------------------- protocol --
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with np.load(sample.brain_path, mmap_mode="r") as data:
            brain = np.asarray(data["activity"], dtype=np.float32)  # (T, V)
        brain, pad_mask = self._fit_length(brain)
        return {
            "brain": torch.from_numpy(brain),                       # (T, V)
            "pad_mask": torch.from_numpy(pad_mask),                 # (T,) True=pad
            "features": torch.from_numpy(sample.features),          # (FEATURE_DIM,)
            "targets": torch.from_numpy(sample.targets),            # (NUM_TARGETS,)
            "target_mask": torch.from_numpy(sample.mask),           # (NUM_TARGETS,)
            "video_id": torch.tensor(sample.video_id, dtype=torch.long),
        }

    # -------------------------------------------------------------- helpers --
    def _fit_length(self, brain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Right-pad or centre-crop to ``max_time_steps``; return (brain, pad_mask)."""
        t, v = brain.shape
        target_t = self.max_time_steps
        pad_mask = np.zeros(target_t, dtype=bool)
        if t == target_t:
            return brain, pad_mask
        if t > target_t:  # centre crop keeps the most informative middle section
            start = (t - target_t) // 2
            return brain[start : start + target_t], pad_mask
        padded = np.full((target_t, v), self.pad_value, dtype=np.float32)
        padded[:t] = brain
        pad_mask[t:] = True
        return padded, pad_mask

    @staticmethod
    def _infer_vertices(samples: list[Sample]) -> int:
        if not samples:
            return 0
        with np.load(samples[0].brain_path, mmap_mode="r") as data:
            return int(data["activity"].shape[1])

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def num_targets(self) -> int:
        return NUM_TARGETS


def _brain_record(session: Session, video_id: int, producer: str) -> Embedding | None:
    from app.db.repositories.embedding import EmbeddingRepository

    return EmbeddingRepository(session).find(video_id, EmbeddingKind.brain_activity, producer)


def _abs(stored: str, settings: Settings) -> str:
    p = Path(stored)
    return str(p if p.is_absolute() else settings.data_dir / p)
