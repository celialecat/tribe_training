"""BrainCache — persistent cache of TRIBE brain-activity artefacts.

TRIBE v2 is a ~1B-parameter model; running it is by far the most expensive step
in the pipeline. This cache guarantees each (video, model) pair is inferred at
most once. Arrays are stored compressed on disk under the cache directory; an
:class:`Embedding` row records the path, shape and a content hash so staleness
is detectable and recomputation avoidable across processes and restarts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models.embedding import Embedding
from app.db.models.enums import BrainStatus, EmbeddingKind
from app.db.models.video import Video
from app.db.repositories.embedding import EmbeddingRepository
from app.tribe.types import BrainActivity

logger = get_logger(__name__)

_HASH_CHUNK = 1 << 20  # 1 MiB streaming hash chunks


def hash_file(path: Path) -> str:
    """Stream a SHA-256 of a file's contents (memory-bounded)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def content_key(video_path: Path, model_id: str) -> str:
    """Deterministic cache key from the source media + producing model."""
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(hash_file(video_path).encode())
    return digest.hexdigest()[:32]


class BrainCache:
    """Disk + database cache for TRIBE brain-activity arrays."""

    def __init__(
        self,
        session: Session,
        *,
        model_id: str,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.model_id = model_id
        self.producer = f"tribev2:{model_id}"
        self.settings = settings or get_settings()
        self.root = self.settings.tribe_cache_dir / "brain_activity"
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo = EmbeddingRepository(session)

    # ---------------------------------------------------------------- paths --
    def _npz_path(self, video: Video, key: str) -> Path:
        return self.root / f"{video.id}_{key}.npz"

    def _record(self, video: Video) -> Embedding | None:
        return self.repo.find(video.id, EmbeddingKind.brain_activity, self.producer)

    # --------------------------------------------------------------- lookup --
    def has(self, video: Video) -> bool:
        record = self._record(video)
        return record is not None and Path(self._abs(record.path)).exists()

    def get(self, video: Video) -> BrainActivity | None:
        """Load cached brain activity for a video, or None if absent."""
        record = self._record(video)
        if record is None:
            return None
        path = Path(self._abs(record.path))
        if not path.exists():
            logger.warning("Cache row for video %s points at missing file %s", video.id, path)
            return None
        with np.load(path) as data:
            return BrainActivity(
                array=data["activity"].astype(np.float32),
                tr_seconds=float(data["tr_seconds"]),
                hemodynamic_offset_seconds=float(data["offset"]),
                model_id=self.model_id,
            )

    # ----------------------------------------------------------------- put --
    def put(self, video: Video, activity: BrainActivity, *, source_path: Path) -> Embedding:
        """Persist brain activity to disk and upsert its Embedding row."""
        key = content_key(source_path, self.model_id)
        path = self._npz_path(video, key)
        np.savez_compressed(
            path,
            activity=activity.array.astype(np.float32),
            tr_seconds=np.float32(activity.tr_seconds),
            offset=np.float32(activity.hemodynamic_offset_seconds),
        )
        embedding = Embedding(
            video_id=video.id,
            kind=EmbeddingKind.brain_activity,
            producer=self.producer,
            path=self._rel(path),
            shape=[activity.n_timesteps, activity.n_vertices],
            dtype="float32",
            num_timesteps=activity.n_timesteps,
            num_features=activity.n_vertices,
            content_hash=key,
        )
        record = self.repo.upsert(embedding)
        video.brain_status = BrainStatus.cached
        self.session.flush()
        logger.info(
            "Cached brain activity for video %s: %sx%s (%.1f MB)",
            video.id, activity.n_timesteps, activity.n_vertices,
            path.stat().st_size / 1e6,
        )
        return record

    # ------------------------------------------------------------ path util --
    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.settings.data_dir.resolve()))
        except ValueError:
            return str(path.resolve())

    def _abs(self, stored: str) -> str:
        p = Path(stored)
        return str(p if p.is_absolute() else self.settings.data_dir / p)
