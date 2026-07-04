"""BrainEmbeddingGenerator — encode brain activity into 512-d latents.

Wraps a trained Brain Encoder to turn a video's ``(T, V)`` brain activity into a
compact latent vector, with optional persistence to the embedding cache. Used
both to pre-compute latents for fast downstream experiments and at inference
time. The encoder is any module implementing ``encode(brain, pad_mask)`` or a
plain ``forward`` returning the latent (see :mod:`app.models.brain_encoder`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.device import resolve_device
from app.core.logging import get_logger
from app.db.models.embedding import Embedding
from app.db.models.enums import EmbeddingKind
from app.db.models.video import Video
from app.db.repositories.embedding import EmbeddingRepository
from app.tribe.types import BrainActivity

logger = get_logger(__name__)


class BrainEmbeddingGenerator:
    """Encode brain activity to latent vectors and (optionally) cache them."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        *,
        session: Session | None = None,
        producer: str = "encoder:default",
        device: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.device = resolve_device(device or self.settings.device)
        self.encoder = encoder.to(self.device).eval()
        self.session = session
        self.producer = producer
        self.root = self.settings.cache_dir / "brain_latents"
        self.root.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def encode_array(self, brain: np.ndarray, pad_mask: np.ndarray | None = None) -> np.ndarray:
        """Encode a single ``(T, V)`` array into a ``(latent_dim,)`` vector."""
        x = torch.from_numpy(np.ascontiguousarray(brain, dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)  # (1, T, V)
        mask = None
        if pad_mask is not None:
            mask = torch.from_numpy(pad_mask).unsqueeze(0).to(self.device)  # (1, T)
        latent = self._forward(x, mask)
        return latent.squeeze(0).float().cpu().numpy()

    def encode_activity(self, activity: BrainActivity) -> np.ndarray:
        return self.encode_array(activity.array)

    def generate_and_cache(self, video: Video, activity: BrainActivity) -> Embedding:
        """Encode + persist the latent for a video. Requires a DB session."""
        if self.session is None:
            raise RuntimeError("A DB session is required to cache latents.")
        latent = self.encode_activity(activity)
        path = self.root / f"{video.id}_{self.producer.replace(':', '_')}.npy"
        np.save(path, latent.astype(np.float32))
        embedding = Embedding(
            video_id=video.id,
            kind=EmbeddingKind.brain_latent,
            producer=self.producer,
            path=self._rel(path),
            shape=list(latent.shape),
            dtype="float32",
            num_features=int(latent.shape[0]),
        )
        record = EmbeddingRepository(self.session).upsert(embedding)
        self.session.flush()
        logger.debug("Cached brain latent for video %s (dim=%d)", video.id, latent.shape[0])
        return record

    # ------------------------------------------------------------ helpers --
    def _forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(x, pad_mask=mask)
        return self.encoder(x, pad_mask=mask)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.settings.data_dir.resolve()))
        except ValueError:
            return str(path.resolve())
