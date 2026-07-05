"""TribeExtractor — thin, faithful wrapper around the official TRIBE v2 model.

This is the ONLY place the platform touches Meta's `tribev2` package, and it
uses the upstream API exactly as documented — nothing is mocked or
reimplemented::

    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)
    events = model.get_events_dataframe(video_path=..., audio_path=..., text_path=...)
    preds, segments = model.predict(events=events)   # preds: (T, n_vertices)

The heavy model is loaded lazily and cached on the instance so a single
extractor can process many videos. If `tribev2` is not installed, a clear
:class:`TribeNotInstalled` error is raised pointing at the install docs — we
never silently substitute fake activity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.constants import DEFAULT_TR_SECONDS
from app.core.device import resolve_device
from app.core.logging import get_logger
from app.tribe.types import BrainActivity

logger = get_logger(__name__)

# TRIBE shifts predictions 5 s backward to account for haemodynamic delay.
TRIBE_HEMODYNAMIC_OFFSET_SECONDS = 5.0


class TribeNotInstalled(RuntimeError):  # noqa: N818 - deliberate, reads better than *Error
    """Raised when the official `tribev2` package/weights are unavailable."""

    INSTALL_HINT = (
        "The official TRIBE v2 package is required and was not found.\n"
        "Install it (on a CUDA host) with:\n"
        "    make install-tribe\n"
        "or manually:\n"
        "    git clone https://github.com/facebookresearch/tribev2\n"
        '    pip install -e "./tribev2[training]"\n'
        "Weights (facebook/tribev2, CC-BY-NC-4.0) download from the HF Hub on "
        "first use. See docs/installation.md."
    )

    def __init__(self, cause: Exception | None = None) -> None:
        super().__init__(self.INSTALL_HINT)
        self.__cause__ = cause


class TribeExtractor:
    """Run TRIBE v2 inference on videos to obtain predicted brain activity."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        cache_folder: Path | None = None,
        device: str | None = None,
        modalities: Sequence[str] = ("video", "audio", "text"),
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_id = model_id or self.settings.tribe_model_id
        self.cache_folder = Path(cache_folder or self.settings.tribe_cache_dir) / "weights"
        self.device = resolve_device(device or self.settings.tribe_device)
        self.modalities = tuple(modalities)
        self._model: Any | None = None

    # ------------------------------------------------------------- loading --
    @staticmethod
    def _import_tribe() -> Any:
        try:
            from tribev2 import TribeModel  # type: ignore[import-not-found]
        except Exception as exc:
            raise TribeNotInstalled(exc) from exc
        return TribeModel

    @property
    def model(self) -> Any:
        """Lazily load and cache the TRIBE model on first access."""
        if self._model is None:
            TribeModel = self._import_tribe()  # noqa: N806 - mirror upstream class name
            self.cache_folder.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Loading TRIBE v2 weights %s onto %s (cache=%s)",
                self.model_id, self.device, self.cache_folder,
            )
            model = TribeModel.from_pretrained(
                self.model_id, cache_folder=str(self.cache_folder)
            )
            # Best-effort device placement; upstream may already handle this.
            for mover in ("to", "cuda"):
                if self.device.startswith("cuda") and hasattr(model, mover):
                    try:
                        model = model.to(self.device) if mover == "to" else model.cuda()
                        break
                    except Exception:
                        logger.debug("model.%s(%s) not applicable", mover, self.device)
            self._model = model
        return self._model

    def is_available(self) -> bool:
        """True if `tribev2` can be imported (does not download weights)."""
        try:
            self._import_tribe()
            return True
        except TribeNotInstalled:
            return False

    # ---------------------------------------------------------- inference --
    def extract(
        self,
        video_path: str | Path,
        *,
        audio_path: str | Path | None = None,
        transcript: str | None = None,
    ) -> BrainActivity:
        """Run TRIBE v2 on a video and return predicted brain activity.

        ``audio_path`` / ``transcript`` are optional extra modalities; when
        absent, TRIBE derives what it can from the video itself.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # Upstream accepts exactly ONE source; a video is trimodal on its own
        # (audio and text events are derived from it internally).
        events = self.model.get_events_dataframe(video_path=str(video_path))
        preds, _segments = self.model.predict(events=events)

        array = self._to_numpy(preds)
        if array.ndim != 2:
            raise ValueError(
                f"Expected TRIBE preds of shape (T, V); got {array.shape}."
            )
        logger.info("TRIBE produced brain activity of shape %s", array.shape)
        return BrainActivity(
            array=array.astype(np.float32),
            tr_seconds=DEFAULT_TR_SECONDS,
            hemodynamic_offset_seconds=TRIBE_HEMODYNAMIC_OFFSET_SECONDS,
            model_id=self.model_id,
        )

    @staticmethod
    def _to_numpy(preds: Any) -> np.ndarray:
        """Coerce TRIBE's prediction (tensor / DataFrame / array) to ndarray."""
        if isinstance(preds, np.ndarray):
            return preds
        # torch.Tensor
        if hasattr(preds, "detach"):
            return preds.detach().cpu().numpy()
        # pandas.DataFrame
        if hasattr(preds, "to_numpy"):
            return preds.to_numpy()
        return np.asarray(preds)
