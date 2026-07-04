"""Official Meta TRIBE v2 integration and brain-activity caching.

Public surface
--------------
- :class:`TribeExtractor`         faithful wrapper over the official `tribev2` API
- :class:`BrainCache`             disk + DB cache of predicted brain activity
- :class:`BrainActivityService`   compute-or-load orchestration
- :class:`BrainActivity`          the ``(T, V)`` prediction container
- :class:`BrainDataset`           PyTorch dataset (torch-only, lazily exported)
- :class:`BrainEmbeddingGenerator` encode activity -> latent (torch-only, lazy)

``BrainDataset`` and ``BrainEmbeddingGenerator`` require PyTorch, so they are
imported lazily via ``__getattr__``: ``import app.tribe`` stays usable (and the
FastAPI/CLI layers importable) on hosts without the DL stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.tribe.cache import BrainCache
from app.tribe.extractor import TribeExtractor, TribeNotInstalled
from app.tribe.service import BrainActivityService
from app.tribe.types import BrainActivity

if TYPE_CHECKING:
    from app.tribe.dataset import BrainDataset, Sample, make_splits
    from app.tribe.embedding import BrainEmbeddingGenerator

_LAZY = {
    "BrainDataset": "app.tribe.dataset",
    "Sample": "app.tribe.dataset",
    "make_splits": "app.tribe.dataset",
    "BrainEmbeddingGenerator": "app.tribe.embedding",
}

__all__ = [
    "BrainActivity",
    "BrainActivityService",
    "BrainCache",
    "BrainDataset",
    "BrainEmbeddingGenerator",
    "Sample",
    "TribeExtractor",
    "TribeNotInstalled",
    "make_splits",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
