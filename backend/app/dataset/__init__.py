"""Dataset builder: download, metadata and media extraction.

Public surface:
    - :class:`DatasetBuilder`      end-to-end ingestion into DB + filesystem
    - :func:`build_feature_vector` causal metadata features for the model
    - :func:`compute_targets`      observed labels + supervision mask
"""

from app.dataset.builder import DatasetBuilder
from app.dataset.features import FEATURE_DIM, FEATURE_NAMES, build_feature_vector
from app.dataset.schemas import MediaArtifacts, VideoMetadata
from app.dataset.targets import TargetSample, compute_targets

__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "DatasetBuilder",
    "MediaArtifacts",
    "TargetSample",
    "VideoMetadata",
    "build_feature_vector",
    "compute_targets",
]
