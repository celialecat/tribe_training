"""Dataset builder: download, metadata and media extraction.

Public surface:
    - :class:`DatasetBuilder`      end-to-end ingestion into DB + filesystem
    - :func:`build_feature_vector` causal metadata features for the model
    - :func:`compute_targets`      observed labels + supervision mask
"""

from app.dataset.builder import BatchIngestReport, DatasetBuilder, IngestionError
from app.dataset.features import FEATURE_DIM, FEATURE_NAMES, build_feature_vector
from app.dataset.report import DatasetReport, build_dataset_report
from app.dataset.schemas import MediaArtifacts, VideoMetadata
from app.dataset.targets import TargetSample, compute_targets
from app.dataset.validation import ValidationResult, VideoValidator

__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "BatchIngestReport",
    "DatasetBuilder",
    "DatasetReport",
    "IngestionError",
    "MediaArtifacts",
    "TargetSample",
    "ValidationResult",
    "VideoMetadata",
    "VideoValidator",
    "build_dataset_report",
    "build_feature_vector",
    "compute_targets",
]
