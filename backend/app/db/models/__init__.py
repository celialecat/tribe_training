"""SQLAlchemy ORM models.

Importing this package registers every mapper on ``Base.metadata``; Alembic's
``env.py`` and ``app.db.base.create_all`` both rely on that side effect.
"""

from app.db.models.embedding import Embedding
from app.db.models.enums import (
    BrainStatus,
    EmbeddingKind,
    ExperimentStatus,
    VideoSource,
    VideoStatus,
)
from app.db.models.experiment import Experiment
from app.db.models.model import ModelArtifact
from app.db.models.prediction import Prediction
from app.db.models.user import User
from app.db.models.video import Video

__all__ = [
    "BrainStatus",
    "Embedding",
    "EmbeddingKind",
    "Experiment",
    "ExperimentStatus",
    "ModelArtifact",
    "Prediction",
    "User",
    "Video",
    "VideoSource",
    "VideoStatus",
]
