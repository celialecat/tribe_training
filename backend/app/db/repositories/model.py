"""Model-registry repository."""

from __future__ import annotations

from sqlalchemy import select, update

from app.db.models.model import ModelArtifact
from app.db.repositories.base import BaseRepository


class ModelRepository(BaseRepository[ModelArtifact]):
    model = ModelArtifact

    def get_active(self) -> ModelArtifact | None:
        return self.session.scalar(select(ModelArtifact).where(ModelArtifact.is_active.is_(True)))

    def get_by_name_version(self, name: str, version: str) -> ModelArtifact | None:
        return self.session.scalar(
            select(ModelArtifact).where(
                ModelArtifact.name == name, ModelArtifact.version == version
            )
        )

    def set_active(self, artifact: ModelArtifact) -> ModelArtifact:
        """Promote one artifact to active, demoting all others atomically."""
        self.session.execute(update(ModelArtifact).values(is_active=False))
        artifact.is_active = True
        self.session.flush()
        return artifact
