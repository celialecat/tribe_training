"""Generic repository base.

Repositories are the only place raw ORM queries live; services and the API talk
to repositories, never to the session directly. This keeps data-access logic
testable and swappable. The base provides the common CRUD verbs; concrete
repositories add domain-specific queries.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD operations over a single ORM model class."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, obj_id: int) -> ModelT | None:
        return self.session.get(self.model, obj_id)

    def add(self, obj: ModelT) -> ModelT:
        self.session.add(obj)
        self.session.flush()  # populate PKs without ending the transaction
        return obj

    def delete(self, obj: ModelT) -> None:
        self.session.delete(obj)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(self.model)) or 0)
