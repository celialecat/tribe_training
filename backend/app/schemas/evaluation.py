"""Evaluation API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EvaluationRunRequest(BaseModel):
    pass


class EvaluationReport(BaseModel):
    report: dict[str, Any]
