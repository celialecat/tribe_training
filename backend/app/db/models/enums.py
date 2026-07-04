"""Enumerations used as string columns across ORM models."""

from __future__ import annotations

from enum import StrEnum


class VideoSource(StrEnum):
    youtube = "youtube"
    local = "local"


class VideoStatus(StrEnum):
    """Lifecycle of a video as it moves through the ingestion pipeline."""

    pending = "pending"
    downloading = "downloading"
    downloaded = "downloaded"
    extracting = "extracting"
    ready = "ready"          # media + metadata present, ready for TRIBE
    failed = "failed"


class BrainStatus(StrEnum):
    """State of the TRIBE brain-activity artefact for a video."""

    absent = "absent"
    computing = "computing"
    cached = "cached"
    failed = "failed"


class ExperimentStatus(StrEnum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class EmbeddingKind(StrEnum):
    brain_activity = "brain_activity"   # raw TRIBE output (T x V), on disk
    brain_latent = "brain_latent"       # 512-d encoder output
