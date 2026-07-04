"""Dataset validation report.

Summarises the state of the ingested corpus straight from the database: how many
videos are valid and trainable, how many were rejected (and why), duplicates,
missing subtitles/metadata, failed downloads, and TRIBE-cache coverage. This is
the artefact to inspect before launching a training run — it answers "is my data
actually good?" without trusting the pipeline blindly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.enums import BrainStatus, VideoStatus
from app.db.models.video import Video

# Statuses that mean ingestion is still in flight (neither done nor failed).
_IN_FLIGHT = {
    VideoStatus.pending, VideoStatus.downloading, VideoStatus.downloaded,
    VideoStatus.extracting, VideoStatus.validating,
}
# Validation checks whose failure indicates genuine file corruption.
_CORRUPTION_CHECKS = {"ffprobe", "frames_decodable", "video_stream", "frame_count"}


@dataclass(slots=True)
class DatasetReport:
    """Aggregate corpus statistics."""

    total_videos: int = 0
    valid_videos: int = 0            # status == ready (downloaded + validated)
    rejected_videos: int = 0         # status == rejected (failed validation)
    failed_downloads: int = 0        # status == failed (no usable file)
    in_flight: int = 0               # still being ingested
    duplicate_video_ids: int = 0     # should be 0 — integrity check on dedup
    corrupted_files: int = 0         # rejected due to corruption-class checks
    missing_subtitles: int = 0       # ready videos with neither transcript nor subs
    missing_metadata: int = 0        # ready videos missing key labels/features
    with_brain_activity: int = 0     # TRIBE output cached (ready for training)
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            "Dataset Validation Report",
            "=========================",
            f"  total videos          : {self.total_videos}",
            f"  valid (ready)         : {self.valid_videos}",
            f"  rejected (invalid)    : {self.rejected_videos}",
            f"    of which corrupted  : {self.corrupted_files}",
            f"  failed downloads      : {self.failed_downloads}",
            f"  in-flight             : {self.in_flight}",
            f"  duplicate video ids   : {self.duplicate_video_ids}",
            f"  missing subtitles     : {self.missing_subtitles}",
            f"  missing metadata      : {self.missing_metadata}",
            f"  with TRIBE activity   : {self.with_brain_activity}",
        ]
        if self.rejection_reasons:
            lines.append("  rejection reasons:")
            for check, n in sorted(self.rejection_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"      - {check}: {n}")
        return "\n".join(lines)


def build_dataset_report(session: Session) -> DatasetReport:
    """Compute a :class:`DatasetReport` from the current database state."""
    report = DatasetReport()

    def count(*conditions) -> int:
        stmt = select(func.count()).select_from(Video)
        for cond in conditions:
            stmt = stmt.where(cond)
        return int(session.scalar(stmt) or 0)

    report.total_videos = count()
    report.valid_videos = count(Video.status == VideoStatus.ready)
    report.rejected_videos = count(Video.status == VideoStatus.rejected)
    report.failed_downloads = count(Video.status == VideoStatus.failed)
    report.in_flight = count(Video.status.in_(list(_IN_FLIGHT)))
    report.with_brain_activity = count(Video.brain_status == BrainStatus.cached)

    # Duplicate integrity check: youtube_ids appearing more than once.
    dup_stmt = (
        select(func.count())
        .select_from(
            select(Video.youtube_id)
            .where(Video.youtube_id.is_not(None))
            .group_by(Video.youtube_id)
            .having(func.count() > 1)
            .subquery()
        )
    )
    report.duplicate_video_ids = int(session.scalar(dup_stmt) or 0)

    # Missing subtitles / metadata among *valid* videos only.
    report.missing_subtitles = count(
        Video.status == VideoStatus.ready,
        Video.transcript.is_(None),
        Video.subtitle_path.is_(None),
    )
    report.missing_metadata = count(
        Video.status == VideoStatus.ready,
        (Video.view_count.is_(None)) | (Video.upload_date.is_(None)) | (Video.channel.is_(None)),
    )

    # Detailed rejection reasons + corruption count from stored validation JSON.
    rejected = session.scalars(
        select(Video).where(Video.status == VideoStatus.rejected)
    ).all()
    reasons: dict[str, int] = {}
    corrupted = 0
    for video in rejected:
        failed = (video.validation or {}).get("failed", []) if video.validation else []
        if any(check in _CORRUPTION_CHECKS for check in failed):
            corrupted += 1
        for check in failed:
            reasons[check] = reasons.get(check, 0) + 1
    report.corrupted_files = corrupted
    report.rejection_reasons = reasons
    return report
