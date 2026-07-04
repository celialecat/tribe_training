"""Target-label computation.

Turns observed video statistics into the multitask regression targets, together
with an observation *mask* (1 = supervised, 0 = missing). Masking is essential
for scientific correctness: some targets are simply not observable from a single
public snapshot, and the training loss must ignore them rather than fit noise.

Observability by target
------------------------
- ``engagement``          : directly observable = (likes + comments) / views.
- ``log_views_7d/30d``    : require the view count *at* 7 and 30 days post-upload.
                            Only exact when longitudinal snapshots exist. For a
                            single snapshot we OPTIONALLY approximate them with a
                            saturating growth model (see ``approximate=True``),
                            clearly flagged; otherwise they are masked out.
- ``retention``           : average watch time is private (YouTube Studio only).
                            Never inferred; supplied externally or masked.
- ``virality``            : ratio of 30d to 7d views; observable only when both
                            view figures are (approximately) known.

The saturating model V(t) = V_now * (1 - exp(-t/tau)) / (1 - exp(-age/tau))
assumes cumulative views follow a first-order saturating curve with time-
constant tau. It
is a bootstrap heuristic for cold-starting a model, NOT a substitute for real
longitudinal data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from app.core.constants import NUM_TARGETS, TARGET_ORDER, Target
from app.db.models.video import Video

# Time-constant (days) of the saturating view-accumulation heuristic.
_VIEW_GROWTH_TAU_DAYS = 20.0


@dataclass(slots=True)
class TargetSample:
    """A target vector and its observation mask, in canonical target order."""

    values: np.ndarray  # shape (NUM_TARGETS,), float32
    mask: np.ndarray    # shape (NUM_TARGETS,), float32 in {0, 1}

    def as_dict(self) -> dict[str, float | None]:
        return {
            t.value: (float(v) if m else None)
            for t, v, m in zip(TARGET_ORDER, self.values, self.mask, strict=True)
        }


def _video_age_days(video: Video, *, now: datetime | None = None) -> float | None:
    if video.upload_date is None:
        return None
    now = now or datetime.now(UTC)
    upload = video.upload_date
    if upload.tzinfo is None:
        upload = upload.replace(tzinfo=UTC)
    return max((now - upload).total_seconds() / 86_400.0, 0.0)


def _saturating_views(views_now: float, age_days: float, t_days: float) -> float:
    """Estimate cumulative views at ``t_days`` given current views at ``age_days``."""
    if age_days <= 0:
        return views_now
    tau = _VIEW_GROWTH_TAU_DAYS
    denom = 1.0 - math.exp(-age_days / tau)
    if denom <= 1e-6:
        return views_now
    return views_now * (1.0 - math.exp(-t_days / tau)) / denom


def compute_targets(
    video: Video,
    *,
    approximate_view_curve: bool = False,
    retention: float | None = None,
    now: datetime | None = None,
) -> TargetSample:
    """Compute the target vector and mask for a video.

    Parameters
    ----------
    approximate_view_curve:
        If True, fill 7d/30d/virality via the saturating heuristic when only a
        single snapshot is available. If False (default), those targets are
        masked unless exactly observable.
    retention:
        Externally-supplied mean fractional watch time in [0, 1], if known.
    """
    values = np.zeros(NUM_TARGETS, dtype=np.float32)
    mask = np.zeros(NUM_TARGETS, dtype=np.float32)
    index = {t: i for i, t in enumerate(TARGET_ORDER)}

    views = video.view_count
    age = _video_age_days(video, now=now)

    # ---- engagement (directly observable) ----
    if views and views > 0:
        interactions = (video.like_count or 0) + (video.comment_count or 0)
        values[index[Target.engagement]] = float(np.clip(interactions / views, 0.0, 1.0))
        mask[index[Target.engagement]] = 1.0

    # ---- view targets ----
    if views and views > 0 and age is not None:
        if age >= 30.0:
            # Enough history that "current" ~ 30d cumulative; 7d needs the curve.
            v30 = float(views)
            values[index[Target.log_views_30d]] = math.log10(1.0 + v30)
            mask[index[Target.log_views_30d]] = 1.0
            if approximate_view_curve:
                v7 = _saturating_views(views, age, 7.0)
                values[index[Target.log_views_7d]] = math.log10(1.0 + v7)
                mask[index[Target.log_views_7d]] = 1.0
                _set_virality(values, mask, index, v7, v30)
        elif approximate_view_curve:
            v7 = _saturating_views(views, age, 7.0)
            v30 = _saturating_views(views, age, 30.0)
            values[index[Target.log_views_7d]] = math.log10(1.0 + v7)
            values[index[Target.log_views_30d]] = math.log10(1.0 + v30)
            mask[index[Target.log_views_7d]] = 1.0
            mask[index[Target.log_views_30d]] = 1.0
            _set_virality(values, mask, index, v7, v30)

    # ---- retention (external only) ----
    if retention is not None:
        values[index[Target.retention]] = float(np.clip(retention, 0.0, 1.0))
        mask[index[Target.retention]] = 1.0

    return TargetSample(values=values, mask=mask)


def _set_virality(values, mask, index, v7: float, v30: float) -> None:  # type: ignore[no-untyped-def]
    """Virality = normalised 30d/7d growth ratio, squashed to [0, 1]."""
    if v7 > 0:
        ratio = v30 / v7                     # in [1, ~4] for saturating growth
        values[index[Target.virality]] = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
        mask[index[Target.virality]] = 1.0
