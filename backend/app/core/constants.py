"""Pipeline-wide constants.

These are structural facts about the TRIBE v2 output space and the prediction
target space. They are intentionally centralised so that model definitions, the
database schema and the frontend all agree on dimensionality and target order.
"""

from __future__ import annotations

from enum import StrEnum

# ---- TRIBE v2 output geometry -------------------------------------------------
# TRIBE predicts fMRI responses on the fsaverage5 cortical surface. fsaverage5
# has 10242 vertices per hemisphere; the medial wall is typically masked out,
# leaving ~20k cortical vertices across both hemispheres. The exact vertex count
# is read from the model at runtime; this is the nominal full-mesh size used for
# validation and documentation.
FSAVERAGE5_VERTICES_PER_HEMI = 10_242
FSAVERAGE5_VERTICES = 2 * FSAVERAGE5_VERTICES_PER_HEMI  # 20484 nominal

# TRIBE emits one brain-activity frame per fMRI repetition time (TR).
DEFAULT_TR_SECONDS = 1.49  # CNeuroMod TR used by the Algonauts 2025 data.

# ---- Brain encoder latent size -----------------------------------------------
BRAIN_LATENT_DIM = 512


# ---- Prediction targets -------------------------------------------------------
class Target(StrEnum):
    """The multitask regression targets, in a fixed canonical order.

    Order matters: model output columns, loss weighting and the DB/JSON schema
    all index targets by this enumeration.
    """

    log_views_7d = "log_views_7d"       # log10(1 + views after 7 days)
    log_views_30d = "log_views_30d"     # log10(1 + views after 30 days)
    engagement = "engagement"           # (likes + comments) / views, in [0, 1]
    retention = "retention"             # mean fractional watch time, in [0, 1]
    virality = "virality"               # views growth ratio 30d/7d, normalised
    log_likes = "log_likes"             # log10(1 + likes) — the primary target


# Canonical ordering used everywhere a target vector is built.
TARGET_ORDER: tuple[Target, ...] = (
    Target.log_views_7d,
    Target.log_views_30d,
    Target.engagement,
    Target.retention,
    Target.virality,
    Target.log_likes,
)
NUM_TARGETS = len(TARGET_ORDER)

# The primary supervision signal for the success predictor.
PRIMARY_TARGET: Target = Target.log_likes

# Targets that live in [0, 1] and are therefore modelled through a sigmoid and
# scored with a bounded loss; the remainder are unbounded log-space regressions.
BOUNDED_TARGETS: frozenset[Target] = frozenset(
    {Target.engagement, Target.retention, Target.virality}
)
