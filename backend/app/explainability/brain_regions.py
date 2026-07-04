"""Aggregate per-vertex attributions into interpretable brain regions.

TRIBE predicts activity on the fsaverage5 mesh (two hemispheres). Vertex-level
attributions are too fine to interpret, so we aggregate them into regions. The
default atlas is deliberately simple and dependency-free - hemisphere split by
coarse anterior-posterior bands - and is fully pluggable: pass a ``labels``
array (one region id per vertex, e.g. a Schaefer/Glasser parcellation loaded
from an atlas file) to aggregate into anatomically named regions instead.
"""

from __future__ import annotations

import numpy as np

from app.core.constants import FSAVERAGE5_VERTICES_PER_HEMI


def default_region_labels(n_vertices: int, *, bands: int = 4) -> tuple[np.ndarray, list[str]]:
    """Build a simple hemisphere-by-band atlas covering ``n_vertices``.

    Assumes the first half of the vertices are the left hemisphere and the rest
    the right (the fsaverage5 convention). Within each hemisphere, vertices are
    split into ``bands`` contiguous groups as a coarse spatial proxy.
    """
    half = n_vertices // 2 if n_vertices != 2 * FSAVERAGE5_VERTICES_PER_HEMI else (
        FSAVERAGE5_VERTICES_PER_HEMI
    )
    labels = np.empty(n_vertices, dtype=np.int64)
    names: list[str] = []
    region_id = 0
    for hemi, (start, end) in enumerate([(0, half), (half, n_vertices)]):
        hemi_name = "L" if hemi == 0 else "R"
        size = max(end - start, 1)
        edges = np.linspace(start, end, bands + 1).astype(int)
        for b in range(bands):
            labels[edges[b] : edges[b + 1]] = region_id
            names.append(f"{hemi_name}-band{b + 1}")
            region_id += 1
        del size
    return labels, names


def aggregate_region_importance(
    vertex_attribution: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    names: list[str] | None = None,
    bands: int = 4,
) -> dict[str, float]:
    """Aggregate ``(V,)`` (or ``(T, V)``) attributions into region importances.

    Importance is the mean absolute attribution over the vertices of a region,
    normalised to sum to 1 across regions for comparability.
    """
    attr = np.asarray(vertex_attribution)
    # (T, V) -> collapse time by mean-abs; (V,) -> abs.
    attr = np.mean(np.abs(attr), axis=0) if attr.ndim == 2 else np.abs(attr)

    n_vertices = attr.shape[0]
    if labels is None:
        labels, names = default_region_labels(n_vertices, bands=bands)
    assert names is not None

    importances: dict[str, float] = {}
    for region_id, name in enumerate(names):
        members = attr[labels == region_id]
        importances[name] = float(members.mean()) if members.size else 0.0

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}
    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
