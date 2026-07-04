"""Shared, dependency-light types for the TRIBE integration.

Kept torch-free so that the extractor, cache and API can be imported without the
deep-learning stack present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BrainActivity:
    """Predicted whole-brain fMRI response for one video.

    Attributes
    ----------
    array:
        Float32 array of shape ``(n_timesteps, n_vertices)`` — the TRIBE v2
        prediction on the fsaverage5 cortical mesh.
    tr_seconds:
        Repetition time: real-world seconds between successive timesteps.
    hemodynamic_offset_seconds:
        Offset already applied by TRIBE (predictions are shifted backward to
        account for haemodynamic delay). Recorded for provenance.
    model_id:
        The TRIBE weights identifier that produced this array.
    """

    array: np.ndarray
    tr_seconds: float
    hemodynamic_offset_seconds: float
    model_id: str

    @property
    def n_timesteps(self) -> int:
        return int(self.array.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.array.shape[1])
