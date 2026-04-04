from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DISTANCE_BIN_GRID_M = tuple(range(25, 501, 25))


@dataclass(frozen=True)
class DistanceBinnedCandidateTable:
    """Precomputed candidate frontiers keyed by fixed distance bins."""

    frontiers_by_distance_m: dict[int, pd.DataFrame]


__all__ = [
    "DISTANCE_BIN_GRID_M",
    "DistanceBinnedCandidateTable",
]
