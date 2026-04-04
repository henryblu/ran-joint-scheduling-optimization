from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from models import PAParams


USER_CANDIDATE_COLUMNS = [
    "user_id",
    "pa_id",
    "n_prb",
    "layers",
    "mcs",
    "n_slots",
    "rate_avg_frame_bps",
    "p_dc_avg_frame_w",
    "p_out_avg_frame_w",
]


@dataclass(frozen=True)
class PreparedJointScheduleProblem:
    """Prepared TDMA scheduling problem passed from space prep to exact joint search."""

    window_n_frames: int
    window_n_slots: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]
    user_candidate_spaces: dict[int, pd.DataFrame]


@dataclass
class MultiUserTdmaSchedulerResult:
    """Optimal TDMA schedule and minimal exact-search diagnostics."""

    best_schedule: dict[str, Any] | None = None
    search_stats: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "MultiUserTdmaSchedulerResult",
    "PreparedJointScheduleProblem",
    "USER_CANDIDATE_COLUMNS",
]
