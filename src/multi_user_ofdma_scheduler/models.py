from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models import PAParams


USER_CANDIDATE_COLUMNS = [
    "user_id",
    "pa_id",
    "n_prb",
    "layers",
    "mcs",
    "bits_per_slot",
    "p_dc_active_w",
    "p_out_total_w",
    "total_prb_slots",
    "schedule_cost",
]


@dataclass(frozen=True)
class PreparedJointOfdmaProblem:
    """Prepared OFDMA frame-allocation problem passed from space prep to a future solver."""

    frame_n_slots: int
    prb_max: int
    frame_prb_budget: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]
    user_candidate_spaces: dict[int, pd.DataFrame]


__all__ = [
    "PreparedJointOfdmaProblem",
    "USER_CANDIDATE_COLUMNS",
]
