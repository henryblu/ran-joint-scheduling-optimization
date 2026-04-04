"""Shared scheduler-facing candidate-table and batch-space contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pa import PAParams


# `rate_active_bps` is the achieved throughput when the operating point stays
# active for the full modeled frame before TDMA slot-share scaling is applied.
BATCH_USER_PARAMETER_SPACE_COLUMNS = [
    "pa_id",
    "n_prb",
    "layers",
    "mcs",
    "rate_active_bps",
    "p_dc_active_w",
    "p_out_total_w",
]


@dataclass(frozen=True)
class BatchUserParameterSpace:
    """Trusted batch single-user artifact consumed by the TDMA scheduler."""

    user_requirements: pd.DataFrame
    user_parameter_spaces: dict[int, pd.DataFrame]
    frame_n_slots: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]


__all__ = [
    "BATCH_USER_PARAMETER_SPACE_COLUMNS",
    "BatchUserParameterSpace",
]
