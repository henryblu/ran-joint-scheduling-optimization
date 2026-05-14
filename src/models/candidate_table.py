"""Shared scheduler-facing candidate-table and batch-space contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pa import PAParams


# The stored artifact is one active-slot PHY operating row:
# user id is carried outside this table by the batch lookup, and each row stores
# pa_id, n_prb, layers, mcs, bits_per_slot, p_dc_active_w, and p_out_total_w.
# p_out_total_w is the total RF output for this complete user allocation in one
# active slot. p_dc_active_w is the allocation-level active PA DC draw for that
# row, including the row's PRB-share scaling when the row occupies only part of
# the carrier.
BATCH_USER_PARAMETER_SPACE_COLUMNS = [
    "pa_id",
    "n_prb",
    "layers",
    "mcs",
    "bits_per_slot",
    "p_dc_active_w",
    "p_out_total_w",
]


@dataclass(frozen=True)
class BatchUserParameterSpace:
    """Trusted batch single-user artifact consumed by multi-user schedulers."""

    user_requirements: pd.DataFrame
    user_parameter_spaces: dict[int, pd.DataFrame]
    frame_n_slots: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]


__all__ = [
    "BATCH_USER_PARAMETER_SPACE_COLUMNS",
    "BatchUserParameterSpace",
]
