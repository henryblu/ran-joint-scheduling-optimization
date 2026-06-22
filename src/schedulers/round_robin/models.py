from __future__ import annotations

"""Internal models for the OFDMA rolling-quantum round-robin baseline."""

from dataclasses import dataclass

import pandas as pd

from models import PAParams, PASwitchPolicy


@dataclass(frozen=True)
class RoundRobinCandidateRow:
    """One trusted candidate-backed row for one UE."""

    global_id: int
    user_id: int
    local_row_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    bits_per_slot: float
    p_out_total_w: float
    p_dc_active_w: float


@dataclass(frozen=True)
class RoundRobinProblem:
    """Prepared rolling-quantum OFDMA baseline problem."""

    frame_n_slots: int
    t_slot_s: float
    delta_f_hz: float
    prb_max: int
    n_tx_chains: int
    use_psd_constraint: bool
    psd_max_w_per_hz: float
    pa_catalog: tuple[PAParams, ...]
    user_requirements: pd.DataFrame
    candidate_rows: tuple[RoundRobinCandidateRow, ...]
    candidate_rows_by_user: dict[int, tuple[RoundRobinCandidateRow, ...]]
    required_rate_by_user: dict[int, float]
    demand_bits_by_user: dict[int, float]
    switch_policy: PASwitchPolicy


@dataclass(frozen=True)
class RoundRobinAttemptResult:
    """One PA-family attempt for the round-robin baseline."""

    attempt_name: str
    allowed_pa_ids: tuple[int, ...]
    success: bool
    fair_prb_share: int
    selected_pa_id: int | None
    selected_rows_by_user: dict[int, RoundRobinCandidateRow]
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...]
    delivered_bits_by_user: dict[int, float]
    unsatisfied_user_ids: tuple[int, ...]
    frame_energy_j: float | None
    active_slot_count: int
    allocation_count: int
    round_robin_cycle_count: int
    failure_reason: str | None


__all__ = [
    "RoundRobinAttemptResult",
    "RoundRobinCandidateRow",
    "RoundRobinProblem",
]
