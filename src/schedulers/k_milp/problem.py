from __future__ import annotations

"""Independent OFDMA MILP oracle input preparation."""

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG
from models import BatchUserParameterSpace, PASwitchPolicy
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import MilpCandidateRow, OfdmaMilpProblem


TOL = 1e-12


def prepare_ofdma_milp_problem(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy,
    prune_candidate_rows: bool = True,
) -> OfdmaMilpProblem:
    """Flatten one trusted batch artifact into the direct slot-indexed MILP input."""

    user_requirements = (
        batch_space.user_requirements[["user_id", "required_rate_bps"]]
        .copy()
        .assign(
            user_id=lambda table: table["user_id"].astype(int),
            required_rate_bps=lambda table: table["required_rate_bps"].astype(float),
        )
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    frame_n_slots = int(batch_space.frame_n_slots)
    t_slot_s = float(SINGLE_USER_SEARCH_CONFIG.t_slot_s)
    frame_duration_s = float(frame_n_slots) * float(t_slot_s)
    required_rate_by_user = {
        int(row.user_id): float(row.required_rate_bps)
        for row in user_requirements.itertuples(index=False)
    }
    demand_bits_by_user = {
        int(user_id): float(required_rate_bps) * float(frame_duration_s)
        for user_id, required_rate_bps in required_rate_by_user.items()
    }
    candidate_rows = build_milp_candidate_rows(
        batch_space,
        user_requirements,
        prune_candidate_rows=bool(prune_candidate_rows),
    )
    return OfdmaMilpProblem(
        frame_n_slots=frame_n_slots,
        t_slot_s=t_slot_s,
        prb_max=compute_prb_budget(),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_requirements=user_requirements,
        candidate_rows=candidate_rows,
        candidate_rows_by_user=group_candidate_rows_by_user(candidate_rows, user_requirements),
        required_rate_by_user=required_rate_by_user,
        demand_bits_by_user=demand_bits_by_user,
        switch_policy=switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy)),
    )


def build_milp_candidate_rows(
    batch_space: BatchUserParameterSpace,
    user_requirements: pd.DataFrame,
    *,
    prune_candidate_rows: bool = True,
) -> tuple[MilpCandidateRow, ...]:
    """Build deterministic MILP candidate rows from the batch user spaces."""

    candidate_rows: list[MilpCandidateRow] = []
    global_id = 0
    for user_row in user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        required_bits = (
            float(user_row.required_rate_bps)
            * float(batch_space.frame_n_slots)
            * float(SINGLE_USER_SEARCH_CONFIG.t_slot_s)
        )
        user_space = (
            batch_space.user_parameter_spaces[int(user_id)]
            .reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
            .copy()
            .reset_index(drop=True)
        )
        user_space = user_space.loc[user_space["bits_per_slot"].astype(float) > TOL].copy()
        user_space = user_space.sort_values(
            ["pa_id", "n_prb", "mcs", "layers", "bits_per_slot", "p_out_total_w", "p_dc_active_w"],
            ascending=[True, True, True, True, True, True, True],
        ).reset_index(drop=True)
        if bool(prune_candidate_rows):
            user_space = prune_dominated_milp_candidate_rows(
                user_space,
                required_bits=float(required_bits),
            )
        for local_row_id, row in enumerate(user_space.itertuples(index=False)):
            candidate_rows.append(
                MilpCandidateRow(
                    global_id=int(global_id),
                    user_id=int(user_id),
                    local_row_id=int(local_row_id),
                    pa_id=int(row.pa_id),
                    n_prb=int(row.n_prb),
                    layers=int(row.layers),
                    mcs=int(row.mcs),
                    bits_per_slot=float(row.bits_per_slot),
                    p_out_total_w=float(row.p_out_total_w),
                    p_dc_active_w=float(row.p_dc_active_w),
                )
            )
            global_id += 1
    return tuple(candidate_rows)


def prune_dominated_milp_candidate_rows(
    user_space: pd.DataFrame,
    *,
    required_bits: float,
) -> pd.DataFrame:
    """Keep only same-PA rows that can still be optimal in a packed OFDMA slot."""

    rows = tuple(user_space.itertuples())
    kept_indices = []
    for row in rows:
        dominated = any(
            candidate_row_dominates(other, row, required_bits=float(required_bits))
            for other in rows
            if int(other.Index) != int(row.Index)
        )
        if dominated:
            continue
        kept_indices.append(int(row.Index))
    return user_space.loc[kept_indices].reset_index(drop=True)


def candidate_row_dominates(candidate, row, *, required_bits: float) -> bool:
    if int(candidate.pa_id) != int(row.pa_id):
        return False

    candidate_effective_bits = min(float(candidate.bits_per_slot), float(required_bits))
    row_effective_bits = min(float(row.bits_per_slot), float(required_bits))
    weakly_better = (
        int(candidate.n_prb) <= int(row.n_prb)
        and float(candidate.p_out_total_w) <= float(row.p_out_total_w) + TOL
        and float(candidate_effective_bits) + TOL >= float(row_effective_bits)
    )
    if not weakly_better:
        return False

    return (
        int(candidate.n_prb) < int(row.n_prb)
        or float(candidate.p_out_total_w) + TOL < float(row.p_out_total_w)
        or float(candidate_effective_bits) > float(row_effective_bits) + TOL
    )


def group_candidate_rows_by_user(
    candidate_rows: tuple[MilpCandidateRow, ...],
    user_requirements: pd.DataFrame,
) -> dict[int, tuple[MilpCandidateRow, ...]]:
    return {
        int(user_row.user_id): tuple(
            row for row in candidate_rows if int(row.user_id) == int(user_row.user_id)
        )
        for user_row in user_requirements.itertuples(index=False)
    }


def candidate_rows_by_user_pa(problem: OfdmaMilpProblem) -> dict[int, dict[int, int]]:
    """Return candidate-row counts by user and PA id for logs and solver details."""

    return {
        int(user_id): {
            int(pa_id): sum(1 for row in rows if int(row.pa_id) == int(pa_id))
            for pa_id in range(len(problem.pa_catalog))
        }
        for user_id, rows in sorted(problem.candidate_rows_by_user.items())
    }


def high_power_pa_ids(problem: OfdmaMilpProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="8W PA", use_highest_power=True)


def low_power_pa_ids(problem: OfdmaMilpProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="4W PA", use_highest_power=False)


def pa_ids_by_label_or_power(
    problem: OfdmaMilpProblem,
    *,
    label: str,
    use_highest_power: bool,
) -> tuple[int, ...]:
    labeled_pa_ids = tuple(
        int(pa_id)
        for pa_id, pa in enumerate(problem.pa_catalog)
        if str(pa.scenario_label) == str(label)
    )
    if labeled_pa_ids:
        return labeled_pa_ids
    if not problem.pa_catalog:
        return ()

    selected_power = (
        max(float(pa.p_max_w) for pa in problem.pa_catalog)
        if bool(use_highest_power)
        else min(float(pa.p_max_w) for pa in problem.pa_catalog)
    )
    return tuple(
        int(pa_id)
        for pa_id, pa in enumerate(problem.pa_catalog)
        if abs(float(pa.p_max_w) - float(selected_power)) <= TOL
    )


def compute_prb_budget() -> int:
    return int(
        float(SINGLE_USER_SEARCH_CONFIG.channel_bw_hz)
        // (12.0 * float(SINGLE_USER_SEARCH_CONFIG.delta_f_hz))
    )


__all__ = [
    "candidate_rows_by_user_pa",
    "compute_prb_budget",
    "high_power_pa_ids",
    "low_power_pa_ids",
    "prepare_ofdma_milp_problem",
    "prune_dominated_milp_candidate_rows",
]
