from __future__ import annotations

"""Input preparation for the OFDMA rolling-quantum round-robin baseline."""

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG
from models import BatchUserParameterSpace, PASwitchPolicy
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import RoundRobinCandidateRow, RoundRobinProblem


TOL = 1e-12


def prepare_round_robin_problem(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy,
) -> RoundRobinProblem:
    """Flatten one trusted batch artifact into deterministic round-robin rows."""

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
    candidate_rows = build_round_robin_candidate_rows(batch_space, user_requirements)
    return RoundRobinProblem(
        frame_n_slots=frame_n_slots,
        t_slot_s=t_slot_s,
        delta_f_hz=float(SINGLE_USER_SEARCH_CONFIG.delta_f_hz),
        prb_max=compute_prb_budget(),
        n_tx_chains=int(batch_space.n_tx_chains),
        use_psd_constraint=bool(SINGLE_USER_SEARCH_CONFIG.use_psd_constraint),
        psd_max_w_per_hz=float(SINGLE_USER_SEARCH_CONFIG.psd_max_w_per_hz),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_requirements=user_requirements,
        candidate_rows=candidate_rows,
        candidate_rows_by_user=group_candidate_rows_by_user(candidate_rows, user_requirements),
        required_rate_by_user=required_rate_by_user,
        demand_bits_by_user=demand_bits_by_user,
        switch_policy=switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy)),
    )


def build_round_robin_candidate_rows(
    batch_space: BatchUserParameterSpace,
    user_requirements: pd.DataFrame,
) -> tuple[RoundRobinCandidateRow, ...]:
    """Build deterministic positive-payload candidate rows for every user."""

    candidate_rows: list[RoundRobinCandidateRow] = []
    global_id = 0
    for user_row in user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
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
        for local_row_id, row in enumerate(user_space.itertuples(index=False)):
            candidate_rows.append(
                RoundRobinCandidateRow(
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


def group_candidate_rows_by_user(
    candidate_rows: tuple[RoundRobinCandidateRow, ...],
    user_requirements: pd.DataFrame,
) -> dict[int, tuple[RoundRobinCandidateRow, ...]]:
    return {
        int(user_row.user_id): tuple(
            row for row in candidate_rows if int(row.user_id) == int(user_row.user_id)
        )
        for user_row in user_requirements.itertuples(index=False)
    }


def candidate_rows_by_user_pa(problem: RoundRobinProblem) -> dict[int, dict[int, int]]:
    """Return candidate-row counts by user and PA id for solver details."""

    return {
        int(user_id): {
            int(pa_id): sum(1 for row in rows if int(row.pa_id) == int(pa_id))
            for pa_id in range(len(problem.pa_catalog))
        }
        for user_id, rows in sorted(problem.candidate_rows_by_user.items())
    }


def high_power_pa_ids(problem: RoundRobinProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="8W PA", use_highest_power=True)


def low_power_pa_ids(problem: RoundRobinProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="4W PA", use_highest_power=False)


def pa_ids_by_label_or_power(
    problem: RoundRobinProblem,
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


__all__ = [
    "candidate_rows_by_user_pa",
    "high_power_pa_ids",
    "low_power_pa_ids",
    "prepare_round_robin_problem",
]


def compute_prb_budget() -> int:
    return int(float(SINGLE_USER_SEARCH_CONFIG.channel_bw_hz) // (12.0 * float(SINGLE_USER_SEARCH_CONFIG.delta_f_hz)))
