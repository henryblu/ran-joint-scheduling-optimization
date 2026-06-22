from __future__ import annotations

import pandas as pd

from configs import MULTI_USER_TDMA_CONFIG
from models import BatchUserParameterSpace, SchedulerMode
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from schedulers.feasibility_bounds import (
    InfeasibilityCertificate,
    log_feasibility_certificate,
    positive_user_count_certificate,
    row_menu_certificate,
)

from .tdma_models import PreparedJointScheduleProblem, USER_CANDIDATE_COLUMNS


TOL = 1e-12


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare trusted single-slot TDMA rows for the joint mixed-plan search."""

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
    frame_duration_s = float(frame_n_slots) * float(MULTI_USER_TDMA_CONFIG.t_slot_s)
    demand_bits_by_user = {
        int(user_row.user_id): float(user_row.required_rate_bps) * float(frame_duration_s)
        for user_row in user_requirements.itertuples(index=False)
    }
    user_candidate_spaces = {}
    infeasible_certificate = positive_user_count_certificate(
        demand_bits_by_user=demand_bits_by_user,
        frame_n_slots=frame_n_slots,
    )

    for user_row in user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        raw_user_space = (
            batch_space.user_parameter_spaces[user_id][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        user_slot_space = prepare_user_tdma_slot_space(user_id=user_id, active_table=raw_user_space)
        user_candidate_spaces[user_id] = user_slot_space

    if infeasible_certificate is None:
        infeasible_certificate = tdma_row_menu_certificate(
            frame_n_slots=frame_n_slots,
            demand_bits_by_user=demand_bits_by_user,
            user_candidate_spaces=user_candidate_spaces,
        )

    if infeasible_certificate is not None:
        log_feasibility_certificate(
            infeasible_certificate,
            scheduler_mode=SchedulerMode.K_MILP.value,
            policy="any",
            attempt_name="tdma_prepare",
        )

    return PreparedJointScheduleProblem(
        frame_n_slots=frame_n_slots,
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_requirements=user_requirements,
        user_candidate_spaces=user_candidate_spaces,
        infeasible_reason=None if infeasible_certificate is None else infeasible_certificate.reason,
    )


def tdma_row_menu_certificate(
    *,
    frame_n_slots: int,
    demand_bits_by_user: dict[int, float],
    user_candidate_spaces: dict[int, pd.DataFrame],
) -> InfeasibilityCertificate | None:
    bits_by_user = {
        int(user_id): tuple(float(bits) for bits in candidate_table["bits_per_slot"].astype(float).tolist())
        for user_id, candidate_table in sorted(user_candidate_spaces.items())
    }
    return row_menu_certificate(
        demand_bits_by_user=demand_bits_by_user,
        bits_by_user=bits_by_user,
        frame_n_slots=int(frame_n_slots),
        max_users_per_slot=1,
    )


def prepare_user_tdma_slot_space(
    *,
    user_id: int,
    active_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return the positive-payload single-slot rows owned by one TDMA user."""

    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    return (
        active_table.loc[active_table["bits_per_slot"].astype(float) > TOL, BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .copy()
        .assign(user_id=int(user_id))
        [USER_CANDIDATE_COLUMNS]
        .sort_values(["pa_id", "p_dc_active_w", "bits_per_slot", "n_prb", "mcs", "layers"], ascending=[True, True, False, True, True, True])
        .reset_index(drop=True)
    )


__all__ = [
    "prepare_joint_schedule_problem",
    "tdma_row_menu_certificate",
]
