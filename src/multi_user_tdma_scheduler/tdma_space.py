from __future__ import annotations

import numpy as np
import pandas as pd

from configs import MULTI_USER_TDMA_CONFIG
from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointScheduleProblem, USER_CANDIDATE_COLUMNS


TOL = 1e-12


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare the exact TDMA problem from one trusted batch artifact."""

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
    user_candidate_spaces = {}
    infeasible_reason = None
    exact_slot_lower_bound = 0

    for user_row in user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        raw_user_space = (
            batch_space.user_parameter_spaces[user_id][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        if raw_user_space.empty and infeasible_reason is None:
            infeasible_reason = f"No feasible slot-normalized operating points were found for user {user_id}."

        user_candidate_space = quantize_and_prune_user_tdma_space(
            user_id=user_id,
            required_rate_bps=float(user_row.required_rate_bps),
            active_table=raw_user_space,
            frame_n_slots=frame_n_slots,
        )
        user_candidate_spaces[user_id] = user_candidate_space
        if user_candidate_space.empty:
            if infeasible_reason is None:
                infeasible_reason = "The requested average rates are not schedulable within one frame after exact slot quantization."
            continue

        exact_slot_lower_bound += int(user_candidate_space["n_slots"].astype(int).min())

    if infeasible_reason is None and exact_slot_lower_bound > frame_n_slots:
        infeasible_reason = "The requested average rates are not schedulable within one frame after exact slot quantization."

    if infeasible_reason is None:
        user_candidate_spaces, infeasible_reason = prune_jointly_infeasible_user_rows(
            user_candidate_spaces,
            frame_n_slots=frame_n_slots,
        )

    return PreparedJointScheduleProblem(
        frame_n_slots=frame_n_slots,
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_requirements=user_requirements,
        user_candidate_spaces=user_candidate_spaces,
        infeasible_reason=infeasible_reason,
    )


def quantize_and_prune_user_tdma_space(
    *,
    user_id: int,
    required_rate_bps: float,
    active_table: pd.DataFrame,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Attach exact one-frame slot counts and keep one row per PA family and slot count."""

    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    bits_per_slot = active_table["bits_per_slot"].astype(float).to_numpy()
    required_slots = np.array(
        [
            _compute_required_slots(
                required_rate_bps=float(required_rate_bps),
                bits_per_slot=float(bits_value),
                frame_n_slots=frame_n_slots,
            )
            for bits_value in bits_per_slot
        ],
        dtype=int,
    )
    feasible_mask = (bits_per_slot > 0.0) & (required_slots >= 1) & (required_slots <= int(frame_n_slots))
    if not np.any(feasible_mask):
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_table = (
        active_table.loc[feasible_mask, BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .copy()
        .assign(
            user_id=int(user_id),
            n_slots=required_slots[feasible_mask],
        )[USER_CANDIDATE_COLUMNS]
        .reset_index(drop=True)
    )
    return exact_prune_user_tdma_space(
        candidate_table,
        frame_n_slots=frame_n_slots,
    )


def exact_prune_user_tdma_space(
    candidate_table: pd.DataFrame,
    *,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Keep one minimum-cost TDMA row for each quantized slot count per PA family."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    return (
        candidate_table.assign(
            schedule_cost=lambda table: (
                table["n_slots"].astype(int)
                * table["p_dc_active_w"].astype(float)
                / float(frame_n_slots)
            )
        )
        .sort_values(
            ["pa_id", "n_slots", "schedule_cost", "n_prb", "mcs", "layers"],
            ascending=[True, True, True, True, True, True],
        )
        .drop_duplicates(subset=["pa_id", "n_slots"], keep="first")
        [USER_CANDIDATE_COLUMNS]
        .reset_index(drop=True)
    )


def prune_dominated_user_tdma_space(
    candidate_table: pd.DataFrame,
    *,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Drop rows that are never better in cost, slots, or delivered rate."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    frame_duration_s = float(frame_n_slots) * float(MULTI_USER_TDMA_CONFIG.t_slot_s)
    ranked_rows = (
        candidate_table.assign(
            schedule_cost=lambda table: (
                table["n_slots"].astype(int)
                * table["p_dc_active_w"].astype(float)
                / float(frame_n_slots)
            ),
            delivered_rate_bps=lambda table: (
                table["n_slots"].astype(int)
                * table["bits_per_slot"].astype(float)
                / float(frame_duration_s)
            ),
        )
        .sort_values(
            ["schedule_cost", "n_slots", "delivered_rate_bps", "pa_id", "n_prb", "mcs", "layers"],
            ascending=[True, True, False, True, True, True, True],
        )
        .to_dict("records")
    )

    kept_rows = []
    for row in ranked_rows:
        if any(
            float(kept_row["schedule_cost"]) <= float(row["schedule_cost"]) + TOL
            and int(kept_row["n_slots"]) <= int(row["n_slots"])
            and float(kept_row["delivered_rate_bps"]) >= float(row["delivered_rate_bps"]) - TOL
            for kept_row in kept_rows
        ):
            continue
        kept_rows.append(row)

    return pd.DataFrame(kept_rows, columns=[*USER_CANDIDATE_COLUMNS, "schedule_cost", "delivered_rate_bps"])[
        USER_CANDIDATE_COLUMNS
    ].reset_index(drop=True)


def prune_jointly_infeasible_user_rows(
    user_candidate_spaces: dict[int, pd.DataFrame],
    *,
    frame_n_slots: int,
) -> tuple[dict[int, pd.DataFrame], str | None]:
    """Drop rows that can never fit once every other user receives its minimum slots."""

    minimum_slots_by_user = {
        int(user_id): int(candidate_table["n_slots"].astype(int).min())
        for user_id, candidate_table in user_candidate_spaces.items()
    }
    total_minimum_slots = int(sum(minimum_slots_by_user.values()))
    pruned_spaces = {}

    for user_id, candidate_table in user_candidate_spaces.items():
        max_jointly_feasible_slots = int(frame_n_slots) - int(total_minimum_slots) + int(minimum_slots_by_user[int(user_id)])
        pruned_table = (
            candidate_table.loc[candidate_table["n_slots"].astype(int) <= int(max_jointly_feasible_slots)]
            .copy()
            .reset_index(drop=True)
        )
        pruned_spaces[int(user_id)] = pruned_table
        if not pruned_table.empty:
            continue
        return pruned_spaces, "The prepared TDMA rows became jointly infeasible after slot-budget pruning."

    return pruned_spaces, None


def _compute_required_slots(*, required_rate_bps: float, bits_per_slot: float, frame_n_slots: int) -> int:
    """Return the exact one-frame slot count implied by a slot-normalized row."""

    if bits_per_slot <= 0.0:
        return int(frame_n_slots) + 1

    return int(
        np.ceil(
            float(required_rate_bps)
            * float(frame_n_slots)
            * float(MULTI_USER_TDMA_CONFIG.t_slot_s)
            / float(bits_per_slot)
            - TOL
        )
    )


__all__ = [
    "prepare_joint_schedule_problem",
]
