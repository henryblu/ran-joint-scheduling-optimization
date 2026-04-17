from __future__ import annotations

import math

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG
from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointOfdmaProblem, USER_CANDIDATE_COLUMNS


_ROUNDING_TOL = 1e-12

_DUPLICATE_SORT_COLUMNS = [
    "pa_id",
    "total_prb_slots",
    "schedule_cost",
    "n_prb",
    "mcs",
    "layers",
    "bits_per_slot",
    "p_dc_active_w",
    "p_out_total_w",
]
_DUPLICATE_SORT_ASCENDING = [True] * len(_DUPLICATE_SORT_COLUMNS)

_FINAL_SORT_COLUMNS = [
    "total_prb_slots",
    "schedule_cost",
    "pa_id",
    "n_prb",
    "mcs",
    "layers",
    "bits_per_slot",
    "p_dc_active_w",
    "p_out_total_w",
]
_FINAL_SORT_ASCENDING = [True] * len(_FINAL_SORT_COLUMNS)


def prepare_joint_ofdma_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointOfdmaProblem:
    """Prepare one solver-ready OFDMA problem from a trusted batch artifact.

    Steps:
    1. Read the trusted per-user slot-normalized feasible spaces from the batch artifact.
    2. Resolve the shared one-frame OFDMA PRB-slot budget.
    3. Expand each user's single-slot operating rows into rate-feasible frame allocations.
    4. Validate one-frame OFDMA feasibility under the scalar PRB-slot resource model.
    5. Exact-prune each user space into scheduler-ready allocations.
    6. Apply one joint PRB-slot infeasibility prune and assemble the prepared problem.
    """

    user_slot_spaces = read_trusted_user_slot_spaces(batch_space)
    frame_n_slots = int(batch_space.frame_n_slots)
    prb_max = resolve_prb_max()
    frame_prb_budget = int(frame_n_slots * prb_max)

    expanded_user_spaces = expand_requirement_feasible_user_spaces(
        batch_space,
        user_slot_spaces,
        frame_n_slots=frame_n_slots,
        frame_prb_budget=frame_prb_budget,
    )
    validate_joint_ofdma_feasibility(
        expanded_user_spaces,
        frame_prb_budget=frame_prb_budget,
    )

    user_candidate_spaces = {
        int(user_id): prune_user_ofdma_space(candidate_table)
        for user_id, candidate_table in expanded_user_spaces.items()
    }
    user_candidate_spaces = prune_jointly_infeasible_user_rows(
        user_candidate_spaces,
        frame_prb_budget=frame_prb_budget,
    )

    return PreparedJointOfdmaProblem(
        frame_n_slots=frame_n_slots,
        prb_max=prb_max,
        frame_prb_budget=frame_prb_budget,
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_candidate_spaces=user_candidate_spaces,
    )


def read_trusted_user_slot_spaces(
    batch_space: BatchUserParameterSpace,
) -> dict[int, pd.DataFrame]:
    """Copy the trusted per-user single-slot spaces from the batch artifact."""

    return {
        int(user_row.user_id): (
            batch_space.user_parameter_spaces[int(user_row.user_id)][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }


def expand_requirement_feasible_user_spaces(
    batch_space: BatchUserParameterSpace,
    user_slot_spaces: dict[int, pd.DataFrame],
    *,
    frame_n_slots: int,
    frame_prb_budget: int,
) -> dict[int, pd.DataFrame]:
    """Expand all trusted single-slot rows into rate-feasible frame allocations."""

    expanded_user_spaces = {}
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        user_slot_space = user_slot_spaces[user_id]
        if user_slot_space.empty:
            raise RuntimeError(
                f"No feasible slot-normalized operating points were found for user {user_id}."
            )

        expanded_space = expand_user_ofdma_space(
            user_id=user_id,
            required_rate_bps=float(user_row.required_rate_bps),
            slot_table=user_slot_space,
            frame_n_slots=frame_n_slots,
            frame_prb_budget=frame_prb_budget,
        )
        if expanded_space.empty:
            raise RuntimeError(
                f"User {user_id} cannot meet the required average rate within one OFDMA frame."
            )

        expanded_user_spaces[user_id] = expanded_space

    return expanded_user_spaces


def resolve_prb_max() -> int:
    """Resolve the single-slot PRB budget from the shared trusted radio config."""

    # The current OFDMA preparation path assumes the repository-wide fixed radio
    # geometry and resolves the PRB budget from the shared config.
    config = SINGLE_USER_SEARCH_CONFIG
    return int(math.floor(float(config.channel_bw_hz) / (12.0 * float(config.delta_f_hz))))


def validate_joint_ofdma_feasibility(
    expanded_user_spaces: dict[int, pd.DataFrame],
    *,
    frame_prb_budget: int,
) -> None:
    """Validate that the expanded user allocations fit inside one OFDMA frame budget."""

    total_minimum_prb_slots = sum(
        int(candidate_table["total_prb_slots"].astype(int).min())
        for candidate_table in expanded_user_spaces.values()
    )
    if total_minimum_prb_slots <= int(frame_prb_budget):
        return

    raise RuntimeError(
        "The requested average rates are not schedulable within one OFDMA frame after PRB-slot quantization: "
        f"PRB-slot lower bound = {int(total_minimum_prb_slots)} > {int(frame_prb_budget)}."
    )


def expand_user_ofdma_space(
    *,
    user_id: int,
    required_rate_bps: float,
    slot_table: pd.DataFrame,
    frame_n_slots: int,
    frame_prb_budget: int,
) -> pd.DataFrame:
    """Expand one user's single-slot rows into frame allocations that meet the rate target."""

    if slot_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_rows = []
    for row in slot_table[BATCH_USER_PARAMETER_SPACE_COLUMNS].itertuples(index=False):
        bits_per_slot = float(row.bits_per_slot)
        if bits_per_slot <= 0.0:
            continue

        required_active_share = (
            float(required_rate_bps) * float(SINGLE_USER_SEARCH_CONFIG.t_slot_s) / bits_per_slot
        )
        if required_active_share <= 0.0:
            continue
        if required_active_share > 1.0 + _ROUNDING_TOL:
            continue

        total_prb_slots = int(
            math.ceil(float(frame_n_slots) * int(row.n_prb) * required_active_share - _ROUNDING_TOL)
        )
        if total_prb_slots < 1:
            continue
        if total_prb_slots > int(frame_prb_budget):
            continue

        candidate_rows.append(
            {
                "user_id": int(user_id),
                "pa_id": int(row.pa_id),
                "n_prb": int(row.n_prb),
                "layers": int(row.layers),
                "mcs": int(row.mcs),
                "bits_per_slot": bits_per_slot,
                "p_dc_active_w": float(row.p_dc_active_w),
                "p_out_total_w": float(row.p_out_total_w),
                "total_prb_slots": total_prb_slots,
                # This remains a slot-share approximation until a later OFDMA
                # physics pass resolves aggregate same-slot RF load exactly.
                "schedule_cost": required_active_share * float(row.p_dc_active_w),
            }
        )

    if not candidate_rows:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_table = pd.DataFrame.from_records(candidate_rows, columns=USER_CANDIDATE_COLUMNS)
    return sort_user_candidate_table(candidate_table)


def prune_user_ofdma_space(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Exact-prune one user's expanded OFDMA allocations into a scheduler-ready space."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    deduplicated_table = (
        candidate_table.sort_values(
            _DUPLICATE_SORT_COLUMNS,
            ascending=_DUPLICATE_SORT_ASCENDING,
        )
        .drop_duplicates(subset=["pa_id", "total_prb_slots"], keep="first")
        .reset_index(drop=True)
    )

    kept_rows = []
    for row in sort_user_candidate_table(deduplicated_table).to_dict("records"):
        if any(row_dominates(kept_row, row) for kept_row in kept_rows):
            continue

        kept_rows = [
            kept_row
            for kept_row in kept_rows
            if not row_dominates(row, kept_row)
        ]
        kept_rows.append(row)

    pruned_table = pd.DataFrame.from_records(kept_rows, columns=USER_CANDIDATE_COLUMNS)
    return sort_user_candidate_table(pruned_table)


def prune_jointly_infeasible_user_rows(
    user_candidate_spaces: dict[int, pd.DataFrame],
    *,
    frame_prb_budget: int,
) -> dict[int, pd.DataFrame]:
    """Drop rows that can never fit after reserving the minimum area of all other users."""

    minimum_area_by_user = {
        int(user_id): int(candidate_table["total_prb_slots"].astype(int).min())
        for user_id, candidate_table in user_candidate_spaces.items()
    }
    total_minimum_area = int(sum(minimum_area_by_user.values()))

    pruned_spaces = {}
    for user_id, candidate_table in user_candidate_spaces.items():
        user_id = int(user_id)
        max_jointly_feasible_area = (
            int(frame_prb_budget)
            - int(total_minimum_area)
            + int(minimum_area_by_user[user_id])
        )
        pruned_table = sort_user_candidate_table(
            candidate_table.loc[
                candidate_table["total_prb_slots"].astype(int).le(int(max_jointly_feasible_area))
            ].copy()
        )
        if pruned_table.empty:
            raise RuntimeError(
                "The prepared OFDMA rows became jointly infeasible after PRB-slot-budget pruning."
            )
        pruned_spaces[user_id] = pruned_table

    return pruned_spaces


def row_dominates(left_row: dict, right_row: dict) -> bool:
    """Return whether one OFDMA candidate row dominates another exactly."""

    left_area = int(left_row["total_prb_slots"])
    right_area = int(right_row["total_prb_slots"])
    left_cost = float(left_row["schedule_cost"])
    right_cost = float(right_row["schedule_cost"])

    area_ok = left_area <= right_area
    cost_ok = left_cost <= right_cost
    if not (area_ok and cost_ok):
        return False

    return (
        left_area < right_area
        or left_cost < right_cost
    )


def sort_user_candidate_table(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic ordering for OFDMA user candidate tables."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    return (
        candidate_table[USER_CANDIDATE_COLUMNS]
        .sort_values(
            _FINAL_SORT_COLUMNS,
            ascending=_FINAL_SORT_ASCENDING,
        )
        .reset_index(drop=True)
    )


__all__ = [
    "prepare_joint_ofdma_problem",
]
