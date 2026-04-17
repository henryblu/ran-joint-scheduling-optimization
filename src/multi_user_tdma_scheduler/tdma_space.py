from __future__ import annotations

import numpy as np
import pandas as pd

from configs import MULTI_USER_TDMA_CONFIG
from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointScheduleProblem, USER_CANDIDATE_COLUMNS


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare the exact scheduler problem from a trusted batch parameter-space artifact.

    Steps:
    1. Read the trusted per-user slot-normalized feasible spaces from the batch artifact.
    2. Check whether those rows are jointly schedulable within one shared frame.
    3. Quantize each user space onto the exact TDMA slot lattice for that frame.
    4. Exact-prune dominated per-user rows and assemble the prepared problem.
    """

    user_spaces = {
        int(user_row.user_id): (
            batch_space.user_parameter_spaces[int(user_row.user_id)][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    frame_n_slots = validate_single_frame_schedule_feasibility(
        batch_space,
        user_spaces,
    )
    user_candidate_spaces = {
        int(user_row.user_id): quantize_and_prune_user_tdma_space(
            user_id=int(user_row.user_id),
            required_rate_bps=float(user_row.required_rate_bps),
            active_table=user_spaces[int(user_row.user_id)],
            frame_n_slots=frame_n_slots,
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    user_candidate_spaces = prune_jointly_infeasible_user_rows(
        user_candidate_spaces,
        frame_n_slots=frame_n_slots,
    )
    return PreparedJointScheduleProblem(
        frame_n_slots=int(frame_n_slots),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_candidate_spaces=user_candidate_spaces,
    )


def validate_single_frame_schedule_feasibility(
    batch_space: BatchUserParameterSpace,
    user_spaces,
):
    """Validate that the batch can be scheduled within one shared frame."""

    exact_frame_share_sum = 0.0
    t_slot_s = float(MULTI_USER_TDMA_CONFIG.t_slot_s)
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_space = user_spaces[int(user_row.user_id)]
        if user_space.empty:
            raise RuntimeError(
                f"No feasible slot-normalized operating points were found for user {int(user_row.user_id)}."
            )

        max_bits_per_slot = float(user_space["bits_per_slot"].max())
        max_active_rate_bps = max_bits_per_slot / t_slot_s
        required_rate_bps = float(user_row.required_rate_bps)
        if required_rate_bps > max_active_rate_bps:
            raise RuntimeError(
                f"User {int(user_row.user_id)} requires a higher average rate than any slot-normalized operating point can deliver within one frame."
            )
        exact_frame_share_sum += required_rate_bps * t_slot_s / max_bits_per_slot

    if exact_frame_share_sum > 1.0 + 1e-12:
        raise RuntimeError(
            "The requested average rates are infeasible within one shared frame: "
            f"exact frame-share lower bound = {float(exact_frame_share_sum):.3f} > 1.0."
        )

    frame_n_slots = int(batch_space.frame_n_slots)
    if slot_lower_bound(batch_space, user_spaces, frame_n_slots) > frame_n_slots:
        raise RuntimeError(
            "The requested average rates are not schedulable within one frame after exact slot quantization."
        )
    return int(frame_n_slots)


def slot_lower_bound(
    batch_space: BatchUserParameterSpace,
    user_spaces,
    frame_n_slots,
):
    """Return the exact one-frame TDMA slot lower bound implied by the fastest user row."""

    required_slot_count = 0
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_space = user_spaces[int(user_row.user_id)]
        required_slot_count += _compute_required_slots(
            required_rate_bps=float(user_row.required_rate_bps),
            bits_per_slot=float(user_space["bits_per_slot"].max()),
            frame_n_slots=frame_n_slots,
        )
    return int(required_slot_count)


def quantize_and_prune_user_tdma_space(
    *,
    user_id: int,
    required_rate_bps: float,
    active_table: pd.DataFrame,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Quantize one user's slot-normalized rows onto the shared single-frame TDMA lattice."""

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
    feasible_mask = (
        (bits_per_slot > 0.0)
        & (required_slots >= 1)
        & (required_slots <= int(frame_n_slots))
    )
    if not np.any(feasible_mask):
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_table = active_table.loc[feasible_mask, BATCH_USER_PARAMETER_SPACE_COLUMNS].copy().reset_index(drop=True)
    candidate_table["user_id"] = int(user_id)
    candidate_table["n_slots"] = required_slots[feasible_mask]
    return exact_prune_user_tdma_space(
        candidate_table[USER_CANDIDATE_COLUMNS].copy(),
        frame_n_slots=frame_n_slots,
    )


def exact_prune_user_tdma_space(
    candidate_table: pd.DataFrame,
    *,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Keep one minimum-power TDMA row for each quantized slot count per PA family."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    ranked_rows = candidate_table.sort_values(
        ["pa_id", "n_slots", "n_prb", "mcs", "layers"],
        ascending=[True, True, True, True, True],
    ).to_dict("records")
    ranked_rows.sort(
        key=lambda row: (
            int(row["pa_id"]),
            int(row["n_slots"]),
            _row_schedule_power(row, frame_n_slots=frame_n_slots),
            int(row["n_prb"]),
            int(row["mcs"]),
            int(row["layers"]),
        )
    )

    kept_rows = []
    kept_slot_counts_by_pa: dict[int, set[int]] = {}
    for row in ranked_rows:
        pa_id = int(row["pa_id"])
        slot_count = int(row["n_slots"])
        kept_slot_counts = kept_slot_counts_by_pa.setdefault(pa_id, set())
        if slot_count in kept_slot_counts:
            continue

        kept_slot_counts.add(slot_count)
        kept_rows.append(row)

    return pd.DataFrame(kept_rows, columns=USER_CANDIDATE_COLUMNS).reset_index(drop=True)


def prune_jointly_infeasible_user_rows(
    user_candidate_spaces: dict[int, pd.DataFrame],
    *,
    frame_n_slots: int,
) -> dict[int, pd.DataFrame]:
    """Drop per-user rows that can never fit once the other users get their minimum slots.

    Steps:
    1. Read the minimum feasible slot count of each user after local per-user pruning.
    2. Reserve that minimum for every other user.
    3. Keep only the rows that can still fit inside the remaining slot budget.
    """

    minimum_slots_by_user = {
        int(user_id): int(candidate_table["n_slots"].astype(int).min())
        for user_id, candidate_table in user_candidate_spaces.items()
    }
    total_minimum_slots = int(sum(minimum_slots_by_user.values()))

    pruned_spaces = {}
    for user_id, candidate_table in user_candidate_spaces.items():
        user_id = int(user_id)
        max_jointly_feasible_slots = (
            int(frame_n_slots)
            - int(total_minimum_slots)
            + int(minimum_slots_by_user[user_id])
        )
        pruned_table = (
            candidate_table.loc[
                candidate_table["n_slots"].astype(int).le(int(max_jointly_feasible_slots))
            ]
            .copy()
            .reset_index(drop=True)
        )
        if pruned_table.empty:
            raise RuntimeError(
                "The prepared TDMA rows became jointly infeasible after slot-budget pruning."
            )
        pruned_spaces[user_id] = pruned_table

    return pruned_spaces


def _row_schedule_power(row, *, frame_n_slots: int) -> float:
    """Return one user's frame-averaged DC power for the selected slot count."""

    return float(int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(frame_n_slots))


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
            - 1e-12
        )
    )


__all__ = [
    "prepare_joint_schedule_problem",
]
