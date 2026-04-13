from __future__ import annotations

import numpy as np
import pandas as pd

from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointScheduleProblem, USER_CANDIDATE_COLUMNS


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare the exact scheduler problem from a trusted batch parameter-space artifact.

    Steps:
    1. Read the trusted per-user full-frame feasible spaces from the batch artifact.
    2. Check whether those rows are jointly schedulable within one shared frame.
    3. Quantize each user space onto the exact TDMA slot lattice for that frame.
    4. Exact-prune dominated per-user rows and assemble the prepared problem.
    """

    full_frame_user_spaces = {
        int(user_row.user_id): (
            batch_space.user_parameter_spaces[int(user_row.user_id)][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    frame_n_slots = validate_single_frame_schedule_feasibility(
        batch_space,
        full_frame_user_spaces,
    )
    user_candidate_spaces = {
        int(user_row.user_id): quantize_and_prune_user_tdma_space(
            user_id=int(user_row.user_id),
            required_rate_bps=float(user_row.required_rate_bps),
            active_table=full_frame_user_spaces[int(user_row.user_id)],
            frame_n_slots=frame_n_slots,
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    return PreparedJointScheduleProblem(
        frame_n_slots=int(frame_n_slots),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_candidate_spaces=user_candidate_spaces,
    )


def validate_single_frame_schedule_feasibility(
    batch_space: BatchUserParameterSpace,
    full_frame_user_spaces,
):
    """Validate that the batch can be scheduled within one shared frame."""

    exact_frame_share_sum = 0.0
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_space = full_frame_user_spaces[int(user_row.user_id)]
        if user_space.empty:
            raise RuntimeError(
                f"No feasible full-frame active operating points were found for user {int(user_row.user_id)}."
            )

        max_active_rate_bps = float(user_space["rate_active_bps"].max())
        required_rate_bps = float(user_row.required_rate_bps)
        if required_rate_bps > max_active_rate_bps:
            raise RuntimeError(
                f"User {int(user_row.user_id)} requires a higher average rate than any full-frame active operating point can deliver."
            )
        exact_frame_share_sum += required_rate_bps / max_active_rate_bps

    if exact_frame_share_sum > 1.0 + 1e-12:
        raise RuntimeError(
            "The requested average rates are infeasible within one shared frame: "
            f"exact frame-share lower bound = {float(exact_frame_share_sum):.3f} > 1.0."
        )

    frame_n_slots = int(batch_space.frame_n_slots)
    if slot_lower_bound(batch_space, full_frame_user_spaces, frame_n_slots) > frame_n_slots:
        raise RuntimeError(
            "The requested average rates are not schedulable within one frame after exact slot quantization."
        )
    return int(frame_n_slots)


def slot_lower_bound(
    batch_space: BatchUserParameterSpace,
    full_frame_user_spaces,
    frame_n_slots,
):
    """Return the exact one-frame TDMA slot lower bound implied by the fastest user row."""

    required_slot_count = 0
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_space = full_frame_user_spaces[int(user_row.user_id)]
        required_slot_count += int(
            np.ceil(
                float(frame_n_slots)
                * float(user_row.required_rate_bps)
                / float(user_space["rate_active_bps"].max())
                - 1e-12
            )
        )
    return int(required_slot_count)


def quantize_and_prune_user_tdma_space(
    *,
    user_id: int,
    required_rate_bps: float,
    active_table: pd.DataFrame,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Quantize one user's full-frame rows onto the shared single-frame TDMA lattice."""

    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    rate_active_bps = active_table["rate_active_bps"].astype(float).to_numpy()
    required_slots = np.ceil(
        float(frame_n_slots) * float(required_rate_bps) / rate_active_bps - 1e-12
    ).astype(int)
    feasible_mask = (
        (rate_active_bps > 0.0)
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


def _row_schedule_power(row, *, frame_n_slots: int) -> float:
    """Return one user's frame-averaged DC power for the selected slot count."""

    return float(int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(frame_n_slots))


__all__ = [
    "prepare_joint_schedule_problem",
]
