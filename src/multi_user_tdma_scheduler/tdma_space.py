from __future__ import annotations

import numpy as np
import pandas as pd

from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointScheduleProblem, USER_CANDIDATE_COLUMNS


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
    *,
    window_n_frames=None,
    max_window_n_frames=32,
) -> PreparedJointScheduleProblem:
    """Prepare the exact scheduler problem from a trusted batch parameter-space artifact.

    Steps:
    1. Read the trusted per-user full-frame feasible spaces from the batch artifact.
    2. Resolve the repeated scheduling window in whole frames.
    3. Quantize each user space onto the exact TDMA slot lattice for that window.
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
    resolved_window_n_frames = resolve_scheduling_window(
        batch_space,
        full_frame_user_spaces,
        window_n_frames=window_n_frames,
        max_window_n_frames=max_window_n_frames,
    )
    window_n_slots = int(resolved_window_n_frames) * int(batch_space.frame_n_slots)
    user_candidate_spaces = {
        int(user_row.user_id): quantize_and_prune_user_tdma_space(
            user_id=int(user_row.user_id),
            required_rate_bps=float(user_row.required_rate_bps),
            active_table=full_frame_user_spaces[int(user_row.user_id)],
            window_n_slots=window_n_slots,
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    return PreparedJointScheduleProblem(
        window_n_frames=int(resolved_window_n_frames),
        window_n_slots=int(window_n_slots),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_candidate_spaces=user_candidate_spaces,
    )


def resolve_scheduling_window(
    batch_space: BatchUserParameterSpace,
    full_frame_user_spaces,
    *,
    window_n_frames=None,
    max_window_n_frames,
):
    """Resolve the repeated scheduling window in whole frames."""

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
            "The requested average rates are infeasible within any repeated scheduling window: "
            f"exact frame-share lower bound = {float(exact_frame_share_sum):.3f} > 1.0."
        )

    frame_n_slots = int(batch_space.frame_n_slots)
    if window_n_frames is not None:
        resolved_window_n_frames = int(window_n_frames)
        if resolved_window_n_frames < 1:
            raise ValueError("window_n_frames must be at least 1.")
        window_n_slots = resolved_window_n_frames * frame_n_slots
        if slot_lower_bound(batch_space, full_frame_user_spaces, window_n_slots) > window_n_slots:
            raise RuntimeError(
                f"window_n_frames={resolved_window_n_frames} does not provide enough slots for the requested average rates."
            )
        return int(resolved_window_n_frames)

    for resolved_window_n_frames in range(1, int(max_window_n_frames) + 1):
        window_n_slots = int(resolved_window_n_frames) * frame_n_slots
        # The fastest full-frame row for each user gives an exact lower bound on
        # how many TDMA slots any feasible window must contain.
        if slot_lower_bound(batch_space, full_frame_user_spaces, window_n_slots) <= window_n_slots:
            return int(resolved_window_n_frames)

    raise RuntimeError(
        "Could not resolve a finite scheduling window within "
        f"{int(max_window_n_frames)} repeated frames."
    )


def slot_lower_bound(
    batch_space: BatchUserParameterSpace,
    full_frame_user_spaces,
    window_n_slots,
):
    """Return the exact TDMA slot lower bound implied by the fastest user row."""

    required_slot_count = 0
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_space = full_frame_user_spaces[int(user_row.user_id)]
        required_slot_count += int(
            np.ceil(
                float(window_n_slots)
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
    window_n_slots: int,
) -> pd.DataFrame:
    """Quantize one user's full-frame rows onto the exact TDMA slot lattice."""

    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    rate_active_bps = active_table["rate_active_bps"].astype(float).to_numpy()
    required_slots = np.ceil(
        float(window_n_slots) * float(required_rate_bps) / rate_active_bps - 1e-12
    ).astype(int)
    feasible_mask = (
        (rate_active_bps > 0.0)
        & (required_slots >= 1)
        & (required_slots <= int(window_n_slots))
    )
    if not np.any(feasible_mask):
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_table = active_table.loc[feasible_mask].copy().reset_index(drop=True)
    candidate_table["user_id"] = int(user_id)
    candidate_table["n_slots"] = required_slots[feasible_mask]

    # The lookup table stores full-frame active rows. TDMA prep rescales those
    # rows onto the exact repeated window chosen for this batch.
    slot_share = candidate_table["n_slots"].astype(float) / float(window_n_slots)
    candidate_table["rate_avg_frame_bps"] = slot_share * candidate_table["rate_active_bps"].astype(float)
    candidate_table["p_dc_avg_frame_w"] = slot_share * candidate_table["p_dc_active_w"].astype(float)
    candidate_table["p_out_avg_frame_w"] = slot_share * candidate_table["p_out_total_w"].astype(float)
    return exact_prune_user_tdma_space(candidate_table[USER_CANDIDATE_COLUMNS].copy())


def exact_prune_user_tdma_space(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exactly dominated without erasing PA-family alternatives."""

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    ranked_rows = candidate_table.sort_values(
        ["pa_id", "n_slots", "p_dc_avg_frame_w", "rate_avg_frame_bps", "n_prb", "mcs"],
        ascending=[True, True, True, False, True, True],
    ).to_dict("records")

    kept_rows = []
    kept_rows_by_pa = {}
    for row in ranked_rows:
        kept_rows_for_pa = kept_rows_by_pa.setdefault(int(row["pa_id"]), [])
        if any(
            int(kept_row["n_slots"]) <= int(row["n_slots"])
            and float(kept_row["p_dc_avg_frame_w"]) <= float(row["p_dc_avg_frame_w"])
            and float(kept_row["rate_avg_frame_bps"]) >= float(row["rate_avg_frame_bps"])
            and (
                int(kept_row["n_slots"]) < int(row["n_slots"])
                or float(kept_row["p_dc_avg_frame_w"]) < float(row["p_dc_avg_frame_w"])
                or float(kept_row["rate_avg_frame_bps"]) > float(row["rate_avg_frame_bps"])
            )
            for kept_row in kept_rows_for_pa
        ):
            continue

        kept_rows_for_pa.append(row)
        kept_rows.append(row)

    return pd.DataFrame(kept_rows, columns=USER_CANDIDATE_COLUMNS).reset_index(drop=True)


__all__ = [
    "prepare_joint_schedule_problem",
]
