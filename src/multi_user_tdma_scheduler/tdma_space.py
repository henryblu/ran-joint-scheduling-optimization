from __future__ import annotations

import numpy as np
import pandas as pd

from single_user_parameter_space.models import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointScheduleProblem

FULL_FRAME_ACTIVE_COLUMNS = list(BATCH_USER_PARAMETER_SPACE_COLUMNS)


USER_CANDIDATE_COLUMNS = [
    "user_id",
    "pa_id",
    "bandwidth_hz",
    "n_prb",
    "layers",
    "mcs",
    "n_slots",
    "rate_avg_frame_bps",
    "p_dc_avg_frame_w",
    "p_out_avg_frame_w",
]


def prepare_joint_schedule_problem(
    batch_space,
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

    full_frame_user_spaces = {}
    for user_row in batch_space.user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        candidate_table = batch_space.user_parameter_spaces.get(user_id)
        if candidate_table is None or candidate_table.empty:
            full_frame_user_spaces[user_id] = pd.DataFrame(columns=FULL_FRAME_ACTIVE_COLUMNS)
            continue
        full_frame_user_spaces[user_id] = (
            candidate_table[FULL_FRAME_ACTIVE_COLUMNS].copy().reset_index(drop=True)
        )
    resolved_window_n_frames = resolve_scheduling_window(
        batch_space,
        full_frame_user_spaces,
        window_n_frames=window_n_frames,
        max_window_n_frames=max_window_n_frames,
    )
    window_n_slots = int(resolved_window_n_frames) * int(batch_space.frame_n_slots)
    quantized_user_spaces = quantize_user_tdma_spaces(
        batch_space,
        full_frame_user_spaces,
        window_n_slots=window_n_slots,
    )
    pruned_user_spaces = prune_user_tdma_spaces(quantized_user_spaces)
    return PreparedJointScheduleProblem(
        window_n_frames=int(resolved_window_n_frames),
        window_n_slots=int(window_n_slots),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_candidate_spaces=pruned_user_spaces,
    )


def resolve_scheduling_window(
    batch_space,
    full_frame_user_spaces,
    *,
    window_n_frames=None,
    max_window_n_frames,
):
    """Resolve the repeated scheduling window in whole frames."""

    frame_n_slots = int(batch_space.frame_n_slots)
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
        window_n_slots = int(resolved_window_n_frames * frame_n_slots)
        if slot_lower_bound(batch_space, full_frame_user_spaces, window_n_slots) <= window_n_slots:
            return int(resolved_window_n_frames)

    raise RuntimeError(
        "Could not resolve a finite scheduling window within "
        f"{int(max_window_n_frames)} repeated frames."
    )


def slot_lower_bound(batch_space, full_frame_user_spaces, window_n_slots):
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


def quantize_user_tdma_spaces(batch_space, full_frame_user_spaces, *, window_n_slots):
    """Quantize each user space onto the exact TDMA slot lattice for one window."""

    quantized_user_spaces = {}
    for user_row in batch_space.user_requirements.itertuples(index=False):
        active_table = full_frame_user_spaces[int(user_row.user_id)]
        required_rate_bps = float(user_row.required_rate_bps)
        rate_active_bps = active_table["rate_active_bps"].astype(float).to_numpy()
        required_slots = np.ceil(
            float(window_n_slots) * required_rate_bps / rate_active_bps - 1e-12
        ).astype(int)
        feasible_mask = (
            (rate_active_bps > 0.0)
            & (required_slots >= 1)
            & (required_slots <= int(window_n_slots))
        )
        if not np.any(feasible_mask):
            quantized_user_spaces[int(user_row.user_id)] = pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)
            continue

        candidate_table = active_table.loc[feasible_mask].copy().reset_index(drop=True)
        candidate_table["user_id"] = int(user_row.user_id)
        candidate_table["n_slots"] = required_slots[feasible_mask]
        slot_share = candidate_table["n_slots"].astype(float) / float(window_n_slots)
        candidate_table["rate_avg_frame_bps"] = slot_share * candidate_table["rate_active_bps"].astype(float)
        candidate_table["p_dc_avg_frame_w"] = slot_share * candidate_table["p_dc_active_w"].astype(float)
        candidate_table["p_out_avg_frame_w"] = slot_share * candidate_table["p_out_total_w"].astype(float)
        quantized_user_spaces[int(user_row.user_id)] = candidate_table[USER_CANDIDATE_COLUMNS].copy()

    return quantized_user_spaces


def prune_user_tdma_spaces(user_tdma_spaces):
    """Exact-prune dominated per-user TDMA rows before the joint search."""

    return {
        int(user_id): exact_prune_user_tdma_space(candidate_table)
        for user_id, candidate_table in user_tdma_spaces.items()
    }


def exact_prune_user_tdma_space(candidate_table):
    """Drop rows that are exactly dominated without erasing PA-family alternatives.

    The joint scheduler may prefer one PA family over another even when a row on a
    different bank is locally better on slots or average-frame power. We therefore
    only prune rows against competitors on the same PA bank.
    """

    ranked_rows = candidate_table.sort_values(
        ["pa_id", "n_slots", "p_dc_avg_frame_w", "rate_avg_frame_bps", "bandwidth_hz", "n_prb", "mcs"],
        ascending=[True, True, True, False, True, True, True],
    ).to_dict("records")

    kept_rows = []
    kept_rows_by_pa = {}
    for row in ranked_rows:
        pa_id = int(row["pa_id"])
        kept_rows_for_pa = kept_rows_by_pa.setdefault(pa_id, [])
        if any(_same_pa_row_dominates(kept_row, row) for kept_row in kept_rows_for_pa):
            continue

        kept_rows_for_pa.append(row)
        kept_rows.append(row)

    return pd.DataFrame(kept_rows, columns=USER_CANDIDATE_COLUMNS).reset_index(drop=True)


def _same_pa_row_dominates(left_row, right_row):
    """Return whether one row strictly dominates another on the same PA bank."""

    return (
        int(left_row["n_slots"]) <= int(right_row["n_slots"])
        and float(left_row["p_dc_avg_frame_w"]) <= float(right_row["p_dc_avg_frame_w"])
        and float(left_row["rate_avg_frame_bps"]) >= float(right_row["rate_avg_frame_bps"])
        and (
            int(left_row["n_slots"]) < int(right_row["n_slots"])
            or float(left_row["p_dc_avg_frame_w"]) < float(right_row["p_dc_avg_frame_w"])
            or float(left_row["rate_avg_frame_bps"]) > float(right_row["rate_avg_frame_bps"])
        )
    )

__all__ = [
    "prepare_joint_schedule_problem",
]
