from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from single_user_search.models import SingleUserRequest
from single_user_search.problem_factory import prepare_single_user_problem
from single_user_search.search import enumerate_active_candidates_from_context


ACTIVE_OPERATING_COLUMNS = [
    "user_id",
    "distance_m",
    "path_loss_db",
    "pa_id",
    "scenario_label",
    "pa_name",
    "bandwidth_hz",
    "n_prb",
    "layers",
    "mcs",
    "rate_active_bps",
    "p_dc_active_w",
    "p_out_total_w",
    "ps_total_w",
    "gamma_req_lin",
    "gamma_req_db",
    "gamma_achieved",
]

USER_CANDIDATE_COLUMNS = [
    "user_id",
    "distance_m",
    "path_loss_db",
    "required_rate_bps",
    "repeated_frames",
    "repeated_period_slots",
    "n_slots",
    "alpha_frame",
    "pa_id",
    "scenario_label",
    "pa_name",
    "bandwidth_hz",
    "n_prb",
    "layers",
    "mcs",
    "rate_active_bps",
    "rate_avg_frame_bps",
    "p_dc_active_w",
    "p_dc_avg_frame_w",
    "p_out_total_w",
    "p_out_avg_frame_w",
    "ps_total_w",
    "gamma_req_lin",
    "gamma_req_db",
    "gamma_achieved",
]

USER_CANDIDATE_REVIEW_COLUMNS = [
    "candidate_rank",
    "user_id",
    "distance_m",
    "path_loss_db",
    "n_slots",
    "n_prb",
    "mcs",
    "rank",
    "scenario_label",
    "required_rate_mbps",
    "achieved_rate_mbps",
    "rate_margin_mbps",
    "p_dc_active_w",
    "p_dc_avg_frame_w",
]

def build_user_candidate_review_table(candidate_table, *, top_n=None):
    """Build the compact reviewer-facing TDMA table for one user's candidate space.

    Steps:
    1. Keep the scheduler-owned raw table unchanged.
    2. Derive a compact ranking view from the existing candidate order.
    3. Rename `layers` to the reviewer-facing `rank` label for display.
    4. Return only the core parameters and evaluation metrics used in notebook review.
    """

    if candidate_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_REVIEW_COLUMNS)

    review_table = candidate_table.copy().reset_index(drop=True)
    review_table["candidate_rank"] = np.arange(1, len(review_table) + 1, dtype=int)
    review_table["rank"] = review_table["layers"].astype(int)
    review_table["required_rate_mbps"] = review_table["required_rate_bps"].astype(float) / 1e6
    review_table["achieved_rate_mbps"] = review_table["rate_avg_frame_bps"].astype(float) / 1e6
    review_table["rate_margin_mbps"] = (
        review_table["achieved_rate_mbps"] - review_table["required_rate_mbps"]
    )
    compact_table = review_table[USER_CANDIDATE_REVIEW_COLUMNS].copy()
    if top_n is None:
        return compact_table
    return compact_table.head(int(top_n)).reset_index(drop=True)


def build_user_candidate_review_tables(user_candidate_spaces, *, top_n=None):
    """Build one compact reviewer-facing TDMA table per user."""

    return {
        int(user_id): build_user_candidate_review_table(candidate_table, top_n=top_n)
        for user_id, candidate_table in user_candidate_spaces.items()
    }


def enumerate_user_active_operating_tables(
    user_table,
    *,
    system_cfg,
    model_inputs,
    search_shape,
    pa_catalog,
    outer_parallel=False,
    max_workers=None,
):
    """Enumerate one full-frame active operating table per user.

    Steps:
    1. Group users only by the deployment input that defines the radio channel: distance.
    2. Reuse the prepared single-user active table for each unique distance.
    3. Keep only full-frame operating points because TDMA burst candidates quantize frame shares later.
    4. Re-label the shared active table into the multi-user study schema for each user id.
    """

    users = user_table.copy()
    frame_slot_count = int(system_cfg.frame_slots)

    group_requests = {}
    for user_row in users.itertuples(index=False):
        group_key = float(user_row.distance_m)
        if group_key not in group_requests:
            group_requests[group_key] = float(user_row.distance_m)

    if bool(outer_parallel) and len(group_requests) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_active_group_worker,
                    group_key,
                    distance_m,
                    model_inputs,
                    search_shape,
                    pa_catalog,
                ): group_key
                for group_key, distance_m in group_requests.items()
            }
            grouped_active_tables = {}
            for future in as_completed(futures):
                group_key, active_table = future.result()
                grouped_active_tables[group_key] = active_table
    else:
        grouped_active_tables = {
            group_key: _build_distance_active_table(
                distance_m,
                model_inputs=model_inputs,
                search_shape=search_shape,
                pa_catalog=pa_catalog,
            )
            for group_key, distance_m in group_requests.items()
        }

    active_candidate_tables = {}
    for user_row in users.itertuples(index=False):
        active_table = _build_full_frame_active_table(
            grouped_active_tables[float(user_row.distance_m)],
            frame_slot_count=frame_slot_count,
        )
        if active_table.empty:
            active_candidate_tables[int(user_row.user_id)] = pd.DataFrame(columns=ACTIVE_OPERATING_COLUMNS)
            continue

        active_table = active_table.copy()
        active_table["user_id"] = int(user_row.user_id)
        active_table["distance_m"] = float(user_row.distance_m)
        active_table["rate_active_bps"] = active_table["rate_ach_bps"].astype(float)
        active_table["p_dc_active_w"] = active_table["p_dc_avg_total_w"].astype(float)
        active_candidate_tables[int(user_row.user_id)] = (
            active_table[ACTIVE_OPERATING_COLUMNS]
            .sort_values(
                ["p_dc_active_w", "bandwidth_hz", "n_prb", "mcs", "layers"],
                ascending=[True, True, True, True, True],
            )
            .reset_index(drop=True)
        )
    return active_candidate_tables


def resolve_repeated_frame_requirement(
    user_table,
    active_candidate_tables,
    *,
    frame_slot_count,
    max_repeated_frames,
):
    """Resolve the minimum repeated-frame count required by slot quantization."""

    share_rows = []
    exact_frame_share_sum = 0.0
    for user_row in user_table.itertuples(index=False):
        active_table = active_candidate_tables[int(user_row.user_id)]
        if active_table.empty:
            return {
                "status": "missing_active_operating_points",
                "user_id": int(user_row.user_id),
                "share_rows": pd.DataFrame(share_rows),
            }

        max_active_rate_bps = float(active_table["rate_active_bps"].max())
        required_rate_bps = float(user_row.required_rate_bps)
        if required_rate_bps > max_active_rate_bps:
            share_rows.append(
                {
                    "user_id": int(user_row.user_id),
                    "required_rate_bps": required_rate_bps,
                    "max_active_rate_bps": max_active_rate_bps,
                    "exact_frame_share_lb": np.inf,
                }
            )
            return {
                "status": "user_target_exceeds_active_rate",
                "user_id": int(user_row.user_id),
                "share_rows": pd.DataFrame(share_rows),
            }

        exact_frame_share_lb = required_rate_bps / max_active_rate_bps
        exact_frame_share_sum += exact_frame_share_lb
        share_rows.append(
            {
                "user_id": int(user_row.user_id),
                "required_rate_bps": required_rate_bps,
                "max_active_rate_bps": max_active_rate_bps,
                "exact_frame_share_lb": exact_frame_share_lb,
            }
        )

    share_df = pd.DataFrame(share_rows).sort_values("user_id").reset_index(drop=True)
    if exact_frame_share_sum > 1.0 + 1e-12:
        return {
            "status": "overloaded",
            "exact_frame_share_sum": float(exact_frame_share_sum),
            "share_rows": share_df,
        }

    for repeated_frames in range(1, int(max_repeated_frames) + 1):
        repeated_period_slots = int(repeated_frames * frame_slot_count)
        slot_lower_bound = int(
            sum(
                np.ceil(repeated_period_slots * float(row["exact_frame_share_lb"]) - 1e-12)
                for _, row in share_df.iterrows()
            )
        )
        if slot_lower_bound <= repeated_period_slots:
            return {
                "status": "ok",
                "min_repeated_frames": int(repeated_frames),
                "repeated_period_slots": int(repeated_period_slots),
                "slot_lower_bound": int(slot_lower_bound),
                "exact_frame_share_sum": float(exact_frame_share_sum),
                "share_rows": share_df,
            }

    return {
        "status": "max_repeated_frames_exceeded",
        "exact_frame_share_sum": float(exact_frame_share_sum),
        "share_rows": share_df,
    }


def build_user_candidate_spaces(
    user_table,
    active_candidate_tables,
    *,
    repeated_frames,
    frame_slot_count,
):
    """Build exact per-user TDMA candidate spaces for one repeated-frame count."""

    users = user_table.copy()

    repeated_period_slots = int(repeated_frames * frame_slot_count)
    user_candidate_spaces = {}
    summary_rows = []
    for user_row in users.itertuples(index=False):
        raw_candidate_table = build_user_candidate_space(
            user_row,
            active_candidate_tables[int(user_row.user_id)],
            repeated_frames=int(repeated_frames),
            repeated_period_slots=repeated_period_slots,
        )
        pruned_candidate_table = prune_exactly_dominated_user_space(raw_candidate_table)
        user_candidate_spaces[int(user_row.user_id)] = pruned_candidate_table
        summary_rows.append(
            {
                "user_id": int(user_row.user_id),
                "distance_m": float(user_row.distance_m),
                "path_loss_db": _resolve_user_path_loss_db(pruned_candidate_table, raw_candidate_table),
                "required_rate_bps": float(user_row.required_rate_bps),
                "repeated_period_slots": int(repeated_period_slots),
                "configs_raw": int(len(raw_candidate_table)),
                "configs_pruned": int(len(pruned_candidate_table)),
                "min_slots": int(pruned_candidate_table["n_slots"].min()) if not pruned_candidate_table.empty else np.nan,
                "max_slots": int(pruned_candidate_table["n_slots"].max()) if not pruned_candidate_table.empty else np.nan,
            }
        )

    return (
        user_candidate_spaces,
        pd.DataFrame(summary_rows).sort_values("user_id").reset_index(drop=True),
    )


def build_user_candidate_space(user_row, active_table, *, repeated_frames, repeated_period_slots):
    """Quantize one active operating table into exact TDMA rows for one repeated frame."""

    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    required_rate_bps = float(user_row.required_rate_bps)
    rate_active_bps = active_table["rate_active_bps"].astype(float).to_numpy()
    n_slots = np.ceil(repeated_period_slots * required_rate_bps / rate_active_bps - 1e-12).astype(int)
    feasible_mask = (rate_active_bps > 0.0) & (n_slots >= 1) & (n_slots <= int(repeated_period_slots))
    if not np.any(feasible_mask):
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    candidate_table = active_table.loc[feasible_mask].copy().reset_index(drop=True)
    candidate_table["required_rate_bps"] = required_rate_bps
    candidate_table["repeated_frames"] = int(repeated_frames)
    candidate_table["repeated_period_slots"] = int(repeated_period_slots)
    candidate_table["n_slots"] = n_slots[feasible_mask]
    candidate_table["alpha_frame"] = candidate_table["n_slots"].astype(float) / float(repeated_period_slots)
    candidate_table["rate_avg_frame_bps"] = candidate_table["alpha_frame"] * candidate_table["rate_active_bps"].astype(float)
    candidate_table["p_dc_avg_frame_w"] = candidate_table["alpha_frame"] * candidate_table["p_dc_active_w"].astype(float)
    candidate_table["p_out_avg_frame_w"] = candidate_table["alpha_frame"] * candidate_table["p_out_total_w"].astype(float)
    return (
        candidate_table[USER_CANDIDATE_COLUMNS]
        .sort_values(
            ["n_slots", "p_dc_avg_frame_w", "rate_avg_frame_bps", "bandwidth_hz", "n_prb", "mcs"],
            ascending=[True, True, False, True, True, True],
        )
        .reset_index(drop=True)
    )


def prune_exactly_dominated_user_space(candidate_table):
    """Remove rows that are strictly dominated in slots, frame-average power, and achieved rate."""

    if candidate_table.empty:
        return candidate_table.copy()

    ranked_table = candidate_table.sort_values(
        ["n_slots", "p_dc_avg_frame_w", "rate_avg_frame_bps"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    kept_rows = []
    for row in ranked_table.to_dict("records"):
        if any(
            int(kept_row["n_slots"]) <= int(row["n_slots"])
            and float(kept_row["p_dc_avg_frame_w"]) <= float(row["p_dc_avg_frame_w"])
            and float(kept_row["rate_avg_frame_bps"]) >= float(row["rate_avg_frame_bps"])
            and (
                int(kept_row["n_slots"]) < int(row["n_slots"])
                or float(kept_row["p_dc_avg_frame_w"]) < float(row["p_dc_avg_frame_w"])
                or float(kept_row["rate_avg_frame_bps"]) > float(row["rate_avg_frame_bps"])
            )
            for kept_row in kept_rows
        ):
            continue
        kept_rows.append(row)
    return pd.DataFrame(kept_rows, columns=USER_CANDIDATE_COLUMNS)


def build_active_candidate_summary_df(user_table, active_candidate_tables):
    """Summarize the active operating-point catalog size and extremes for each user."""

    rows = []
    for user_row in user_table.itertuples(index=False):
        active_table = active_candidate_tables[int(user_row.user_id)]
        rows.append(
            {
                "user_id": int(user_row.user_id),
                "distance_m": float(user_row.distance_m),
                "path_loss_db": _resolve_user_path_loss_db(active_table),
                "required_rate_bps": float(user_row.required_rate_bps),
                "active_operating_points": int(len(active_table)),
                "max_active_rate_mbps": float(active_table["rate_active_bps"].max()) / 1e6 if not active_table.empty else np.nan,
                "min_active_power_w": float(active_table["p_dc_active_w"].min()) if not active_table.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)


def _evaluate_active_group_worker(
    group_key,
    distance_m,
    model_inputs,
    search_shape,
    pa_catalog,
):
    """Evaluate one shared active table in a worker process."""

    return (
        group_key,
        _build_distance_active_table(
            float(distance_m),
            model_inputs=model_inputs,
            search_shape=search_shape,
            pa_catalog=pa_catalog,
        ),
    )


def _build_distance_active_table(distance_m, *, model_inputs, search_shape, pa_catalog):
    """Build one resolved full-search active table for a shared deployment distance."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )
    return enumerate_active_candidates_from_context(context)


def _resolve_user_path_loss_db(*candidate_tables):
    """Return the resolved path loss shown in one user's candidate tables."""

    for candidate_table in candidate_tables:
        if candidate_table.empty:
            continue
        return float(candidate_table["path_loss_db"].iloc[0])
    return np.nan


def _build_full_frame_active_table(active_table, *, frame_slot_count):
    """Keep only full-frame active operating points from the shared single-user table."""

    if active_table.empty:
        return active_table.copy()
    return active_table[active_table["n_slots_on"] == int(frame_slot_count)].copy().reset_index(drop=True)
