from __future__ import annotations

"""Load-chain breakpoint analysis for scheduler-comparison HPC results."""

import pandas as pd

from .row_states import bool_like, certified_skipped_row_mask


LOAD_CHAIN_COLUMNS = (
    "scheduler_mode",
    "switch_policy",
    "active_user_count",
    "distance_min_m",
    "distance_max_m",
    "distance_model",
    "mean_distance_m",
    "sigma_distance_m",
    "reference_backlog_bits",
    "frame_duration_s",
)


def build_breakpoint_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sorted_results = results.sort_values(list(LOAD_CHAIN_COLUMNS) + ["load_factor", "point_id"])
    for chain_key, chain in sorted_results.groupby(list(LOAD_CHAIN_COLUMNS), dropna=False):
        chain = chain.sort_values(["load_factor", "point_id"]).reset_index(drop=True)
        rows.append(build_breakpoint_row(chain_key, chain))

    return pd.DataFrame(rows)


def build_infeasibility_reason_summary(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["infeasible_reason"] = frame["infeasible_reason"].fillna("")
    frame["source_bound"] = frame["source_bound"].fillna("")
    grouped = frame.groupby(
        ["scheduler_mode", "switch_policy", "status", "infeasible_reason", "source_bound"],
        dropna=False,
    )
    return grouped.agg(
        row_count=("point_id", "count"),
        min_load_factor=("load_factor", "min"),
        max_load_factor=("load_factor", "max"),
        min_active_user_count=("active_user_count", "min"),
        max_active_user_count=("active_user_count", "max"),
    ).reset_index().sort_values(
        ["row_count", "scheduler_mode", "switch_policy"],
        ascending=[False, True, True],
    )


def build_breakpoint_row(chain_key: tuple[object, ...], chain: pd.DataFrame) -> dict[str, object]:
    solved_mask = solved_row_mask(chain)
    unsolved = chain.loc[~solved_mask]
    infeasible = chain.loc[infeasible_row_mask(chain)]
    skipped = chain.loc[certified_skipped_row_mask(chain)]
    failed_or_other = chain.loc[failed_or_other_row_mask(chain)]

    first_unsolved = first_row(unsolved)
    first_infeasible = first_row(infeasible)
    first_skipped = first_row(skipped)
    first_failed_or_other = first_row(failed_or_other)
    last_solved = last_row(chain.loc[solved_mask])

    first_unsolved_index = None if first_unsolved is None else int(first_unsolved.name)
    solved_after_unsolved = False
    if first_unsolved_index is not None:
        solved_after_unsolved = bool(solved_mask.iloc[first_unsolved_index + 1 :].any())

    row = dict(zip(LOAD_CHAIN_COLUMNS, chain_key))
    row.update(
        {
            "point_count": int(len(chain)),
            "observed_load_min": float(chain["load_factor"].min()),
            "observed_load_max": float(chain["load_factor"].max()),
            "solved_count": int(solved_mask.sum()),
            "infeasible_count": int(infeasible_row_mask(chain).sum()),
            "certified_skipped_count": int(certified_skipped_row_mask(chain).sum()),
            "failed_or_other_count": int(failed_or_other_row_mask(chain).sum()),
            "last_solved_load_factor": row_value(last_solved, "load_factor"),
            "last_solved_point_id": row_value(last_solved, "point_id"),
            "first_unsolved_load_factor": row_value(first_unsolved, "load_factor"),
            "first_unsolved_point_id": row_value(first_unsolved, "point_id"),
            "first_unsolved_status": row_value(first_unsolved, "status"),
            "first_unsolved_reason": row_value(first_unsolved, "infeasible_reason"),
            "first_infeasible_load_factor": row_value(first_infeasible, "load_factor"),
            "first_infeasible_point_id": row_value(first_infeasible, "point_id"),
            "first_infeasible_reason": row_value(first_infeasible, "infeasible_reason"),
            "first_certified_skipped_load_factor": row_value(first_skipped, "load_factor"),
            "first_certified_skipped_point_id": row_value(first_skipped, "point_id"),
            "first_certified_skipped_source_point_id": row_value(first_skipped, "source_point_id"),
            "first_certified_skipped_source_bound": row_value(first_skipped, "source_bound"),
            "first_failed_or_other_load_factor": row_value(first_failed_or_other, "load_factor"),
            "first_failed_or_other_point_id": row_value(first_failed_or_other, "point_id"),
            "first_failed_or_other_status": row_value(first_failed_or_other, "status"),
            "breakpoint_category": breakpoint_category(
                solved_after_unsolved=solved_after_unsolved,
                first_unsolved=first_unsolved,
                first_infeasible=first_infeasible,
                first_skipped=first_skipped,
                first_failed_or_other=first_failed_or_other,
            ),
            "unexpected_breakpoint_flag": bool(solved_after_unsolved),
        }
    )
    return row


def breakpoint_category(
    *,
    solved_after_unsolved: bool,
    first_unsolved: pd.Series | None,
    first_infeasible: pd.Series | None,
    first_skipped: pd.Series | None,
    first_failed_or_other: pd.Series | None,
) -> str:
    if solved_after_unsolved:
        return "mixed_or_nonmonotone"
    if first_unsolved is None:
        return "all_solved"
    if first_failed_or_other is not None and same_point(first_unsolved, first_failed_or_other):
        return "first_failed_or_other"
    if first_skipped is not None and same_point(first_unsolved, first_skipped):
        return "first_certified_skipped"
    if first_infeasible is not None and same_point(first_unsolved, first_infeasible):
        return "first_infeasible"
    return "mixed_or_nonmonotone"


def solved_row_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["status"].astype(str).eq("solved") & frame["feasible"].map(bool_like)


def infeasible_row_mask(frame: pd.DataFrame) -> pd.Series:
    skip_reason = frame["skip_reason"].fillna("").astype(str)
    status = frame["status"].fillna("").astype(str)
    feasible = frame["feasible"].map(bool_like)
    return (~feasible) & skip_reason.eq("") & ~status.eq("certified_skipped")


def failed_or_other_row_mask(frame: pd.DataFrame) -> pd.Series:
    status = frame["status"].fillna("").astype(str)
    solved_or_expected_infeasible = status.isin(("solved", "infeasible", "certified_skipped"))
    return ~solved_or_expected_infeasible


def first_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]


def last_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[-1]


def row_value(row: pd.Series | None, column: str) -> object:
    if row is None:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return value


def same_point(left: pd.Series, right: pd.Series) -> bool:
    return str(left["point_id"]) == str(right["point_id"])


__all__ = [
    "build_breakpoint_summary",
    "build_infeasibility_reason_summary",
]
