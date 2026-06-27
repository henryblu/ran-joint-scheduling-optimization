from __future__ import annotations

"""Reviewer-facing summary tables for scheduler-comparison results."""

import pandas as pd

from .breakpoints import LOAD_CHAIN_COLUMNS, infeasible_row_mask, solved_row_mask
from .row_states import certified_skipped_row_mask


def build_scheduler_summary(results: pd.DataFrame) -> pd.DataFrame:
    frame = result_state_frame(results)
    return aggregate_summary(frame, ["scheduler_mode"])


def build_policy_summary(results: pd.DataFrame) -> pd.DataFrame:
    frame = result_state_frame(results)
    return aggregate_summary(frame, ["scheduler_mode", "switch_policy"])


def build_load_chain_summary(results: pd.DataFrame) -> pd.DataFrame:
    frame = result_state_frame(results)
    grouped = frame.groupby(list(LOAD_CHAIN_COLUMNS), dropna=False)
    return grouped.agg(
        point_count=("point_id", "count"),
        solved_count=("is_solved", "sum"),
        infeasible_count=("is_infeasible", "sum"),
        certified_skipped_count=("is_certified_skipped", "sum"),
        min_load_factor=("load_factor", "min"),
        max_load_factor=("load_factor", "max"),
        min_solved_load_factor=("solved_load_factor", "min"),
        max_solved_load_factor=("solved_load_factor", "max"),
        min_frame_energy_j=("frame_energy_numeric", "min"),
        mean_frame_energy_j=("frame_energy_numeric", "mean"),
    ).reset_index()


def result_state_frame(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["is_solved"] = solved_row_mask(frame)
    frame["is_infeasible"] = infeasible_row_mask(frame)
    frame["is_certified_skipped"] = certified_skipped_row_mask(frame)
    frame["frame_energy_numeric"] = pd.to_numeric(frame["frame_energy_j"], errors="coerce")
    frame["average_power_numeric"] = pd.to_numeric(frame["average_frame_dc_power_w"], errors="coerce")
    frame["solved_load_factor"] = pd.to_numeric(frame["load_factor"], errors="coerce").where(frame["is_solved"])
    return frame


def aggregate_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)
    summary = grouped.agg(
        row_count=("point_id", "count"),
        solved_count=("is_solved", "sum"),
        infeasible_count=("is_infeasible", "sum"),
        certified_skipped_count=("is_certified_skipped", "sum"),
        min_load_factor=("load_factor", "min"),
        max_load_factor=("load_factor", "max"),
        min_frame_energy_j=("frame_energy_numeric", "min"),
        mean_frame_energy_j=("frame_energy_numeric", "mean"),
        min_average_power_w=("average_power_numeric", "min"),
        mean_average_power_w=("average_power_numeric", "mean"),
    ).reset_index()
    summary["solved_fraction"] = summary["solved_count"] / summary["row_count"]
    return summary


__all__ = [
    "build_load_chain_summary",
    "build_policy_summary",
    "build_scheduler_summary",
]
