from __future__ import annotations

"""Lean day-cycle support for Notebook 2 and downstream notebook examples."""

from pathlib import Path

import numpy as np
import pandas as pd

from day_cycle_simulation.generation import (
    _build_scheduler_day_user_table_from_sessions,
    generate_synthetic_day_population,
)
from day_cycle_simulation.load_curve import (
    BITS_PER_GB,
    build_15_minute_target_load_table,
    load_hourly_load_curve,
)


def export_doc_figure(fig, filename: str, doc_img_dir: Path) -> Path:
    """Save one notebook figure into the repository image directory."""

    output_path = Path(doc_img_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def bin_index_to_clock(bin_index: int, *, bin_duration_min: int = 15) -> str:
    """Convert one quarter-hour bin index into a clock label."""

    total_minutes = int(bin_index) * int(bin_duration_min)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def build_day_cycle_discussion_artifacts(load_curve_csv: Path, config) -> dict[str, object]:
    """Build the compact day-cycle views used across the notebook walkthroughs.

    Steps:
    1. Load and expand the hourly load curve with the production day-cycle layer.
    2. Generate one deterministic synthetic session population from that target curve.
    3. Keep only the lean notebook tables that explain the demand-generation flow.
    """

    hourly_load_table = load_hourly_load_curve(load_curve_csv)
    target_load_table = build_15_minute_target_load_table(hourly_load_table)
    sessions = generate_synthetic_day_population(
        config=config,
        target_load_table=target_load_table,
    )
    scheduler_day_user_table = _build_scheduler_day_user_table_from_sessions(
        sessions,
        bin_duration_s=float(config.bin_duration_s),
    )
    session_table = _build_session_table(
        sessions,
        bin_duration_s=float(config.bin_duration_s),
    )
    lane_table = _assign_session_lanes(session_table)
    bin_validation_table = _build_bin_validation_table(
        target_load_table=target_load_table,
        scheduler_day_user_table=scheduler_day_user_table,
        day_bin_count=int(config.day_bin_count),
        bin_duration_s=float(config.bin_duration_s),
    )
    example_session_view, example_scheduler_rows = _build_example_views(
        session_table,
        scheduler_day_user_table=scheduler_day_user_table,
    )

    return {
        "hourly_load_table": hourly_load_table.copy(),
        "target_load_table": target_load_table.copy(),
        "lane_table": lane_table,
        "bin_validation_table": bin_validation_table,
        "example_session_view": example_session_view,
        "example_scheduler_rows": example_scheduler_rows,
        "scheduler_day_user_table": scheduler_day_user_table,
    }


def _build_session_table(sessions, *, bin_duration_s: float) -> pd.DataFrame:
    rows = []
    for session in sessions:
        duration_bins = int(session.nominal_duration_bins)
        start_bin = int(session.start_bin)
        stop_bin = start_bin + duration_bins
        rows.append(
            {
                "session_id": int(session.session_id),
                "distance_m": float(session.distance_m),
                "total_data_gb": float(session.total_data_bits) / BITS_PER_GB,
                "start_bin": int(start_bin),
                "stop_bin": int(stop_bin),
                "start_time": bin_index_to_clock(start_bin),
                "stop_time": bin_index_to_clock(stop_bin),
                "duration_bins": int(duration_bins),
                "duration_min": int(duration_bins * float(bin_duration_s) / 60.0),
                "required_rate_mbps": float(
                    float(session.total_data_bits) / float(duration_bins) / float(bin_duration_s) / 1e6
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["start_bin", "stop_bin", "session_id"])
        .reset_index(drop=True)
    )


def _assign_session_lanes(session_table: pd.DataFrame) -> pd.DataFrame:
    if session_table.empty:
        return session_table.assign(lane_index=pd.Series(dtype=int))

    lane_stop_by_index: list[int] = []
    lane_indices: list[int] = []
    for session_row in session_table.itertuples(index=False):
        assigned_lane = _find_available_lane(
            start_bin=int(session_row.start_bin),
            lane_stop_by_index=lane_stop_by_index,
        )
        lane_indices.append(int(assigned_lane))
        lane_stop_by_index[assigned_lane] = int(session_row.stop_bin)

    return session_table.assign(lane_index=lane_indices).reset_index(drop=True)


def _find_available_lane(*, start_bin: int, lane_stop_by_index: list[int]) -> int:
    for lane_index, stop_bin in enumerate(lane_stop_by_index):
        if int(stop_bin) <= int(start_bin):
            return int(lane_index)

    lane_stop_by_index.append(int(start_bin))
    return len(lane_stop_by_index) - 1


def _build_bin_validation_table(
    *,
    target_load_table: pd.DataFrame,
    scheduler_day_user_table: pd.DataFrame,
    day_bin_count: int,
    bin_duration_s: float,
) -> pd.DataFrame:
    rebuilt_load_by_bin = (
        scheduler_day_user_table.groupby("bin_index", dropna=False)["required_rate_bps"]
        .sum()
        .mul(float(bin_duration_s) / BITS_PER_GB)
    )
    active_users_by_bin = scheduler_day_user_table.groupby("bin_index", dropna=False)["user_id"].nunique()
    rows = []
    for bin_index in range(int(day_bin_count)):
        target_bits_in_bin = float(
            target_load_table.loc[target_load_table["bin_index"].eq(int(bin_index)), "target_bits_in_bin"].iloc[0]
        )
        rebuilt_load_gb = float(rebuilt_load_by_bin.get(int(bin_index), 0.0))
        rows.append(
            {
                "bin_index": int(bin_index),
                "target_load_gb_in_bin": float(target_bits_in_bin / BITS_PER_GB),
                "rebuilt_load_gb_in_bin": float(rebuilt_load_gb),
                "residual_load_gb_in_bin": float(target_bits_in_bin / BITS_PER_GB - rebuilt_load_gb),
                "active_users": int(active_users_by_bin.get(int(bin_index), 0)),
            }
        )

    return pd.DataFrame(rows)


def _build_example_views(
    session_table: pd.DataFrame,
    *,
    scheduler_day_user_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if session_table.empty:
        empty_session_view = pd.DataFrame(
            columns=[
                "Session ID",
                "Distance (m)",
                "Total data (GB)",
                "Start time",
                "Stop time",
                "Duration (bins)",
                "Duration (min)",
                "Required rate (Mbps)",
            ]
        )
        empty_scheduler_rows = pd.DataFrame(
            columns=[
                "Bin index",
                "User ID",
                "Distance (m)",
                "Required rate (Mbps)",
            ]
        )
        return empty_session_view, empty_scheduler_rows

    example_session = session_table.iloc[0]
    example_session_view = pd.DataFrame(
        [
            {
                "Session ID": int(example_session["session_id"]),
                "Distance (m)": float(example_session["distance_m"]),
                "Total data (GB)": float(example_session["total_data_gb"]),
                "Start time": str(example_session["start_time"]),
                "Stop time": str(example_session["stop_time"]),
                "Duration (bins)": int(example_session["duration_bins"]),
                "Duration (min)": int(example_session["duration_min"]),
                "Required rate (Mbps)": float(example_session["required_rate_mbps"]),
            }
        ]
    )
    example_scheduler_rows = (
        scheduler_day_user_table.loc[
            scheduler_day_user_table["user_id"].eq(int(example_session["session_id"])),
            ["bin_index", "user_id", "distance_m", "required_rate_bps"],
        ]
        .assign(required_rate_mbps=lambda table: table["required_rate_bps"].astype(float) / 1e6)
        .rename(
            columns={
                "bin_index": "Bin index",
                "user_id": "User ID",
                "distance_m": "Distance (m)",
                "required_rate_mbps": "Required rate (Mbps)",
            }
        )[["Bin index", "User ID", "Distance (m)", "Required rate (Mbps)"]]
        .reset_index(drop=True)
    )
    return example_session_view, example_scheduler_rows


__all__ = [
    "BITS_PER_GB",
    "bin_index_to_clock",
    "build_day_cycle_discussion_artifacts",
    "export_doc_figure",
]
