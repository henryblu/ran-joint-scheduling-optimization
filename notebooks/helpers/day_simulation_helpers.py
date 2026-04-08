from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from day_cycle_simulation.generation import build_scheduler_day_user_table
from day_run import run_bin
from models import PASwitchPolicy


def build_day_simulation_artifacts(
    *,
    load_curve_csv,
    session_generation_config,
    switch_policy: PASwitchPolicy = PASwitchPolicy.STANDBY,
    target_user_count: int = 4,
) -> SimpleNamespace:
    """Build the day-simulation preview used in Notebook 7.

    Steps:
    1. Rebuild the scheduler-facing user table for the full day.
    2. Summarize that demand as one small per-bin overview table.
    3. Solve one representative non-empty bin to show how the day runner uses the table.
    """

    scheduler_day_user_table = build_scheduler_day_user_table(
        load_curve_csv=load_curve_csv,
        config=session_generation_config,
    )
    bin_summary_table = _build_bin_summary_table(scheduler_day_user_table)
    example_bin_index = _pick_example_bin(
        bin_summary_table,
        target_user_count=int(target_user_count),
    )
    example_user_table = (
        scheduler_day_user_table.loc[
            scheduler_day_user_table["bin_index"].eq(int(example_bin_index)),
            ["user_id", "distance_m", "required_rate_bps"],
        ]
        .copy()
        .reset_index(drop=True)
    )
    example_result = run_bin(
        int(example_bin_index),
        example_user_table,
        None,
        switch_policy,
    )

    return SimpleNamespace(
        scheduler_day_user_table=scheduler_day_user_table,
        bin_summary_table=bin_summary_table,
        example_bin_index=int(example_bin_index),
        example_user_table=example_user_table,
        example_result=example_result,
        example_schedule_summary=_build_example_schedule_summary(example_result),
        example_allocation_table=_build_example_allocation_table(example_result),
    )


def _build_bin_summary_table(scheduler_day_user_table: pd.DataFrame) -> pd.DataFrame:
    """Return one compact demand summary row per active day bin."""

    return (
        scheduler_day_user_table.groupby("bin_index", as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            requested_rate_bps=("required_rate_bps", "sum"),
        )
        .assign(requested_rate_mbps=lambda table: table["requested_rate_bps"] / 1e6)
        .sort_values("bin_index")
        .reset_index(drop=True)
    )


def _pick_example_bin(
    bin_summary_table: pd.DataFrame,
    *,
    target_user_count: int,
) -> int:
    """Pick one small non-empty bin for a cheap illustrative `run_bin` solve."""

    non_empty_bins = bin_summary_table.loc[bin_summary_table["user_count"].gt(0)].copy()
    if non_empty_bins.empty:
        raise ValueError("The generated day does not contain any active bins.")

    non_empty_bins["distance_to_target"] = (
        non_empty_bins["user_count"].astype(int) - int(target_user_count)
    ).abs()
    chosen_row = non_empty_bins.sort_values(
        ["distance_to_target", "user_count", "bin_index"],
        ascending=[True, True, True],
    ).iloc[0]
    return int(chosen_row["bin_index"])


def _build_example_schedule_summary(example_result) -> pd.DataFrame:
    """Return the lean solved-bin summary shown in the day-simulation notebook."""

    best_schedule = {} if example_result.best_schedule is None else example_result.best_schedule
    return pd.DataFrame(
        [
            {
                "bin_index": int(example_result.bin_index),
                "status": str(example_result.status),
                "user_count": int(example_result.user_count),
                "scheduled_users": int(len(best_schedule.get("rows", []))),
                "used_slots": float(best_schedule.get("slot_total", float("nan"))),
                "unused_slots": float(best_schedule.get("unused_slots", float("nan"))),
                "delivered_rate_mbps": float(best_schedule.get("total_rate_bps", float("nan"))) / 1e6,
                "dc_total_w": float(best_schedule.get("schedule_p_dc_total_avg_frame_w", float("nan"))),
                "rf_total_w": float(best_schedule.get("schedule_p_out_total_avg_frame_w", float("nan"))),
            }
        ]
    )


def _build_example_allocation_table(example_result) -> pd.DataFrame:
    """Return the selected per-user schedule rows for the illustrative day bin."""

    if example_result.best_schedule is None:
        return pd.DataFrame(
            columns=[
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
        )

    return pd.DataFrame(example_result.best_schedule["rows"]).sort_values("user_id").reset_index(drop=True)


__all__ = [
    "PROJECT_ROOT",
    "build_day_simulation_artifacts",
]
