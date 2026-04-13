from __future__ import annotations

"""Lean day-result flattening for Notebook 5.

This module keeps only the notebook support that is still on the active path:
1. load one day-run JSON export,
2. flatten it into scenario/bin/allocation tables,
3. derive the PA-choice slice used by the discussion notebook.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .day_cycle import bin_index_to_clock


def load_day_run_json(path: Path | str) -> dict[str, object]:
    """Load one day-run export document from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_day_run(day_run: dict[str, object], *, scenario_label: str) -> dict[str, object]:
    """Flatten one day-run export into the lean notebook tables."""

    run = dict(day_run.get("run", {}))
    pa_label_map = _build_pa_label_map(day_run.get("pa_lookup", []))
    bin_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []

    for bin_document in day_run.get("bins", []):
        bin_rows.append(
            _build_bin_row(
                bin_document,
                scenario_label=str(scenario_label),
            )
        )
        allocation_rows.extend(
            _build_allocation_rows(
                bin_document,
                scenario_label=str(scenario_label),
                pa_label_map=pa_label_map,
            )
        )

    return {
        "run": run,
        "pa_label_map": pa_label_map,
        "bin_table": pd.DataFrame(bin_rows).sort_values("bin_index").reset_index(drop=True),
        "allocation_table": (
            pd.DataFrame(allocation_rows)
            .sort_values(["bin_index", "user_id"])
            .reset_index(drop=True)
        ),
    }


def build_day_results_artifacts(scenario_files: dict[str, Path | str]) -> dict[str, object]:
    """Load and flatten the compared day-run exports for Notebook 5."""

    scenario_runs: dict[str, dict[str, object]] = {}
    bin_tables: list[pd.DataFrame] = []
    allocation_tables: list[pd.DataFrame] = []
    run_overview_rows: list[dict[str, object]] = []
    bin_duration_s = float("nan")

    for scenario_label, path in scenario_files.items():
        flattened_run = flatten_day_run(
            load_day_run_json(path),
            scenario_label=str(scenario_label),
        )
        scenario_runs[str(scenario_label)] = flattened_run
        bin_table = flattened_run["bin_table"]
        allocation_table = flattened_run["allocation_table"]
        bin_tables.append(bin_table)
        allocation_tables.append(allocation_table)

        scenario_bin_duration_s = float(flattened_run["run"].get("bin_duration_s", float("nan")))
        if np.isnan(bin_duration_s) and np.isfinite(scenario_bin_duration_s):
            bin_duration_s = scenario_bin_duration_s
        run_overview_rows.append(
            _build_run_overview_row(
                scenario_label=str(scenario_label),
                run=flattened_run["run"],
                bin_table=bin_table,
                bin_duration_s=scenario_bin_duration_s,
            )
        )

    return {
        "scenario_runs": scenario_runs,
        "bin_table_all": pd.concat(bin_tables, ignore_index=True),
        "allocation_table_all": pd.concat(allocation_tables, ignore_index=True),
        "run_overview_table": pd.DataFrame(run_overview_rows),
        "bin_duration_s": float(bin_duration_s),
    }


def build_scenario_pa_choice_table(
    allocation_table_all: pd.DataFrame,
    bin_table_all: pd.DataFrame,
    *,
    scenario_label: str,
    pa_label_map: dict[int, str],
) -> pd.DataFrame:
    """Classify each solved bin by the PA families selected in the schedule."""

    scenario_bins = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(str(scenario_label))]
        .copy()
        .sort_values("bin_index")
        .reset_index(drop=True)
    )
    scenario_allocations = allocation_table_all.loc[
        allocation_table_all["scenario_label"].eq(str(scenario_label))
    ].copy()
    rows = []

    for bin_row in scenario_bins.itertuples(index=False):
        bin_allocations = scenario_allocations.loc[
            scenario_allocations["bin_index"].eq(int(bin_row.bin_index))
        ]
        pa_ids = sorted(bin_allocations["pa_id"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "scenario_label": str(scenario_label),
                "bin_index": int(bin_row.bin_index),
                "clock_label": str(bin_row.clock_label),
                "status": str(bin_row.status),
                "requested_rate_mbps": float(bin_row.requested_rate_mbps),
                "requested_rate_300m_to_499m_mbps": float(bin_row.requested_rate_300m_to_499m_mbps),
                "requested_rate_500m_plus_mbps": float(bin_row.requested_rate_500m_plus_mbps),
                "requested_rate_300m_plus_mbps": float(bin_row.requested_rate_300m_plus_mbps),
                "max_requested_rate_500m_plus_mbps": float(bin_row.max_requested_rate_500m_plus_mbps),
                "dc_total_w": float(bin_row.dc_total_w),
                "used_slots": float(bin_row.used_slots),
                "unused_slots": float(bin_row.unused_slots),
                "pa_ids": tuple(pa_ids),
                "pa_choice_label": _resolve_pa_choice_label(
                    status=str(bin_row.status),
                    pa_ids=pa_ids,
                    pa_label_map=pa_label_map,
                ),
            }
        )

    return pd.DataFrame(rows)


def filter_pa_choice_table_to_500m_user_bins(pa_choice_table: pd.DataFrame) -> pd.DataFrame:
    """Keep only bins that carry at least one user at or beyond 500 m."""

    return (
        pa_choice_table.loc[pa_choice_table["requested_rate_500m_plus_mbps"].gt(0.0)]
        .copy()
        .reset_index(drop=True)
    )


def _build_pa_label_map(pa_lookup_rows: list[dict[str, object]]) -> dict[int, str]:
    return {
        int(row["pa_id"]): str(row.get("pa_label", f"PA {int(row['pa_id'])}"))
        for row in pa_lookup_rows
        if row.get("pa_id") is not None
    }


def _build_bin_row(
    bin_document: dict[str, object],
    *,
    scenario_label: str,
) -> dict[str, object]:
    demand = dict(bin_document.get("demand", {}) or {})
    schedule = dict(bin_document.get("schedule", {}) or {})
    power = dict(schedule.get("power_w", {}) or {})
    users = list(demand.get("users", []) or [])
    requested_rate_bps = float(demand.get("requested_rate_sum_bps", 0.0))
    requested_rate_300m_to_499m_bps = _sum_requested_rate_for_distance_band(
        users,
        min_distance_m=300.0,
        max_distance_m=500.0,
    )
    requested_rate_500m_plus_bps = _sum_requested_rate_for_distance_band(
        users,
        min_distance_m=500.0,
        max_distance_m=None,
    )
    requested_rate_300m_plus_bps = _sum_requested_rate_for_distance_band(
        users,
        min_distance_m=300.0,
        max_distance_m=None,
    )
    max_requested_rate_500m_plus_bps = _max_requested_rate_for_distance_band(
        users,
        min_distance_m=500.0,
        max_distance_m=None,
    )
    delivered_rate_bps = _float_or_nan(schedule.get("delivered_rate_sum_bps"))
    used_slots = _float_or_nan(schedule.get("slot_total"))
    unused_slots = _float_or_nan(schedule.get("unused_slots"))
    frame_slots = used_slots + unused_slots if np.isfinite(used_slots) and np.isfinite(unused_slots) else float("nan")

    return {
        "scenario_label": str(scenario_label),
        "bin_index": int(bin_document.get("bin_index", 0)),
        "clock_label": bin_index_to_clock(int(bin_document.get("bin_index", 0))),
        "status": str(bin_document.get("status", "unknown")),
        "user_count": int(demand.get("user_count", 0)),
        "requested_rate_bps": float(requested_rate_bps),
        "requested_rate_mbps": float(requested_rate_bps) / 1e6,
        "requested_rate_300m_to_499m_bps": float(requested_rate_300m_to_499m_bps),
        "requested_rate_300m_to_499m_mbps": float(requested_rate_300m_to_499m_bps) / 1e6,
        "requested_rate_500m_plus_bps": float(requested_rate_500m_plus_bps),
        "requested_rate_500m_plus_mbps": float(requested_rate_500m_plus_bps) / 1e6,
        "requested_rate_300m_plus_bps": float(requested_rate_300m_plus_bps),
        "requested_rate_300m_plus_mbps": float(requested_rate_300m_plus_bps) / 1e6,
        "max_requested_rate_500m_plus_bps": float(max_requested_rate_500m_plus_bps),
        "max_requested_rate_500m_plus_mbps": float(max_requested_rate_500m_plus_bps) / 1e6,
        "delivered_rate_bps": float(delivered_rate_bps),
        "delivered_rate_mbps": float(delivered_rate_bps) / 1e6,
        "used_slots": float(used_slots),
        "unused_slots": float(unused_slots),
        "frame_slots": float(frame_slots),
        "dc_total_w": _float_or_nan(power.get("dc_total")),
    }


def _build_allocation_rows(
    bin_document: dict[str, object],
    *,
    scenario_label: str,
    pa_label_map: dict[int, str],
) -> list[dict[str, object]]:
    schedule = dict(bin_document.get("schedule", {}) or {})
    rows = []
    for allocation in schedule.get("selected_allocations", []) or []:
        pa_id = int(allocation.get("pa_id", -1))
        rows.append(
            {
                "scenario_label": str(scenario_label),
                "bin_index": int(bin_document.get("bin_index", 0)),
                "user_id": int(allocation.get("user_id", 0)),
                "pa_id": int(pa_id),
                "pa_label": str(pa_label_map.get(pa_id, f"PA {pa_id}")),
                "n_slots": int(allocation.get("n_slots", 0)),
                "p_dc_avg_frame_w": _float_or_nan(allocation.get("p_dc_avg_frame_w")),
            }
        )
    return rows


def _build_run_overview_row(
    *,
    scenario_label: str,
    run: dict[str, object],
    bin_table: pd.DataFrame,
    bin_duration_s: float,
) -> dict[str, object]:
    solved_bins = bin_table.loc[bin_table["status"].eq("solved")]
    day_energy_wh = float(
        np.nansum(bin_table["dc_total_w"].to_numpy(dtype=float) * float(bin_duration_s) / 3600.0)
    )
    return {
        "Scenario": str(scenario_label),
        "Switch policy": str(run.get("switch_policy", "unknown")),
        "Load curve": Path(str(run.get("load_curve_csv", "unknown"))).name,
        "Quarter-hour bins": int(run.get("day_bin_count", len(bin_table))),
        "Solved bins": int(len(solved_bins)),
        "Infeasible bins": int(bin_table["status"].ne("solved").sum()),
        "Mean total power (W)": float(solved_bins["dc_total_w"].mean()),
        "Peak total power (W)": float(solved_bins["dc_total_w"].max()),
        "Day energy (Wh)": float(day_energy_wh),
    }


def _resolve_pa_choice_label(
    *,
    status: str,
    pa_ids: list[int],
    pa_label_map: dict[int, str],
) -> str:
    if status != "solved":
        return "Infeasible"
    if len(pa_ids) == 1:
        return str(pa_label_map.get(int(pa_ids[0]), f"PA {int(pa_ids[0])}"))
    if len(pa_ids) > 1:
        return "Mixed PA use"
    return "Infeasible"


def _sum_requested_rate_for_distance_band(
    users: list[dict[str, object]],
    *,
    min_distance_m: float,
    max_distance_m: float | None,
) -> float:
    return float(
        sum(
            float(user.get("required_rate_bps", 0.0))
            for user in users
            if _distance_in_band(
                float(user.get("distance_m", float("nan"))),
                min_distance_m=min_distance_m,
                max_distance_m=max_distance_m,
            )
        )
    )


def _max_requested_rate_for_distance_band(
    users: list[dict[str, object]],
    *,
    min_distance_m: float,
    max_distance_m: float | None,
) -> float:
    return float(
        max(
            (
                float(user.get("required_rate_bps", 0.0))
                for user in users
                if _distance_in_band(
                    float(user.get("distance_m", float("nan"))),
                    min_distance_m=min_distance_m,
                    max_distance_m=max_distance_m,
                )
            ),
            default=0.0,
        )
    )


def _distance_in_band(
    distance_m: float,
    *,
    min_distance_m: float,
    max_distance_m: float | None,
) -> bool:
    if not np.isfinite(distance_m) or distance_m < float(min_distance_m):
        return False
    if max_distance_m is None:
        return True
    return distance_m < float(max_distance_m)


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


__all__ = [
    "build_day_results_artifacts",
    "build_scenario_pa_choice_table",
    "filter_pa_choice_table_to_500m_user_bins",
    "flatten_day_run",
    "load_day_run_json",
]
