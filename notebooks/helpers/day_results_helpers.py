from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from helpers.DayCycleSimulationHelpers import bin_index_to_clock, style_dataframe


DAY_RESULT_BIN_COLUMNS = [
    "scenario_label",
    "bin_index",
    "clock_label",
    "status",
    "outcome_code",
    "user_count",
    "requested_rate_bps",
    "requested_rate_mbps",
    "requested_rate_300m_to_499m_bps",
    "requested_rate_300m_to_499m_mbps",
    "requested_rate_500m_plus_bps",
    "requested_rate_500m_plus_mbps",
    "requested_rate_300m_plus_bps",
    "requested_rate_300m_plus_mbps",
    "max_requested_rate_500m_plus_bps",
    "max_requested_rate_500m_plus_mbps",
    "delivered_rate_bps",
    "delivered_rate_mbps",
    "served_fraction",
    "used_slots",
    "unused_slots",
    "window_slots",
    "dc_total_w",
    "dc_active_w",
    "dc_inactive_w",
    "rf_total_w",
    "single_user_time_s",
    "joint_time_s",
    "total_time_s",
]


DAY_RESULT_ALLOCATION_COLUMNS = [
    "scenario_label",
    "bin_index",
    "clock_label",
    "user_id",
    "pa_id",
    "pa_label",
    "bandwidth_hz",
    "bandwidth_mhz",
    "n_prb",
    "layers",
    "mcs",
    "n_slots",
    "delivered_rate_bps",
    "delivered_rate_mbps",
    "p_dc_avg_frame_w",
    "p_out_avg_frame_w",
]


DAY_RESULT_DEMAND_USER_COLUMNS = [
    "scenario_label",
    "bin_index",
    "clock_label",
    "user_id",
    "distance_m",
    "distance_class",
    "required_rate_bps",
    "required_rate_mbps",
]


BASE_USER_COLORS = ["#c75d2c", "#0b7a75", "#4e79a7", "#8e6c8a", "#7f7f7f", "#59a14f"]
BASE_PA_COLORS = ["#c75d2c", "#0b7a75", "#4e79a7", "#8e6c8a"]
SPECIAL_PA_CHOICE_COLORS = {
    "Mixed PA use": "#8e6c8a",
    "Infeasible": "#7f7f7f",
}


def load_day_run_json(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _distance_class_label(distance_m: float) -> str:
    if distance_m >= 500.0:
        return "Far (>=500 m)"
    if distance_m >= 300.0:
        return "Mid (300-499 m)"
    return "Near (<300 m)"


def flatten_day_run(day_run: dict, *, scenario_label: str) -> dict[str, object]:
    run = day_run.get("run", {})
    pa_lookup_table = pd.DataFrame(day_run.get("pa_lookup", []))
    pa_label_map = {
        int(row.pa_id): str(getattr(row, "pa_label", f"PA {int(row.pa_id)}"))
        for row in pa_lookup_table.itertuples(index=False)
        if getattr(row, "pa_id", None) is not None
    }

    bin_rows = []
    allocation_rows = []
    demand_user_rows = []

    for bin_result in day_run.get("bins", []):
        bin_index = int(bin_result.get("bin_index", 0))
        demand = bin_result.get("demand", {}) or {}
        schedule = bin_result.get("schedule", {}) or {}
        power = schedule.get("power_w", {}) or {}
        timings = bin_result.get("timings_s", {}) or {}
        users = demand.get("users", []) or []

        requested_rate_bps = _safe_float(demand.get("requested_rate_sum_bps"))
        requested_rate_300m_to_499m_bps = float(
            sum(
                _safe_float(user.get("required_rate_bps"))
                for user in users
                if 300.0 <= _safe_float(user.get("distance_m")) < 500.0
            )
        )
        requested_rate_500m_plus_bps = float(
            sum(
                _safe_float(user.get("required_rate_bps"))
                for user in users
                if _safe_float(user.get("distance_m")) >= 500.0
            )
        )
        requested_rate_300m_plus_bps = float(
            sum(
                _safe_float(user.get("required_rate_bps"))
                for user in users
                if _safe_float(user.get("distance_m")) >= 300.0
            )
        )
        max_requested_rate_500m_plus_bps = float(
            max(
                (
                    _safe_float(user.get("required_rate_bps"))
                    for user in users
                    if _safe_float(user.get("distance_m")) >= 500.0
                ),
                default=0.0,
            )
        )
        delivered_rate_bps = _safe_float(schedule.get("delivered_rate_sum_bps"))
        used_slots = _safe_float(schedule.get("slot_total"))
        unused_slots = _safe_float(schedule.get("unused_slots"))
        window_slots = used_slots + unused_slots if np.isfinite(used_slots) and np.isfinite(unused_slots) else float("nan")
        served_fraction = (
            delivered_rate_bps / requested_rate_bps
            if requested_rate_bps > 0.0 and np.isfinite(delivered_rate_bps)
            else float("nan")
        )

        bin_rows.append(
            {
                "scenario_label": str(scenario_label),
                "bin_index": bin_index,
                "clock_label": bin_index_to_clock(bin_index),
                "status": str(bin_result.get("status", "unknown")),
                "outcome_code": str(bin_result.get("outcome_code", "unknown")),
                "user_count": int(demand.get("user_count", 0)),
                "requested_rate_bps": requested_rate_bps,
                "requested_rate_mbps": requested_rate_bps / 1e6,
                "requested_rate_300m_to_499m_bps": requested_rate_300m_to_499m_bps,
                "requested_rate_300m_to_499m_mbps": requested_rate_300m_to_499m_bps / 1e6,
                "requested_rate_500m_plus_bps": requested_rate_500m_plus_bps,
                "requested_rate_500m_plus_mbps": requested_rate_500m_plus_bps / 1e6,
                "requested_rate_300m_plus_bps": requested_rate_300m_plus_bps,
                "requested_rate_300m_plus_mbps": requested_rate_300m_plus_bps / 1e6,
                "max_requested_rate_500m_plus_bps": max_requested_rate_500m_plus_bps,
                "max_requested_rate_500m_plus_mbps": max_requested_rate_500m_plus_bps / 1e6,
                "delivered_rate_bps": delivered_rate_bps,
                "delivered_rate_mbps": delivered_rate_bps / 1e6,
                "served_fraction": served_fraction,
                "used_slots": used_slots,
                "unused_slots": unused_slots,
                "window_slots": window_slots,
                "dc_total_w": _safe_float(power.get("dc_total")),
                "dc_active_w": _safe_float(power.get("dc_active")),
                "dc_inactive_w": _safe_float(power.get("dc_inactive")),
                "rf_total_w": _safe_float(power.get("rf_total")),
                "single_user_time_s": _safe_float(timings.get("single_user")),
                "joint_time_s": _safe_float(timings.get("joint")),
                "total_time_s": _safe_float(timings.get("total")),
            }
        )

        for user in users:
            distance_m = _safe_float(user.get("distance_m"))
            required_rate_bps = _safe_float(user.get("required_rate_bps"))
            demand_user_rows.append(
                {
                    "scenario_label": str(scenario_label),
                    "bin_index": bin_index,
                    "clock_label": bin_index_to_clock(bin_index),
                    "user_id": int(user.get("user_id", 0)),
                    "distance_m": distance_m,
                    "distance_class": _distance_class_label(distance_m),
                    "required_rate_bps": required_rate_bps,
                    "required_rate_mbps": required_rate_bps / 1e6,
                }
            )

        for allocation in schedule.get("selected_allocations", []) or []:
            pa_id = _safe_int(allocation.get("pa_id"))
            bandwidth_hz = _safe_float(allocation.get("bandwidth_hz"))
            delivered_rate_alloc_bps = _safe_float(allocation.get("delivered_rate_bps"))
            allocation_rows.append(
                {
                    "scenario_label": str(scenario_label),
                    "bin_index": bin_index,
                    "clock_label": bin_index_to_clock(bin_index),
                    "user_id": int(allocation.get("user_id", 0)),
                    "pa_id": pa_id,
                    "pa_label": pa_label_map.get(pa_id, f"PA {pa_id}" if pa_id is not None else "Unknown PA"),
                    "bandwidth_hz": bandwidth_hz,
                    "bandwidth_mhz": bandwidth_hz / 1e6,
                    "n_prb": int(allocation.get("n_prb", 0)),
                    "layers": int(allocation.get("layers", 0)),
                    "mcs": int(allocation.get("mcs", 0)),
                    "n_slots": int(allocation.get("n_slots", 0)),
                    "delivered_rate_bps": delivered_rate_alloc_bps,
                    "delivered_rate_mbps": delivered_rate_alloc_bps / 1e6,
                    "p_dc_avg_frame_w": _safe_float(allocation.get("p_dc_avg_frame_w")),
                    "p_out_avg_frame_w": _safe_float(allocation.get("p_out_avg_frame_w")),
                }
            )

    bin_table = pd.DataFrame(bin_rows, columns=DAY_RESULT_BIN_COLUMNS).sort_values("bin_index").reset_index(drop=True)
    demand_user_table = pd.DataFrame(demand_user_rows, columns=DAY_RESULT_DEMAND_USER_COLUMNS).sort_values(
        ["bin_index", "user_id"]
    ).reset_index(drop=True)
    allocation_table = pd.DataFrame(allocation_rows, columns=DAY_RESULT_ALLOCATION_COLUMNS).sort_values(
        ["bin_index", "user_id"]
    ).reset_index(drop=True)

    return {
        "run": run,
        "pa_lookup_table": pa_lookup_table,
        "pa_label_map": pa_label_map,
        "bin_table": bin_table,
        "demand_user_table": demand_user_table,
        "allocation_table": allocation_table,
    }


def build_day_results_artifacts(scenario_files: Mapping[str, Path | str]) -> Dict[str, object]:
    scenario_labels = list(scenario_files.keys())
    flattened_runs: dict[str, dict[str, object]] = {}
    run_overview_rows = []
    bin_tables = []
    demand_user_tables = []
    allocation_tables = []

    bin_duration_s = float("nan")

    for scenario_label, path in scenario_files.items():
        day_run = load_day_run_json(path)
        flattened = flatten_day_run(day_run, scenario_label=scenario_label)
        flattened_runs[scenario_label] = flattened

        run = flattened["run"]
        bin_table = flattened["bin_table"]
        demand_user_table = flattened["demand_user_table"]
        allocation_table = flattened["allocation_table"]
        bin_tables.append(bin_table)
        demand_user_tables.append(demand_user_table)
        allocation_tables.append(allocation_table)

        scenario_bin_duration_s = _safe_float(run.get("bin_duration_s"))
        if np.isnan(bin_duration_s) and np.isfinite(scenario_bin_duration_s):
            bin_duration_s = scenario_bin_duration_s

        solved_mask = bin_table["status"].eq("solved")
        infeasible_mask = bin_table["status"].ne("solved")
        energy_wh = float(np.nansum(bin_table["dc_total_w"] * scenario_bin_duration_s / 3600.0)) if np.isfinite(scenario_bin_duration_s) else float("nan")

        run_overview_rows.append(
            {
                "Scenario": str(scenario_label),
                "Switch policy": str(run.get("switch_policy", "unknown")),
                "Load curve": Path(str(run.get("load_curve_csv", "unknown"))).name,
                "Quarter-hour bins": int(run.get("day_bin_count", len(bin_table))),
                "Solved bins": int(solved_mask.sum()),
                "Infeasible bins": int(infeasible_mask.sum()),
                "Mean total power (W)": float(bin_table.loc[solved_mask, "dc_total_w"].mean()),
                "Peak total power (W)": float(bin_table.loc[solved_mask, "dc_total_w"].max()),
                "Day energy (Wh)": energy_wh,
            }
        )

    bin_table_all = pd.concat(bin_tables, ignore_index=True) if bin_tables else pd.DataFrame(columns=DAY_RESULT_BIN_COLUMNS)
    demand_user_table_all = (
        pd.concat(demand_user_tables, ignore_index=True)
        if demand_user_tables
        else pd.DataFrame(columns=DAY_RESULT_DEMAND_USER_COLUMNS)
    )
    allocation_table_all = (
        pd.concat(allocation_tables, ignore_index=True)
        if allocation_tables
        else pd.DataFrame(columns=DAY_RESULT_ALLOCATION_COLUMNS)
    )
    run_overview_table = pd.DataFrame(run_overview_rows)

    return {
        "scenario_labels": scenario_labels,
        "scenario_runs": flattened_runs,
        "bin_table_all": bin_table_all,
        "demand_user_table_all": demand_user_table_all,
        "allocation_table_all": allocation_table_all,
        "run_overview_table": run_overview_table,
        "bin_duration_s": bin_duration_s,
    }


def build_reference_demand_table(bin_table_all: pd.DataFrame, *, reference_scenario: str) -> pd.DataFrame:
    return (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(reference_scenario), [
            "bin_index",
            "clock_label",
            "user_count",
            "requested_rate_bps",
            "requested_rate_mbps",
        ]]
        .sort_values("bin_index")
        .reset_index(drop=True)
    )


def _build_pa_choice_color_map(pa_label_map: Mapping[int, str]) -> dict[str, str]:
    color_map = {
        str(pa_label): BASE_PA_COLORS[idx % len(BASE_PA_COLORS)]
        for idx, (_pa_id, pa_label) in enumerate(sorted(pa_label_map.items()))
    }
    color_map.update(SPECIAL_PA_CHOICE_COLORS)
    return color_map


def build_scenario_pa_choice_table(
    allocation_table_all: pd.DataFrame,
    bin_table_all: pd.DataFrame,
    *,
    scenario_label: str,
    pa_label_map: Mapping[int, str],
) -> pd.DataFrame:
    scenario_bins = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)].copy()
        .sort_values("bin_index")
        .reset_index(drop=True)
    )
    scenario_allocations = allocation_table_all.loc[
        allocation_table_all["scenario_label"].eq(scenario_label)
    ].copy()

    choice_rows = []
    for row in scenario_bins.itertuples(index=False):
        bin_allocations = scenario_allocations.loc[
            scenario_allocations["bin_index"].eq(int(row.bin_index))
        ].copy()
        pa_ids = sorted(
            int(pa_id)
            for pa_id in bin_allocations["pa_id"].dropna().astype(int).unique().tolist()
        )
        if str(row.status) != "solved":
            pa_choice_label = "Infeasible"
        elif len(pa_ids) == 1:
            pa_choice_label = str(pa_label_map.get(pa_ids[0], f"PA {pa_ids[0]}"))
        elif len(pa_ids) > 1:
            pa_choice_label = "Mixed PA use"
        else:
            pa_choice_label = "Infeasible"

        choice_rows.append(
            {
                "scenario_label": str(scenario_label),
                "bin_index": int(row.bin_index),
                "clock_label": str(row.clock_label),
                "status": str(row.status),
                "requested_rate_mbps": float(row.requested_rate_mbps),
                "requested_rate_300m_to_499m_mbps": float(row.requested_rate_300m_to_499m_mbps),
                "requested_rate_500m_plus_mbps": float(row.requested_rate_500m_plus_mbps),
                "requested_rate_300m_plus_mbps": float(row.requested_rate_300m_plus_mbps),
                "max_requested_rate_500m_plus_mbps": float(row.max_requested_rate_500m_plus_mbps),
                "dc_total_w": float(row.dc_total_w),
                "used_slots": float(row.used_slots),
                "unused_slots": float(row.unused_slots),
                "pa_choice_label": pa_choice_label,
                "pa_ids": tuple(pa_ids),
            }
        )

    pa_choice_table = pd.DataFrame(choice_rows)
    pa_choice_table["pa_choice_color"] = pa_choice_table["pa_choice_label"].map(
        _build_pa_choice_color_map(pa_label_map)
    )
    return pa_choice_table


def pick_representative_bins(
    bin_table_all: pd.DataFrame,
    *,
    reference_scenario: str,
    comparison_scenario: str | None = None,
) -> Dict[str, int]:
    reference = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(reference_scenario)]
        .sort_values("bin_index")
        .reset_index(drop=True)
    )
    solved_reference = reference.loc[reference["status"].eq("solved")].copy()
    if solved_reference.empty:
        raise ValueError("The reference scenario does not contain any solved bins.")

    quiet_row = solved_reference.loc[solved_reference["requested_rate_bps"].idxmin()]
    peak_row = solved_reference.loc[solved_reference["requested_rate_bps"].idxmax()]

    if comparison_scenario is None:
        comparison_scenario = reference_scenario

    comparison = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(comparison_scenario), ["bin_index", "dc_total_w"]]
        .rename(columns={"dc_total_w": "dc_total_w_compare"})
    )
    delta_table = reference.merge(comparison, on="bin_index", how="inner")
    delta_table["dc_total_delta_abs_w"] = (
        delta_table["dc_total_w"] - delta_table["dc_total_w_compare"]
    ).abs()
    delta_table = delta_table.loc[delta_table["status"].eq("solved")]

    if delta_table.empty:
        divergence_row = peak_row
    else:
        divergence_row = delta_table.loc[delta_table["dc_total_delta_abs_w"].idxmax()]

    return {
        "quiet_bin": int(quiet_row["bin_index"]),
        "peak_bin": int(peak_row["bin_index"]),
        "divergence_bin": int(divergence_row["bin_index"]),
    }


def build_highlighted_bin_table(
    bin_table_all: pd.DataFrame,
    *,
    reference_scenario: str,
    highlighted_bins: Mapping[str, int],
) -> pd.DataFrame:
    reference = bin_table_all.loc[bin_table_all["scenario_label"].eq(reference_scenario)].copy()
    label_map = {
        "quiet_bin": "Quiet period",
        "peak_bin": "Peak period",
        "divergence_bin": "Largest scenario gap",
    }
    rows = []
    for role_key, bin_index in highlighted_bins.items():
        row = reference.loc[reference["bin_index"].eq(int(bin_index))].iloc[0]
        rows.append(
            {
                "Role": label_map.get(role_key, str(role_key)),
                "Bin index": int(bin_index),
                "Clock": str(row["clock_label"]),
                "Users": int(row["user_count"]),
                "Requested rate (Mbps)": float(row["requested_rate_mbps"]),
                "Used slots": float(row["used_slots"]),
                "Unused slots": float(row["unused_slots"]),
            }
        )
    return pd.DataFrame(rows)


def build_bin_comparison_table(
    bin_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_index: int,
) -> pd.DataFrame:
    scenario_order = list(scenario_order)
    comparison_table = (
        bin_table_all.loc[bin_table_all["bin_index"].eq(int(bin_index))]
        .set_index("scenario_label")
        .reindex(scenario_order)
        .reset_index()
    )
    return comparison_table.rename(
        columns={
            "scenario_label": "Scenario",
            "status": "Status",
            "outcome_code": "Outcome",
            "user_count": "Users",
            "requested_rate_mbps": "Requested rate (Mbps)",
            "delivered_rate_mbps": "Delivered rate (Mbps)",
            "dc_total_w": "Total power (W)",
            "dc_active_w": "Active power (W)",
            "dc_inactive_w": "Inactive power (W)",
            "used_slots": "Used slots",
            "unused_slots": "Unused slots",
        }
    )[
        [
            "Scenario",
            "Status",
            "Outcome",
            "Users",
            "Requested rate (Mbps)",
            "Delivered rate (Mbps)",
            "Total power (W)",
            "Active power (W)",
            "Inactive power (W)",
            "Used slots",
            "Unused slots",
        ]
    ]


def build_allocation_choice_table(
    allocation_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_index: int,
) -> pd.DataFrame:
    scenario_order = list(scenario_order)
    choice_table = allocation_table_all.loc[allocation_table_all["bin_index"].eq(int(bin_index))].copy()
    choice_table["scenario_label"] = pd.Categorical(choice_table["scenario_label"], categories=scenario_order, ordered=True)
    choice_table = choice_table.sort_values(["scenario_label", "user_id"]).reset_index(drop=True)
    return choice_table.rename(
        columns={
            "scenario_label": "Scenario",
            "user_id": "User id",
            "pa_label": "PA",
            "bandwidth_mhz": "Bandwidth (MHz)",
            "n_prb": "PRBs",
            "layers": "Layers",
            "mcs": "MCS",
            "n_slots": "Slots",
            "delivered_rate_mbps": "Delivered rate (Mbps)",
            "p_dc_avg_frame_w": "Avg frame DC power (W)",
            "p_out_avg_frame_w": "Avg frame RF power (W)",
        }
    )[
        [
            "Scenario",
            "User id",
            "PA",
            "Bandwidth (MHz)",
            "PRBs",
            "Layers",
            "MCS",
            "Slots",
            "Delivered rate (Mbps)",
            "Avg frame DC power (W)",
            "Avg frame RF power (W)",
        ]
    ]


def build_day_total_summary_table(
    bin_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_duration_s: float,
) -> pd.DataFrame:
    scenario_order = list(scenario_order)
    rows = []
    for scenario_label in scenario_order:
        scenario_bins = bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)].copy()
        solved_mask = scenario_bins["status"].eq("solved")
        day_energy_wh = float(np.nansum(scenario_bins["dc_total_w"] * float(bin_duration_s) / 3600.0))
        rows.append(
            {
                "Scenario": str(scenario_label),
                "Solved bins": int(solved_mask.sum()),
                "Infeasible bins": int((~solved_mask).sum()),
                "Mean total power (W)": float(scenario_bins.loc[solved_mask, "dc_total_w"].mean()),
                "Peak total power (W)": float(scenario_bins.loc[solved_mask, "dc_total_w"].max()),
                "Day energy (Wh)": day_energy_wh,
                "Mean unused slots": float(scenario_bins.loc[solved_mask, "unused_slots"].mean()),
                "Peak requested rate (Mbps)": float(scenario_bins["requested_rate_mbps"].max()),
            }
        )
    return pd.DataFrame(rows)


def filter_pa_choice_table_to_500m_user_bins(pa_choice_table: pd.DataFrame) -> pd.DataFrame:
    return (
        pa_choice_table.loc[pa_choice_table["requested_rate_500m_plus_mbps"].gt(0.0)]
        .copy()
        .sort_values("bin_index")
        .reset_index(drop=True)
    )


def build_filtered_500m_summary_table(filtered_pa_choice_table: pd.DataFrame) -> pd.DataFrame:
    if filtered_pa_choice_table.empty:
        return pd.DataFrame(
            columns=[
                "PA choice",
                "Bins",
                "Mean total load (Mbps)",
                "Mean load from 500 m+ users (Mbps)",
                "Mean load from 300-499 m users (Mbps)",
                "Mean max 500 m user (Mbps)",
            ]
        )

    choice_order = []
    for value in filtered_pa_choice_table["pa_choice_label"].tolist():
        if value not in choice_order:
            choice_order.append(value)

    summary = (
        filtered_pa_choice_table.groupby("pa_choice_label", dropna=False)
        .agg(
            bins=("bin_index", "count"),
            mean_total_load_mbps=("requested_rate_mbps", "mean"),
            mean_500m_plus_load_mbps=("requested_rate_500m_plus_mbps", "mean"),
            mean_300m_to_499m_load_mbps=("requested_rate_300m_to_499m_mbps", "mean"),
            mean_max_500m_user_mbps=("max_requested_rate_500m_plus_mbps", "mean"),
        )
        .reset_index()
    )
    summary["pa_choice_label"] = pd.Categorical(
        summary["pa_choice_label"],
        categories=choice_order,
        ordered=True,
    )
    summary = summary.sort_values("pa_choice_label").reset_index(drop=True)
    return summary.rename(
        columns={
            "pa_choice_label": "PA choice",
            "bins": "Bins",
            "mean_total_load_mbps": "Mean total load (Mbps)",
            "mean_500m_plus_load_mbps": "Mean load from 500 m+ users (Mbps)",
            "mean_300m_to_499m_load_mbps": "Mean load from 300-499 m users (Mbps)",
            "mean_max_500m_user_mbps": "Mean max 500 m user (Mbps)",
        }
    )


def build_infeasibility_boundary_table(filtered_pa_choice_table: pd.DataFrame) -> pd.DataFrame:
    feasible = filtered_pa_choice_table.loc[filtered_pa_choice_table["status"].eq("solved")].copy()
    infeasible = filtered_pa_choice_table.loc[filtered_pa_choice_table["status"].ne("solved")].copy()
    feasible_max = float(feasible["requested_rate_500m_plus_mbps"].max()) if not feasible.empty else float("nan")
    infeasible_min = float(infeasible["requested_rate_500m_plus_mbps"].min()) if not infeasible.empty else float("nan")
    return pd.DataFrame(
        [
            {
                "Feasible bins": int(len(feasible)),
                "Infeasible bins": int(len(infeasible)),
                "Feasible median load from >= 500 m users (Mbps)": float(feasible["requested_rate_500m_plus_mbps"].median())
                if not feasible.empty
                else float("nan"),
                "Infeasible median load from >= 500 m users (Mbps)": float(infeasible["requested_rate_500m_plus_mbps"].median())
                if not infeasible.empty
                else float("nan"),
                "Feasible max load from >= 500 m users (Mbps)": feasible_max,
                "Infeasible min load from >= 500 m users (Mbps)": infeasible_min,
                "Separation gap (Mbps)": infeasible_min - feasible_max
                if np.isfinite(feasible_max) and np.isfinite(infeasible_min)
                else float("nan"),
            }
        ]
    )


def _solved_500m_pa_subset(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    return (
        filtered_pa_choice_table.loc[
            filtered_pa_choice_table["status"].eq("solved")
            & filtered_pa_choice_table["pa_choice_label"].isin(class_labels)
        ]
        .copy()
        .reset_index(drop=True)
    )


def build_binary_pa_feature_summary_table(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    feature_columns = [
        "requested_rate_500m_plus_mbps",
        "requested_rate_300m_to_499m_mbps",
    ]
    feature_labels = {
        "requested_rate_500m_plus_mbps": "Load from >= 500 m users (Mbps)",
        "requested_rate_300m_to_499m_mbps": "Load from 300-499 m users (Mbps)",
    }
    rows = []
    for feature in feature_columns:
        row = {"Feature": feature_labels[feature]}
        for class_label in class_labels:
            subset = solved_subset.loc[solved_subset["pa_choice_label"].eq(class_label), feature]
            row[f"{class_label} bins"] = int(len(subset))
            row[f"{class_label} mean"] = float(subset.mean())
            row[f"{class_label} median"] = float(subset.median())
            row[f"{class_label} std"] = float(subset.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_binary_pa_univariate_test_table(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    from scipy import stats

    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    feature_columns = [
        "requested_rate_500m_plus_mbps",
        "requested_rate_300m_to_499m_mbps",
    ]
    feature_labels = {
        "requested_rate_500m_plus_mbps": "Load from >= 500 m users (Mbps)",
        "requested_rate_300m_to_499m_mbps": "Load from 300-499 m users (Mbps)",
    }
    rows = []
    for feature in feature_columns:
        left = solved_subset.loc[solved_subset["pa_choice_label"].eq(class_labels[0]), feature].to_numpy(dtype=float)
        right = solved_subset.loc[solved_subset["pa_choice_label"].eq(class_labels[1]), feature].to_numpy(dtype=float)
        statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
        rows.append(
            {
                "Feature": feature_labels[feature],
                "Mann-Whitney U": float(statistic),
                "Two-sided p-value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def _lda_binary_parameters(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_labels: tuple[str, str],
) -> tuple[np.ndarray, float, Mapping[str, np.ndarray], np.ndarray]:
    p = X.shape[1]
    means = {
        class_label: X[y == class_label].mean(axis=0)
        for class_label in class_labels
    }
    pooled_covariance = np.zeros((p, p), dtype=float)
    for class_label in class_labels:
        Xi = X[y == class_label]
        pooled_covariance += (len(Xi) - 1) * np.cov(Xi, rowvar=False, ddof=1)
    pooled_covariance /= max(len(X) - len(class_labels), 1)
    ridge = max(float(np.trace(pooled_covariance)) * 1e-9, 1e-9)
    pooled_covariance = pooled_covariance + ridge * np.eye(p)
    inverse_covariance = np.linalg.pinv(pooled_covariance)
    mu_left = means[class_labels[0]]
    mu_right = means[class_labels[1]]
    prior_ratio = np.log(
        max((y == class_labels[1]).mean(), 1e-12) / max((y == class_labels[0]).mean(), 1e-12)
    )
    coefficients = inverse_covariance @ (mu_right - mu_left)
    intercept = -0.5 * (mu_right @ inverse_covariance @ mu_right - mu_left @ inverse_covariance @ mu_left) + prior_ratio
    return coefficients, float(intercept), means, pooled_covariance


def _lda_binary_predict(
    X: np.ndarray,
    *,
    coefficients: np.ndarray,
    intercept: float,
    class_labels: tuple[str, str],
) -> np.ndarray:
    scores = X @ coefficients + intercept
    return np.where(scores > 0.0, class_labels[1], class_labels[0])


def _best_binary_threshold(
    feature_values: np.ndarray,
    labels: np.ndarray,
    *,
    class_labels: tuple[str, str],
) -> float:
    unique_values = np.sort(np.unique(feature_values))
    if len(unique_values) == 1:
        return float(unique_values[0])
    candidate_thresholds = [float(unique_values[0] - 1e-9)]
    candidate_thresholds.extend(
        float((unique_values[idx] + unique_values[idx + 1]) / 2.0)
        for idx in range(len(unique_values) - 1)
    )
    candidate_thresholds.append(float(unique_values[-1] + 1e-9))
    best_threshold = candidate_thresholds[0]
    best_accuracy = -1.0
    for threshold in candidate_thresholds:
        predictions = np.where(feature_values >= threshold, class_labels[1], class_labels[0])
        accuracy = float((predictions == labels).mean())
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    return float(best_threshold)


def _binary_confusion_counts(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    class_labels: tuple[str, str],
) -> dict[str, int]:
    left, right = class_labels
    return {
        f"{left} -> {left}": int(((actual == left) & (predicted == left)).sum()),
        f"{left} -> {right}": int(((actual == left) & (predicted == right)).sum()),
        f"{right} -> {left}": int(((actual == right) & (predicted == left)).sum()),
        f"{right} -> {right}": int(((actual == right) & (predicted == right)).sum()),
    }


def build_binary_pa_classifier_table(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    labels = solved_subset["pa_choice_label"].to_numpy()

    feature_map = {
        "max_requested_rate_500m_plus_mbps": "Max single-user rate at >= 500 m",
        "requested_rate_500m_plus_mbps": "Total load from >= 500 m users",
    }
    rows = []
    for feature_column, classifier_name in feature_map.items():
        feature_values = solved_subset[feature_column].to_numpy(dtype=float)
        threshold = _best_binary_threshold(feature_values, labels, class_labels=class_labels)
        train_predictions = np.where(feature_values >= threshold, class_labels[1], class_labels[0])
        loo_predictions = []
        for idx in range(len(feature_values)):
            mask = np.ones(len(feature_values), dtype=bool)
            mask[idx] = False
            loo_threshold = _best_binary_threshold(
                feature_values[mask],
                labels[mask],
                class_labels=class_labels,
            )
            loo_predictions.append(
                class_labels[1] if feature_values[idx] >= loo_threshold else class_labels[0]
            )
        loo_predictions = np.asarray(loo_predictions, dtype=object)
        rows.append(
            {
                "Classifier": classifier_name,
                "Decision rule": f"Predict {class_labels[1]} when {feature_column} >= {threshold:.2f}",
                "Training accuracy": float((train_predictions == labels).mean()),
                "LOO accuracy": float((loo_predictions == labels).mean()),
                **_binary_confusion_counts(labels, loo_predictions, class_labels=class_labels),
            }
        )

    feature_columns = [
        "requested_rate_500m_plus_mbps",
        "requested_rate_300m_to_499m_mbps",
    ]
    X = solved_subset[feature_columns].to_numpy(dtype=float)
    coefficients, intercept, _means, _pooled = _lda_binary_parameters(
        X,
        labels,
        class_labels=class_labels,
    )
    train_predictions = _lda_binary_predict(
        X,
        coefficients=coefficients,
        intercept=intercept,
        class_labels=class_labels,
    )
    loo_predictions = []
    for idx in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[idx] = False
        loo_coefficients, loo_intercept, _loo_means, _loo_pooled = _lda_binary_parameters(
            X[mask],
            labels[mask],
            class_labels=class_labels,
        )
        loo_predictions.append(
            _lda_binary_predict(
                X[~mask],
                coefficients=loo_coefficients,
                intercept=loo_intercept,
                class_labels=class_labels,
            )[0]
        )
    loo_predictions = np.asarray(loo_predictions, dtype=object)
    if abs(coefficients[1]) > 1e-12:
        normalized_left_weight = float(coefficients[0] / coefficients[1])
        normalized_right_weight = 1.0
        normalized_threshold = float((-intercept) / coefficients[1])
        decision_rule = (
            f"Predict {class_labels[1]} when "
            f"{normalized_left_weight:.2f} * load_500m_plus + {normalized_right_weight:.2f} * load_300_499m "
            f"> {normalized_threshold:.2f}"
        )
    else:
        decision_rule = (
            f"Predict {class_labels[1]} when "
            f"{coefficients[0]:.3f} * load_500m_plus + {coefficients[1]:.3f} * load_300_499m "
            f"> {(-intercept):.3f}"
        )
    rows.append(
        {
            "Classifier": "Two-feature LDA",
            "Decision rule": decision_rule,
            "Training accuracy": float((train_predictions == labels).mean()),
            "LOO accuracy": float((loo_predictions == labels).mean()),
            **_binary_confusion_counts(labels, loo_predictions, class_labels=class_labels),
        }
    )
    return pd.DataFrame(rows)


def build_binary_pa_multivariate_test_table(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    from scipy import stats

    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    feature_columns = [
        "requested_rate_500m_plus_mbps",
        "requested_rate_300m_to_499m_mbps",
    ]
    X = solved_subset[feature_columns].to_numpy(dtype=float)
    y = solved_subset["pa_choice_label"].to_numpy()
    left = X[y == class_labels[0]]
    right = X[y == class_labels[1]]
    n_left, n_right = len(left), len(right)
    p = X.shape[1]
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    covariance_left = np.cov(left, rowvar=False, ddof=1)
    covariance_right = np.cov(right, rowvar=False, ddof=1)
    pooled_covariance = (
        ((n_left - 1) * covariance_left + (n_right - 1) * covariance_right)
        / max(n_left + n_right - 2, 1)
    )
    ridge = max(float(np.trace(pooled_covariance)) * 1e-9, 1e-9)
    pooled_covariance = pooled_covariance + ridge * np.eye(p)
    mean_difference = mean_left - mean_right
    hotelling_t2 = float((n_left * n_right / (n_left + n_right)) * (mean_difference.T @ np.linalg.pinv(pooled_covariance) @ mean_difference))
    f_statistic = float(((n_left + n_right - p - 1) / (p * max(n_left + n_right - 2, 1))) * hotelling_t2)
    p_value = float(stats.f.sf(f_statistic, p, n_left + n_right - p - 1))

    coefficients, intercept, _means, _pooled = _lda_binary_parameters(
        X,
        y,
        class_labels=class_labels,
    )
    ratio = float(coefficients[0] / coefficients[1]) if abs(coefficients[1]) > 0.0 else float("inf")
    return pd.DataFrame(
        [
            {
                "4W bins": int(n_left),
                "8W bins": int(n_right),
                "Hotelling T^2": hotelling_t2,
                "F statistic": f_statistic,
                "p-value": p_value,
                "LDA weight on >= 500 m load": float(coefficients[0]),
                "LDA weight on 300-499 m load": float(coefficients[1]),
                "Relative weight >= 500 m / 300-499 m": ratio,
                "LDA threshold": float(-intercept),
            }
        ]
    )


def _binary_pa_observational_subset(
    filtered_pa_choice_table: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    allocation_table_all: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    merge_keys = ["scenario_label", "bin_index"]
    user_keys = ["scenario_label", "bin_index", "user_id"]
    user_subset = demand_user_table_all.merge(
        solved_subset[merge_keys + ["pa_choice_label"]],
        on=merge_keys,
        how="inner",
    )
    allocation_columns = [
        "scenario_label",
        "bin_index",
        "user_id",
        "delivered_rate_bps",
        "delivered_rate_mbps",
        "n_slots",
        "p_dc_avg_frame_w",
    ]
    user_subset = user_subset.merge(
        allocation_table_all[allocation_columns],
        on=user_keys,
        how="left",
    )
    for column in ["delivered_rate_bps", "delivered_rate_mbps", "n_slots", "p_dc_avg_frame_w"]:
        user_subset[column] = user_subset[column].fillna(0.0)
    return user_subset


def build_binary_pa_distance_class_resource_table(
    filtered_pa_choice_table: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    allocation_table_all: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    user_subset = _binary_pa_observational_subset(
        filtered_pa_choice_table,
        demand_user_table_all,
        allocation_table_all,
        class_labels=class_labels,
    )
    class_order = ["Near (<300 m)", "Mid (300-499 m)", "Far (>=500 m)"]
    rows = []
    for pa_choice_label in class_labels:
        pa_subset = user_subset.loc[user_subset["pa_choice_label"].eq(pa_choice_label)].copy()
        totals = pa_subset[["required_rate_mbps", "delivered_rate_mbps", "n_slots", "p_dc_avg_frame_w"]].sum()
        for distance_class in class_order:
            class_subset = pa_subset.loc[pa_subset["distance_class"].eq(distance_class)].copy()
            demand_total = float(class_subset["required_rate_mbps"].sum())
            delivered_total = float(class_subset["delivered_rate_mbps"].sum())
            slot_total = float(class_subset["n_slots"].sum())
            active_dc_total = float(class_subset["p_dc_avg_frame_w"].sum())
            demand_share = demand_total / float(totals["required_rate_mbps"]) if totals["required_rate_mbps"] > 0.0 else float("nan")
            slot_share = slot_total / float(totals["n_slots"]) if totals["n_slots"] > 0.0 else float("nan")
            active_dc_share = active_dc_total / float(totals["p_dc_avg_frame_w"]) if totals["p_dc_avg_frame_w"] > 0.0 else float("nan")
            rows.append(
                {
                    "PA choice": pa_choice_label,
                    "Distance class": distance_class,
                    "Bins": int(pa_subset["bin_index"].nunique()),
                    "Demand total (Mbps)": demand_total,
                    "Demand share": demand_share,
                    "Delivered total (Mbps)": delivered_total,
                    "Slot total": slot_total,
                    "Slot share": slot_share,
                    "Active DC total (W)": active_dc_total,
                    "Active DC share": active_dc_share,
                    "Slots per demanded Mbps": slot_total / demand_total if demand_total > 0.0 else float("nan"),
                    "Active DC W per demanded Mbps": active_dc_total / demand_total if demand_total > 0.0 else float("nan"),
                    "Slot-share / demand-share": slot_share / demand_share if demand_share > 0.0 else float("nan"),
                    "DC-share / demand-share": active_dc_share / demand_share if demand_share > 0.0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_binary_pa_bin_shape_table(
    filtered_pa_choice_table: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    user_subset = demand_user_table_all.merge(
        solved_subset[["scenario_label", "bin_index", "pa_choice_label"]],
        on=["scenario_label", "bin_index"],
        how="inner",
    )
    counts = (
        user_subset.groupby(["pa_choice_label", "bin_index", "distance_class"])["user_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["Near (<300 m)", "Mid (300-499 m)", "Far (>=500 m)"]:
        if label not in counts.columns:
            counts[label] = 0
    per_bin = solved_subset.merge(
        counts,
        on=["pa_choice_label", "bin_index"],
        how="left",
    ).fillna(0.0)
    per_bin["Far demand share"] = per_bin["requested_rate_500m_plus_mbps"] / per_bin["requested_rate_mbps"]
    per_bin["Mid demand share"] = per_bin["requested_rate_300m_to_499m_mbps"] / per_bin["requested_rate_mbps"]
    rows = []
    for pa_choice_label in class_labels:
        subset = per_bin.loc[per_bin["pa_choice_label"].eq(pa_choice_label)].copy()
        rows.append(
            {
                "PA choice": pa_choice_label,
                "Bins": int(len(subset)),
                "Mean total load (Mbps)": float(subset["requested_rate_mbps"].mean()),
                "Mean >= 500 m load (Mbps)": float(subset["requested_rate_500m_plus_mbps"].mean()),
                "Mean 300-499 m load (Mbps)": float(subset["requested_rate_300m_to_499m_mbps"].mean()),
                "Mean far-user count": float(subset["Far (>=500 m)"].mean()),
                "Mean mid-user count": float(subset["Mid (300-499 m)"].mean()),
                "Mean near-user count": float(subset["Near (<300 m)"].mean()),
                "Median far demand share": float(subset["Far demand share"].median()),
                "Median mid demand share": float(subset["Mid demand share"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_binary_pa_nearest_match_table(
    filtered_pa_choice_table: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    *,
    class_labels: tuple[str, str] = ("4W PA", "8W PA"),
) -> pd.DataFrame:
    solved_subset = _solved_500m_pa_subset(filtered_pa_choice_table, class_labels=class_labels)
    user_subset = demand_user_table_all.merge(
        solved_subset[["scenario_label", "bin_index", "pa_choice_label"]],
        on=["scenario_label", "bin_index"],
        how="inner",
    )
    count_table = (
        user_subset.groupby(["pa_choice_label", "bin_index", "distance_class"])["user_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["Near (<300 m)", "Mid (300-499 m)", "Far (>=500 m)"]:
        if label not in count_table.columns:
            count_table[label] = 0
    bin_features = solved_subset.merge(
        count_table,
        on=["pa_choice_label", "bin_index"],
        how="left",
    ).fillna(0.0)

    left_label, right_label = class_labels
    left_bins = bin_features.loc[bin_features["pa_choice_label"].eq(left_label)].copy()
    right_bins = bin_features.loc[bin_features["pa_choice_label"].eq(right_label)].copy()

    rows = []
    for _, right_row in right_bins.iterrows():
        candidates = left_bins.copy()
        candidates["abs_total_load_difference"] = (
            candidates["requested_rate_mbps"] - right_row["requested_rate_mbps"]
        ).abs()
        best_match = candidates.sort_values(["abs_total_load_difference", "bin_index"]).iloc[0]
        rows.append(
            {
                f"{right_label} bin": int(right_row["bin_index"]),
                f"{left_label} match": int(best_match["bin_index"]),
                "Absolute total-load difference (Mbps)": float(best_match["abs_total_load_difference"]),
                f"{right_label} >= 500 m load (Mbps)": float(right_row["requested_rate_500m_plus_mbps"]),
                f"{left_label} >= 500 m load (Mbps)": float(best_match["requested_rate_500m_plus_mbps"]),
                f"{right_label} 300-499 m load (Mbps)": float(right_row["requested_rate_300m_to_499m_mbps"]),
                f"{left_label} 300-499 m load (Mbps)": float(best_match["requested_rate_300m_to_499m_mbps"]),
                "Increase in >= 500 m load (Mbps)": float(
                    right_row["requested_rate_500m_plus_mbps"] - best_match["requested_rate_500m_plus_mbps"]
                ),
                "Increase in 300-499 m load (Mbps)": float(
                    right_row["requested_rate_300m_to_499m_mbps"] - best_match["requested_rate_300m_to_499m_mbps"]
                ),
                f"{right_label} far-user count": int(right_row["Far (>=500 m)"]),
                f"{left_label} far-user count": int(best_match["Far (>=500 m)"]),
                f"{right_label} mid-user count": int(right_row["Mid (300-499 m)"]),
                f"{left_label} mid-user count": int(best_match["Mid (300-499 m)"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Absolute total-load difference (Mbps)").reset_index(drop=True)


def build_binary_pa_nearest_match_summary_table(
    nearest_match_table: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Matched pairs": int(len(nearest_match_table)),
                "Mean absolute total-load difference (Mbps)": float(
                    nearest_match_table["Absolute total-load difference (Mbps)"].mean()
                ),
                "Median absolute total-load difference (Mbps)": float(
                    nearest_match_table["Absolute total-load difference (Mbps)"].median()
                ),
                "Mean increase in >= 500 m load (Mbps)": float(
                    nearest_match_table["Increase in >= 500 m load (Mbps)"].mean()
                ),
                "Median increase in >= 500 m load (Mbps)": float(
                    nearest_match_table["Increase in >= 500 m load (Mbps)"].median()
                ),
                "Mean increase in 300-499 m load (Mbps)": float(
                    nearest_match_table["Increase in 300-499 m load (Mbps)"].mean()
                ),
                "Median increase in 300-499 m load (Mbps)": float(
                    nearest_match_table["Increase in 300-499 m load (Mbps)"].median()
                ),
            }
        ]
    )


def _standby_8w_user_subset(
    allocation_table_all: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    *,
    scenario_label: str = "standby",
    eight_w_label: str = "8W PA",
) -> pd.DataFrame:
    standby_8w_allocations = allocation_table_all.loc[
        allocation_table_all["scenario_label"].eq(scenario_label)
        & allocation_table_all["pa_label"].eq(eight_w_label)
    ].copy()
    if standby_8w_allocations.empty:
        return pd.DataFrame(
            columns=[
                "scenario_label",
                "bin_index",
                "clock_label",
                "user_id",
                "pa_id",
                "pa_label",
                "bandwidth_hz",
                "bandwidth_mhz",
                "n_prb",
                "layers",
                "mcs",
                "n_slots",
                "delivered_rate_bps",
                "delivered_rate_mbps",
                "p_dc_avg_frame_w",
                "p_out_avg_frame_w",
                "distance_m",
                "distance_class",
                "required_rate_bps",
                "required_rate_mbps",
            ]
        )
    return standby_8w_allocations.merge(
        demand_user_table_all[
            [
                "scenario_label",
                "bin_index",
                "user_id",
                "distance_m",
                "distance_class",
                "required_rate_bps",
                "required_rate_mbps",
            ]
        ],
        on=["scenario_label", "bin_index", "user_id"],
        how="left",
    )


def build_standby_8w_user_profile_table(
    allocation_table_all: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    *,
    scenario_label: str = "standby",
    eight_w_label: str = "8W PA",
) -> pd.DataFrame:
    standby_8w_users = _standby_8w_user_subset(
        allocation_table_all,
        demand_user_table_all,
        scenario_label=scenario_label,
        eight_w_label=eight_w_label,
    )
    if standby_8w_users.empty:
        return pd.DataFrame(
            columns=[
                "Distance class",
                "Allocated users",
                "Bins",
                "Requested total (Mbps)",
                "Requested mean (Mbps)",
                "Requested median (Mbps)",
                "Delivered total (Mbps)",
                "Slot total",
                "Active DC total (W)",
            ]
        )
    class_order = ["Near (<300 m)", "Mid (300-499 m)", "Far (>=500 m)"]
    summary = (
        standby_8w_users.groupby("distance_class", dropna=False)
        .agg(
            allocated_users=("user_id", "size"),
            bins=("bin_index", "nunique"),
            requested_total_mbps=("required_rate_mbps", "sum"),
            requested_mean_mbps=("required_rate_mbps", "mean"),
            requested_median_mbps=("required_rate_mbps", "median"),
            delivered_total_mbps=("delivered_rate_mbps", "sum"),
            slot_total=("n_slots", "sum"),
            active_dc_total_w=("p_dc_avg_frame_w", "sum"),
        )
        .reset_index()
    )
    summary["distance_class"] = pd.Categorical(summary["distance_class"], categories=class_order, ordered=True)
    summary = summary.sort_values("distance_class").reset_index(drop=True)
    return summary.rename(
        columns={
            "distance_class": "Distance class",
            "allocated_users": "Allocated users",
            "bins": "Bins",
            "requested_total_mbps": "Requested total (Mbps)",
            "requested_mean_mbps": "Requested mean (Mbps)",
            "requested_median_mbps": "Requested median (Mbps)",
            "delivered_total_mbps": "Delivered total (Mbps)",
            "slot_total": "Slot total",
            "active_dc_total_w": "Active DC total (W)",
        }
    )


def build_standby_hard_off_8w_footprint_table(
    allocation_table_all: pd.DataFrame,
    demand_user_table_all: pd.DataFrame,
    hard_off_pa_choice_table: pd.DataFrame,
    *,
    standby_label: str = "standby",
    eight_w_label: str = "8W PA",
) -> pd.DataFrame:
    standby_8w_users = _standby_8w_user_subset(
        allocation_table_all,
        demand_user_table_all,
        scenario_label=standby_label,
        eight_w_label=eight_w_label,
    )
    hard_off_bin_view = hard_off_pa_choice_table[
        [
            "bin_index",
            "status",
            "pa_choice_label",
            "requested_rate_mbps",
            "requested_rate_500m_plus_mbps",
            "requested_rate_300m_to_499m_mbps",
        ]
    ].rename(
        columns={
            "status": "hard_off_status",
            "pa_choice_label": "hard_off_pa_choice_label",
            "requested_rate_mbps": "hard_off_requested_rate_mbps",
            "requested_rate_500m_plus_mbps": "hard_off_requested_rate_500m_plus_mbps",
            "requested_rate_300m_to_499m_mbps": "hard_off_requested_rate_300m_to_499m_mbps",
        }
    )

    if standby_8w_users.empty:
        footprint_table = hard_off_bin_view.copy()
        footprint_table["standby_8w_user_count"] = 0
        footprint_table["standby_8w_requested_rate_mbps"] = 0.0
        footprint_table["standby_8w_delivered_rate_mbps"] = 0.0
        footprint_table["standby_8w_slot_total"] = 0.0
        footprint_table["standby_8w_active_dc_total_w"] = 0.0
        footprint_table["standby_8w_near_load_mbps"] = 0.0
        footprint_table["standby_8w_mid_load_mbps"] = 0.0
        footprint_table["standby_8w_far_load_mbps"] = 0.0
        footprint_table["standby_8w_near_user_count"] = 0
        footprint_table["standby_8w_mid_user_count"] = 0
        footprint_table["standby_8w_far_user_count"] = 0
    else:
        load_by_class = (
            standby_8w_users.groupby(["bin_index", "distance_class"])["required_rate_mbps"]
            .sum()
            .unstack(fill_value=0.0)
            .reset_index()
        )
        user_count_by_class = (
            standby_8w_users.groupby(["bin_index", "distance_class"])["user_id"]
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        class_column_map = {
            "Near (<300 m)": "standby_8w_near_load_mbps",
            "Mid (300-499 m)": "standby_8w_mid_load_mbps",
            "Far (>=500 m)": "standby_8w_far_load_mbps",
        }
        count_column_map = {
            "Near (<300 m)": "standby_8w_near_user_count",
            "Mid (300-499 m)": "standby_8w_mid_user_count",
            "Far (>=500 m)": "standby_8w_far_user_count",
        }
        for source_name, target_name in class_column_map.items():
            if source_name not in load_by_class.columns:
                load_by_class[source_name] = 0.0
            load_by_class = load_by_class.rename(columns={source_name: target_name})
        for source_name, target_name in count_column_map.items():
            if source_name not in user_count_by_class.columns:
                user_count_by_class[source_name] = 0
            user_count_by_class = user_count_by_class.rename(columns={source_name: target_name})

        standby_8w_per_bin = (
            standby_8w_users.groupby("bin_index")
            .agg(
                standby_8w_user_count=("user_id", "size"),
                standby_8w_requested_rate_mbps=("required_rate_mbps", "sum"),
                standby_8w_delivered_rate_mbps=("delivered_rate_mbps", "sum"),
                standby_8w_slot_total=("n_slots", "sum"),
                standby_8w_active_dc_total_w=("p_dc_avg_frame_w", "sum"),
            )
            .reset_index()
            .merge(load_by_class, on="bin_index", how="left")
            .merge(user_count_by_class, on="bin_index", how="left")
        )
        footprint_table = hard_off_bin_view.merge(standby_8w_per_bin, on="bin_index", how="left")
        fill_zero_columns = [
            "standby_8w_user_count",
            "standby_8w_requested_rate_mbps",
            "standby_8w_delivered_rate_mbps",
            "standby_8w_slot_total",
            "standby_8w_active_dc_total_w",
            "standby_8w_near_load_mbps",
            "standby_8w_mid_load_mbps",
            "standby_8w_far_load_mbps",
            "standby_8w_near_user_count",
            "standby_8w_mid_user_count",
            "standby_8w_far_user_count",
        ]
        footprint_table[fill_zero_columns] = footprint_table[fill_zero_columns].fillna(0.0)

    footprint_table["standby_any_8w"] = footprint_table["standby_8w_user_count"].gt(0)
    total_8w_load = footprint_table["standby_8w_requested_rate_mbps"]
    footprint_table["standby_8w_near_load_share"] = np.where(
        total_8w_load.gt(0.0),
        footprint_table["standby_8w_near_load_mbps"] / total_8w_load,
        0.0,
    )
    footprint_table["standby_8w_mid_load_share"] = np.where(
        total_8w_load.gt(0.0),
        footprint_table["standby_8w_mid_load_mbps"] / total_8w_load,
        0.0,
    )
    footprint_table["standby_8w_far_load_share"] = np.where(
        total_8w_load.gt(0.0),
        footprint_table["standby_8w_far_load_mbps"] / total_8w_load,
        0.0,
    )
    footprint_table["standby_hard_off_group"] = np.select(
        [
            footprint_table["standby_any_8w"].eq(False),
            footprint_table["hard_off_status"].ne("solved"),
            footprint_table["hard_off_pa_choice_label"].eq("8W PA"),
            footprint_table["hard_off_pa_choice_label"].eq("4W PA"),
        ],
        [
            "No standby 8W",
            "Standby 8W with hard-off infeasible",
            "Hard-off 8W overlap",
            "Standby-only 8W",
        ],
        default="Standby 8W with hard-off mixed",
    )
    color_map = _build_pa_choice_color_map({0: "8W PA", 1: "4W PA"})
    footprint_table["hard_off_pa_choice_color"] = footprint_table["hard_off_pa_choice_label"].map(color_map).fillna("#7f7f7f")
    return footprint_table.sort_values("bin_index").reset_index(drop=True)


def build_standby_hard_off_overlap_summary_table(
    standby_hard_off_footprint_table: pd.DataFrame,
) -> pd.DataFrame:
    solved = standby_hard_off_footprint_table.loc[
        standby_hard_off_footprint_table["hard_off_status"].eq("solved")
    ].copy()
    standby_any_8w = solved["standby_any_8w"].to_numpy(dtype=bool)
    hard_off_8w = solved["hard_off_pa_choice_label"].eq("8W PA").to_numpy(dtype=bool)
    overlap = standby_any_8w & hard_off_8w
    phi_coefficient = float(np.corrcoef(standby_any_8w.astype(float), hard_off_8w.astype(float))[0, 1])
    return pd.DataFrame(
        [
            {
                "Solved bins": int(len(solved)),
                "Standby bins with any 8W user": int(standby_any_8w.sum()),
                "Hard-off 8W bins": int(hard_off_8w.sum()),
                "Overlap bins": int(overlap.sum()),
                "Precision of standby-any-8W for hard-off 8W": float(overlap.sum() / standby_any_8w.sum()) if standby_any_8w.sum() > 0 else float("nan"),
                "Recall of standby-any-8W for hard-off 8W": float(overlap.sum() / hard_off_8w.sum()) if hard_off_8w.sum() > 0 else float("nan"),
                "Phi coefficient": phi_coefficient,
            }
        ]
    )


def build_standby_hard_off_group_shape_table(
    standby_hard_off_footprint_table: pd.DataFrame,
) -> pd.DataFrame:
    group_order = [
        "Hard-off 8W overlap",
        "Standby-only 8W",
        "Standby 8W with hard-off infeasible",
    ]
    subset = standby_hard_off_footprint_table.loc[
        standby_hard_off_footprint_table["standby_hard_off_group"].isin(group_order)
    ].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "Group",
                "Bins",
                "Mean hard-off total load (Mbps)",
                "Mean hard-off >= 500 m load (Mbps)",
                "Mean hard-off 300-499 m load (Mbps)",
                "Mean standby 8W users",
                "Mean standby 8W total load (Mbps)",
                "Mean standby 8W >= 500 m load (Mbps)",
                "Mean standby 8W 300-499 m load (Mbps)",
                "Mean standby 8W < 300 m load (Mbps)",
                "Mean standby 8W far-load share",
                "Mean standby 8W mid-load share",
                "Mean standby 8W near-load share",
            ]
        )
    summary = (
        subset.groupby("standby_hard_off_group", dropna=False)
        .agg(
            bins=("bin_index", "size"),
            mean_hard_off_total_load_mbps=("hard_off_requested_rate_mbps", "mean"),
            mean_hard_off_500m_plus_load_mbps=("hard_off_requested_rate_500m_plus_mbps", "mean"),
            mean_hard_off_300_499_load_mbps=("hard_off_requested_rate_300m_to_499m_mbps", "mean"),
            mean_standby_8w_users=("standby_8w_user_count", "mean"),
            mean_standby_8w_total_load_mbps=("standby_8w_requested_rate_mbps", "mean"),
            mean_standby_8w_500m_plus_load_mbps=("standby_8w_far_load_mbps", "mean"),
            mean_standby_8w_300_499_load_mbps=("standby_8w_mid_load_mbps", "mean"),
            mean_standby_8w_near_load_mbps=("standby_8w_near_load_mbps", "mean"),
            mean_standby_8w_far_load_share=("standby_8w_far_load_share", "mean"),
            mean_standby_8w_mid_load_share=("standby_8w_mid_load_share", "mean"),
            mean_standby_8w_near_load_share=("standby_8w_near_load_share", "mean"),
        )
        .reset_index()
    )
    summary["standby_hard_off_group"] = pd.Categorical(summary["standby_hard_off_group"], categories=group_order, ordered=True)
    summary = summary.sort_values("standby_hard_off_group").reset_index(drop=True)
    return summary.rename(
        columns={
            "standby_hard_off_group": "Group",
            "bins": "Bins",
            "mean_hard_off_total_load_mbps": "Mean hard-off total load (Mbps)",
            "mean_hard_off_500m_plus_load_mbps": "Mean hard-off >= 500 m load (Mbps)",
            "mean_hard_off_300_499_load_mbps": "Mean hard-off 300-499 m load (Mbps)",
            "mean_standby_8w_users": "Mean standby 8W users",
            "mean_standby_8w_total_load_mbps": "Mean standby 8W total load (Mbps)",
            "mean_standby_8w_500m_plus_load_mbps": "Mean standby 8W >= 500 m load (Mbps)",
            "mean_standby_8w_300_499_load_mbps": "Mean standby 8W 300-499 m load (Mbps)",
            "mean_standby_8w_near_load_mbps": "Mean standby 8W < 300 m load (Mbps)",
            "mean_standby_8w_far_load_share": "Mean standby 8W far-load share",
            "mean_standby_8w_mid_load_share": "Mean standby 8W mid-load share",
            "mean_standby_8w_near_load_share": "Mean standby 8W near-load share",
        }
    )


def build_standby_hard_off_feature_test_table(
    standby_hard_off_footprint_table: pd.DataFrame,
) -> pd.DataFrame:
    from scipy import stats

    subset = standby_hard_off_footprint_table.loc[
        standby_hard_off_footprint_table["standby_hard_off_group"].isin(["Hard-off 8W overlap", "Standby-only 8W"])
    ].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Hard-off 8W overlap mean",
                "Hard-off 8W overlap median",
                "Standby-only 8W mean",
                "Standby-only 8W median",
                "Mann-Whitney U",
                "Two-sided p-value",
            ]
        )
    feature_map = {
        "Standby 8W total load (Mbps)": "standby_8w_requested_rate_mbps",
        "Standby 8W load from >= 500 m users (Mbps)": "standby_8w_far_load_mbps",
        "Standby 8W far-load share": "standby_8w_far_load_share",
        "Standby 8W near-load share": "standby_8w_near_load_share",
    }
    rows = []
    left_rows = subset.loc[subset["standby_hard_off_group"].eq("Hard-off 8W overlap")]
    right_rows = subset.loc[subset["standby_hard_off_group"].eq("Standby-only 8W")]
    for feature_label, feature_column in feature_map.items():
        left = left_rows[feature_column].to_numpy(dtype=float)
        right = right_rows[feature_column].to_numpy(dtype=float)
        statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
        rows.append(
            {
                "Feature": feature_label,
                "Hard-off 8W overlap mean": float(np.mean(left)),
                "Hard-off 8W overlap median": float(np.median(left)),
                "Standby-only 8W mean": float(np.mean(right)),
                "Standby-only 8W median": float(np.median(right)),
                "Mann-Whitney U": float(statistic),
                "Two-sided p-value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def build_policy_summary_view_for_bin(
    bin_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_index: int,
) -> pd.DataFrame:
    scenario_order = list(scenario_order)
    rows = []
    for scenario_label in scenario_order:
        row = (
            bin_table_all.loc[
                bin_table_all["scenario_label"].eq(scenario_label)
                & bin_table_all["bin_index"].eq(int(bin_index))
            ]
            .iloc[0]
        )
        rows.append(
            {
                "switch_policy": str(scenario_label),
                "total_power_w": float(row["dc_total_w"]),
                "active_power_w": float(row["dc_active_w"]),
                "inactive_power_w": float(row["dc_inactive_w"]),
                "slot_total": int(row["used_slots"]) if np.isfinite(row["used_slots"]) else 0,
                "unused_slots": int(row["unused_slots"]) if np.isfinite(row["unused_slots"]) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_allocation_views_for_bin(
    allocation_table_all: pd.DataFrame,
    bin_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_index: int,
) -> Dict[str, dict]:
    scenario_order = list(scenario_order)
    allocation_views = {}

    all_user_ids = sorted(
        allocation_table_all.loc[allocation_table_all["bin_index"].eq(int(bin_index)), "user_id"].unique().tolist()
    )
    color_levels = np.linspace(0.25, 0.85, max(len(all_user_ids), 1))
    user_color_map = {
        int(user_id): colors.to_hex(plt.cm.cividis(level))
        for user_id, level in zip(all_user_ids, color_levels[::-1], strict=False)
    }
    if not user_color_map:
        user_color_map = {0: BASE_USER_COLORS[0]}

    for scenario_label in scenario_order:
        scenario_rows = (
            allocation_table_all.loc[
                allocation_table_all["scenario_label"].eq(scenario_label)
                & allocation_table_all["bin_index"].eq(int(bin_index))
            ]
            .sort_values("user_id")
            .reset_index(drop=True)
        )
        scenario_bin_row = (
            bin_table_all.loc[
                bin_table_all["scenario_label"].eq(scenario_label)
                & bin_table_all["bin_index"].eq(int(bin_index))
            ]
            .iloc[0]
        )
        used_slots = int(scenario_bin_row["used_slots"]) if np.isfinite(scenario_bin_row["used_slots"]) else 0
        unused_slots = int(scenario_bin_row["unused_slots"]) if np.isfinite(scenario_bin_row["unused_slots"]) else 0
        total_slots = max(used_slots + unused_slots, int(scenario_rows["n_slots"].sum()) if not scenario_rows.empty else 0)
        total_prbs = int(scenario_rows["n_prb"].max()) if not scenario_rows.empty else 1

        blocks = []
        slot_cursor = 0
        for row in scenario_rows.itertuples(index=False):
            block = {
                "user_id": int(row.user_id),
                "pa_label": str(row.pa_label),
                "n_prb": int(row.n_prb),
                "n_slots": int(row.n_slots),
                "layers": int(row.layers),
                "mcs": int(row.mcs),
                "p_dc_avg_frame_w": float(row.p_dc_avg_frame_w),
                "slot_start": int(slot_cursor),
                "slot_end": int(slot_cursor + int(row.n_slots)),
                "color": str(user_color_map.get(int(row.user_id), BASE_USER_COLORS[int(row.user_id) % len(BASE_USER_COLORS)])),
            }
            blocks.append(block)
            slot_cursor = block["slot_end"]

        unused_blocks = []
        for block in blocks:
            unused_prbs = total_prbs - int(block["n_prb"])
            if unused_prbs > 0:
                unused_blocks.append(
                    {
                        "x": int(block["n_prb"]),
                        "y": int(block["slot_start"]),
                        "width": int(unused_prbs),
                        "height": int(block["n_slots"]),
                    }
                )
        if slot_cursor < total_slots:
            unused_blocks.append(
                {
                    "x": 0,
                    "y": int(slot_cursor),
                    "width": int(total_prbs),
                    "height": int(total_slots - slot_cursor),
                }
            )

        allocation_views[scenario_label] = {
            "blocks": blocks,
            "total_prbs": total_prbs,
            "total_slots": total_slots,
            "frame_slots": total_slots,
            "window_boundaries": [],
            "unused_blocks": unused_blocks,
        }

    return allocation_views


def build_problem_stub_from_allocation_views(allocation_views: Mapping[str, dict]) -> SimpleNamespace:
    max_layers = 1
    for allocation_view in allocation_views.values():
        for block in allocation_view.get("blocks", []):
            max_layers = max(max_layers, int(block.get("layers", 1)))
    return SimpleNamespace(n_tx_chains=max_layers)


def style_run_overview_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean total power (W)": "{:.2f}",
            "Peak total power (W)": "{:.2f}",
            "Day energy (Wh)": "{:.2f}",
        },
        caption=caption,
    )


def style_highlighted_bin_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Requested rate (Mbps)": "{:.2f}",
            "Used slots": "{:.0f}",
            "Unused slots": "{:.0f}",
        },
        caption=caption,
    )


def style_bin_comparison_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Requested rate (Mbps)": "{:.2f}",
            "Delivered rate (Mbps)": "{:.2f}",
            "Total power (W)": "{:.2f}",
            "Active power (W)": "{:.2f}",
            "Inactive power (W)": "{:.2f}",
            "Used slots": "{:.0f}",
            "Unused slots": "{:.0f}",
        },
        caption=caption,
    )


def style_allocation_choice_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Bandwidth (MHz)": "{:.0f}",
            "Delivered rate (Mbps)": "{:.2f}",
            "Avg frame DC power (W)": "{:.2f}",
            "Avg frame RF power (W)": "{:.2f}",
        },
        caption=caption,
    )


def style_day_total_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean total power (W)": "{:.2f}",
            "Peak total power (W)": "{:.2f}",
            "Day energy (Wh)": "{:.2f}",
            "Mean unused slots": "{:.2f}",
            "Peak requested rate (Mbps)": "{:.2f}",
        },
        caption=caption,
    )


def style_filtered_500m_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean total load (Mbps)": "{:.2f}",
            "Mean load from 500 m+ users (Mbps)": "{:.2f}",
            "Mean load from 300-499 m users (Mbps)": "{:.2f}",
            "Mean max 500 m user (Mbps)": "{:.2f}",
        },
        caption=caption,
    )


def style_infeasibility_boundary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Feasible median load from >= 500 m users (Mbps)": "{:.2f}",
            "Infeasible median load from >= 500 m users (Mbps)": "{:.2f}",
            "Feasible max load from >= 500 m users (Mbps)": "{:.2f}",
            "Infeasible min load from >= 500 m users (Mbps)": "{:.2f}",
            "Separation gap (Mbps)": "{:.2f}",
        },
        caption=caption,
    )


def style_binary_pa_feature_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "4W PA mean": "{:.2f}",
            "4W PA median": "{:.2f}",
            "4W PA std": "{:.2f}",
            "8W PA mean": "{:.2f}",
            "8W PA median": "{:.2f}",
            "8W PA std": "{:.2f}",
        },
        caption=caption,
    )


def style_binary_pa_univariate_test_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mann-Whitney U": "{:.1f}",
            "Two-sided p-value": "{:.3e}",
        },
        caption=caption,
    )


def style_binary_pa_classifier_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Training accuracy": "{:.3f}",
            "LOO accuracy": "{:.3f}",
        },
        caption=caption,
    )


def style_binary_pa_multivariate_test_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Hotelling T^2": "{:.2f}",
            "F statistic": "{:.2f}",
            "p-value": "{:.3e}",
            "LDA weight on >= 500 m load": "{:.3f}",
            "LDA weight on 300-499 m load": "{:.3f}",
            "Relative weight >= 500 m / 300-499 m": "{:.2f}",
            "LDA threshold": "{:.3f}",
        },
        caption=caption,
    )


def style_binary_pa_distance_class_resource_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Demand total (Mbps)": "{:.2f}",
            "Demand share": "{:.3f}",
            "Delivered total (Mbps)": "{:.2f}",
            "Slot total": "{:.0f}",
            "Slot share": "{:.3f}",
            "Active DC total (W)": "{:.2f}",
            "Active DC share": "{:.3f}",
            "Slots per demanded Mbps": "{:.3f}",
            "Active DC W per demanded Mbps": "{:.3f}",
            "Slot-share / demand-share": "{:.3f}",
            "DC-share / demand-share": "{:.3f}",
        },
        caption=caption,
    )


def style_binary_pa_bin_shape_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean total load (Mbps)": "{:.2f}",
            "Mean >= 500 m load (Mbps)": "{:.2f}",
            "Mean 300-499 m load (Mbps)": "{:.2f}",
            "Mean far-user count": "{:.2f}",
            "Mean mid-user count": "{:.2f}",
            "Mean near-user count": "{:.2f}",
            "Median far demand share": "{:.3f}",
            "Median mid demand share": "{:.3f}",
        },
        caption=caption,
    )


def style_binary_pa_nearest_match_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Absolute total-load difference (Mbps)": "{:.2f}",
            "4W PA >= 500 m load (Mbps)": "{:.2f}",
            "8W PA >= 500 m load (Mbps)": "{:.2f}",
            "4W PA 300-499 m load (Mbps)": "{:.2f}",
            "8W PA 300-499 m load (Mbps)": "{:.2f}",
            "Increase in >= 500 m load (Mbps)": "{:.2f}",
            "Increase in 300-499 m load (Mbps)": "{:.2f}",
        },
        caption=caption,
    )


def style_binary_pa_nearest_match_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean absolute total-load difference (Mbps)": "{:.2f}",
            "Median absolute total-load difference (Mbps)": "{:.2f}",
            "Mean increase in >= 500 m load (Mbps)": "{:.2f}",
            "Median increase in >= 500 m load (Mbps)": "{:.2f}",
            "Mean increase in 300-499 m load (Mbps)": "{:.2f}",
            "Median increase in 300-499 m load (Mbps)": "{:.2f}",
        },
        caption=caption,
    )


def style_standby_8w_user_profile_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Requested total (Mbps)": "{:.2f}",
            "Requested mean (Mbps)": "{:.2f}",
            "Requested median (Mbps)": "{:.2f}",
            "Delivered total (Mbps)": "{:.2f}",
            "Slot total": "{:.0f}",
            "Active DC total (W)": "{:.2f}",
        },
        caption=caption,
    )


def style_standby_hard_off_overlap_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Precision of standby-any-8W for hard-off 8W": "{:.3f}",
            "Recall of standby-any-8W for hard-off 8W": "{:.3f}",
            "Phi coefficient": "{:.3f}",
        },
        caption=caption,
    )


def style_standby_hard_off_group_shape_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Mean hard-off total load (Mbps)": "{:.2f}",
            "Mean hard-off >= 500 m load (Mbps)": "{:.2f}",
            "Mean hard-off 300-499 m load (Mbps)": "{:.2f}",
            "Mean standby 8W users": "{:.2f}",
            "Mean standby 8W total load (Mbps)": "{:.2f}",
            "Mean standby 8W >= 500 m load (Mbps)": "{:.2f}",
            "Mean standby 8W 300-499 m load (Mbps)": "{:.2f}",
            "Mean standby 8W < 300 m load (Mbps)": "{:.2f}",
            "Mean standby 8W far-load share": "{:.3f}",
            "Mean standby 8W mid-load share": "{:.3f}",
            "Mean standby 8W near-load share": "{:.3f}",
        },
        caption=caption,
    )


def style_standby_hard_off_feature_test_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={
            "Hard-off 8W overlap mean": "{:.2f}",
            "Hard-off 8W overlap median": "{:.2f}",
            "Standby-only 8W mean": "{:.2f}",
            "Standby-only 8W median": "{:.2f}",
            "Mann-Whitney U": "{:.1f}",
            "Two-sided p-value": "{:.3e}",
        },
        caption=caption,
    )


def _set_day_bin_boundaries(ax, *, day_bin_count: int) -> None:
    for boundary in range(0, int(day_bin_count) + 1, 4):
        ax.axvline(boundary - 0.5, color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, int(day_bin_count) - 0.5)
    ax.set_xticks(range(0, int(day_bin_count), 8))


def _plot_infeasible_markers(ax, scenario_bins: pd.DataFrame, *, y_value: float) -> None:
    infeasible_bins = scenario_bins.loc[scenario_bins["status"].ne("solved"), "bin_index"]
    if not infeasible_bins.empty:
        ax.scatter(
            infeasible_bins.to_numpy(dtype=float),
            np.full(len(infeasible_bins), y_value, dtype=float),
            marker="x",
            s=36,
            linewidths=1.1,
            color="black",
            zorder=5,
        )


def plot_day_demand_context(reference_demand_table: pd.DataFrame):
    day_bin_count = int(len(reference_demand_table))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    axes[0].bar(
        reference_demand_table["bin_index"],
        reference_demand_table["requested_rate_mbps"],
        width=0.9,
        color="#4c78a8",
        alpha=0.35,
    )
    axes[0].plot(
        reference_demand_table["bin_index"],
        reference_demand_table["requested_rate_mbps"],
        color="#1f4e79",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Requested rate (Mbps)")
    axes[0].set_title("Shared day demand seen by every compared scenario")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(
        reference_demand_table["bin_index"],
        reference_demand_table["user_count"],
        width=0.9,
        color="#dd8452",
        alpha=0.85,
    )
    axes[1].set_xlabel("Quarter-hour bin index")
    axes[1].set_ylabel("Active users")
    axes[1].set_title("Scheduler-facing active users in each bin")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)

    plt.tight_layout()
    return fig, axes


def plot_day_power_trajectories(bin_table_all: pd.DataFrame, *, scenario_order: Iterable[str]):
    scenario_order = list(scenario_order)
    day_bin_count = int(bin_table_all["bin_index"].max()) + 1
    fig, ax = plt.subplots(figsize=(12, 4.8))

    for scenario_label in scenario_order:
        scenario_bins = (
            bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)]
            .sort_values("bin_index")
            .reset_index(drop=True)
        )
        ax.plot(
            scenario_bins["bin_index"],
            scenario_bins["dc_total_w"],
            linewidth=2.0,
            marker="o",
            markersize=3.2,
            label=str(scenario_label),
        )
        finite_values = scenario_bins["dc_total_w"].to_numpy(dtype=float)
        if np.isfinite(finite_values).any():
            _plot_infeasible_markers(ax, scenario_bins, y_value=float(np.nanmax(finite_values)) * 1.02)

    _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel("Total PA DC power (W)")
    ax.set_title("Day-level total power under the compared scenarios")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def plot_day_power_decomposition(bin_table_all: pd.DataFrame, *, scenario_order: Iterable[str]):
    scenario_order = list(scenario_order)
    day_bin_count = int(bin_table_all["bin_index"].max()) + 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)
    metric_columns = [
        ("dc_total_w", "Total PA DC power (W)", "Total power"),
        ("dc_active_w", "Active PA DC power (W)", "Active contribution"),
        ("dc_inactive_w", "Inactive PA DC power (W)", "Inactive contribution"),
    ]

    for ax, (column, ylabel, title) in zip(axes, metric_columns, strict=False):
        for scenario_label in scenario_order:
            scenario_bins = (
                bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)]
                .sort_values("bin_index")
                .reset_index(drop=True)
            )
            ax.plot(
                scenario_bins["bin_index"],
                scenario_bins[column],
                linewidth=1.9,
                label=str(scenario_label),
            )
        _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("Quarter-hour bin index")
    axes[0].legend(frameon=True, ncol=min(len(scenario_order), 3))
    fig.suptitle("The same day demand produces different active and inactive power traces", y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_scenario_pa_choice_trace(
    pa_choice_table: pd.DataFrame,
    *,
    scenario_label: str,
) -> tuple[plt.Figure, np.ndarray]:
    day_bin_count = int(pa_choice_table["bin_index"].max()) + 1
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.6]},
    )

    axes[0].plot(
        pa_choice_table["bin_index"],
        pa_choice_table["requested_rate_300m_plus_mbps"],
        color="#444444",
        linewidth=1.8,
        alpha=0.9,
    )
    axes[1].plot(
        pa_choice_table["bin_index"],
        pa_choice_table["dc_total_w"],
        color="#444444",
        linewidth=1.8,
        alpha=0.9,
    )

    choice_order = []
    for value in pa_choice_table["pa_choice_label"].tolist():
        if value not in choice_order:
            choice_order.append(value)

    for choice_label in choice_order:
        choice_rows = pa_choice_table.loc[pa_choice_table["pa_choice_label"].eq(choice_label)].copy()
        choice_color = choice_rows["pa_choice_color"].iloc[0]
        top_marker = "x" if choice_label == "Infeasible" else "o"
        top_scatter_kwargs = {
            "s": 36,
            "marker": top_marker,
            "color": choice_color,
            "label": choice_label,
            "zorder": 3,
        }
        if choice_label != "Infeasible":
            top_scatter_kwargs.update(
                {
                    "edgecolor": "black",
                    "linewidth": 0.45,
                }
            )
        else:
            top_scatter_kwargs.update(
                {
                    "linewidth": 1.1,
                }
            )
        axes[0].scatter(
            choice_rows["bin_index"],
            choice_rows["requested_rate_300m_plus_mbps"],
            **top_scatter_kwargs,
        )
        if choice_label == "Infeasible":
            axes[1].scatter(
                choice_rows["bin_index"],
                np.full(len(choice_rows), 0.0),
                s=42,
                marker="x",
                color=choice_color,
                linewidth=1.2,
                label=choice_label,
                zorder=4,
            )
            continue

        axes[1].scatter(
            choice_rows["bin_index"],
            choice_rows["dc_total_w"],
            s=36,
            color=choice_color,
            edgecolor="black",
            linewidth=0.45,
            label=choice_label,
            zorder=3,
        )

    axes[0].set_ylabel("Requested rate from users at >= 300 m (Mbps)")
    axes[0].set_title(
        f"{scenario_label}: distant-user demand burden coloured by the selected PA"
    )
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].set_xlabel("Quarter-hour bin index")
    axes[1].set_ylabel("Total PA DC power (W)")
    axes[1].set_title(
        f"{scenario_label}: the same PA choice projected onto the power trace"
    )
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(labels), 4), frameon=True)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def plot_scenario_max_500m_rate_trace(
    pa_choice_table: pd.DataFrame,
    *,
    scenario_label: str,
) -> tuple[plt.Figure, plt.Axes]:
    day_bin_count = int(pa_choice_table["bin_index"].max()) + 1
    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.plot(
        pa_choice_table["bin_index"],
        pa_choice_table["max_requested_rate_500m_plus_mbps"],
        color="#444444",
        linewidth=1.8,
        alpha=0.9,
    )

    choice_order = []
    for value in pa_choice_table["pa_choice_label"].tolist():
        if value not in choice_order:
            choice_order.append(value)

    for choice_label in choice_order:
        choice_rows = pa_choice_table.loc[pa_choice_table["pa_choice_label"].eq(choice_label)].copy()
        choice_color = choice_rows["pa_choice_color"].iloc[0]
        marker = "x" if choice_label == "Infeasible" else "o"
        scatter_kwargs = {
            "s": 38,
            "marker": marker,
            "color": choice_color,
            "label": choice_label,
            "zorder": 3,
        }
        if choice_label != "Infeasible":
            scatter_kwargs.update({"edgecolor": "black", "linewidth": 0.45})
        else:
            scatter_kwargs.update({"linewidth": 1.1})
        ax.scatter(
            choice_rows["bin_index"],
            choice_rows["max_requested_rate_500m_plus_mbps"],
            **scatter_kwargs,
        )

    _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel("Max requested rate among users at >= 500 m (Mbps)")
    ax.set_title(
        f"{scenario_label}: bins coloured by the selected PA against the strongest 500 m user"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True, ncol=min(len(choice_order), 4))
    plt.tight_layout()
    return fig, ax


def plot_pa_choice_load_scatter(
    pa_choice_table: pd.DataFrame,
    *,
    scenario_label: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    choice_order = []
    for value in pa_choice_table["pa_choice_label"].tolist():
        if value not in choice_order:
            choice_order.append(value)

    for choice_label in choice_order:
        choice_rows = pa_choice_table.loc[pa_choice_table["pa_choice_label"].eq(choice_label)].copy()
        choice_color = choice_rows["pa_choice_color"].iloc[0]
        marker = "x" if choice_label == "Infeasible" else "o"
        scatter_kwargs = {
            "s": 52,
            "marker": marker,
            "color": choice_color,
            "label": choice_label,
            "zorder": 3,
        }
        if choice_label != "Infeasible":
            scatter_kwargs.update({"edgecolor": "black", "linewidth": 0.5})
        else:
            scatter_kwargs.update({"linewidth": 1.2})
        ax.scatter(
            choice_rows["requested_rate_mbps"],
            choice_rows["requested_rate_300m_plus_mbps"],
            **scatter_kwargs,
        )
        if choice_label == "Infeasible":
            for row in choice_rows.itertuples(index=False):
                ax.annotate(
                    f"{int(row.bin_index)}",
                    (float(row.requested_rate_mbps), float(row.requested_rate_300m_plus_mbps)),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_xlabel("Total requested rate in bin (Mbps)")
    ax.set_ylabel("Requested rate from users at >= 300 m (Mbps)")
    ax.set_title(
        f"{scenario_label}: total-load versus distant-load space coloured by the selected PA"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def plot_filtered_500m_shape_scatter(
    filtered_pa_choice_table: pd.DataFrame,
    *,
    scenario_label: str,
) -> tuple[plt.Figure, np.ndarray]:
    if filtered_pa_choice_table.empty:
        raise ValueError("The filtered PA-choice table does not contain any bins with a 500 m user.")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7), sharey=True)

    choice_order = []
    for value in filtered_pa_choice_table["pa_choice_label"].tolist():
        if value not in choice_order:
            choice_order.append(value)

    for choice_label in choice_order:
        choice_rows = filtered_pa_choice_table.loc[
            filtered_pa_choice_table["pa_choice_label"].eq(choice_label)
        ].copy()
        choice_color = choice_rows["pa_choice_color"].iloc[0]
        marker = "x" if choice_label == "Infeasible" else "o"
        scatter_kwargs = {
            "s": 56,
            "marker": marker,
            "color": choice_color,
            "label": choice_label,
            "zorder": 3,
        }
        if choice_label != "Infeasible":
            scatter_kwargs.update({"edgecolor": "black", "linewidth": 0.5})
        else:
            scatter_kwargs.update({"linewidth": 1.2})

        axes[0].scatter(
            choice_rows["max_requested_rate_500m_plus_mbps"],
            choice_rows["requested_rate_300m_to_499m_mbps"],
            **scatter_kwargs,
        )
        axes[1].scatter(
            choice_rows["requested_rate_500m_plus_mbps"],
            choice_rows["requested_rate_300m_to_499m_mbps"],
            **scatter_kwargs,
        )

        if choice_label in {"8W PA", "Infeasible"}:
            for row in choice_rows.itertuples(index=False):
                axes[0].annotate(
                    f"{int(row.bin_index)}",
                    (
                        float(row.max_requested_rate_500m_plus_mbps),
                        float(row.requested_rate_300m_to_499m_mbps),
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
                axes[1].annotate(
                    f"{int(row.bin_index)}",
                    (
                        float(row.requested_rate_500m_plus_mbps),
                        float(row.requested_rate_300m_to_499m_mbps),
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    axes[0].set_xlabel("Max requested rate among users at >= 500 m (Mbps)")
    axes[0].set_ylabel("Requested rate from users at 300-499 m (Mbps)")
    axes[0].set_title("Strongest 500 m user versus the 300-499 m burden")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Requested rate from users at >= 500 m (Mbps)")
    axes[1].set_title("Total 500 m burden versus the 300-499 m burden")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(len(labels), 4),
            frameon=True,
        )

    fig.suptitle(
        f"{scenario_label}: only bins with at least one 500 m user, coloured by the selected PA",
        y=1.06,
    )
    plt.tight_layout()
    return fig, axes


def plot_standby_hard_off_8w_footprint_scatter(
    standby_hard_off_footprint_table: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    subset = standby_hard_off_footprint_table.loc[
        standby_hard_off_footprint_table["standby_any_8w"]
        & standby_hard_off_footprint_table["hard_off_status"].eq("solved")
    ].copy()
    if subset.empty:
        raise ValueError("No solved bins with standby 8W assignments were found for the current scenario pair.")

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    choice_order = []
    for value in subset["hard_off_pa_choice_label"].tolist():
        label = str(value)
        if label not in choice_order:
            choice_order.append(label)

    for choice_label in choice_order:
        choice_rows = subset.loc[subset["hard_off_pa_choice_label"].eq(choice_label)].copy()
        choice_color = choice_rows["hard_off_pa_choice_color"].iloc[0]
        ax.scatter(
            choice_rows["standby_8w_far_load_mbps"],
            choice_rows["standby_8w_mid_load_mbps"],
            s=68,
            alpha=0.85,
            color=choice_color,
            edgecolor="white",
            linewidth=0.7,
            label=choice_label,
        )

    ax.set_xlabel("Standby 8W load from >= 500 m users (Mbps)")
    ax.set_ylabel("Standby 8W load from 300-499 m users (Mbps)")
    ax.set_title("Standby bins that assign 8W, coloured by the hard-off bin outcome")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig, ax


def plot_day_slot_pressure(bin_table_all: pd.DataFrame, *, scenario_order: Iterable[str]):
    scenario_order = list(scenario_order)
    day_bin_count = int(bin_table_all["bin_index"].max()) + 1
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.7, 1.1]})

    for scenario_label in scenario_order:
        scenario_bins = (
            bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)]
            .sort_values("bin_index")
            .reset_index(drop=True)
        )
        axes[0].plot(
            scenario_bins["bin_index"],
            scenario_bins["used_slots"],
            linewidth=1.9,
            marker="o",
            markersize=2.8,
            label=str(scenario_label),
        )
        axes[1].plot(
            scenario_bins["bin_index"],
            scenario_bins["unused_slots"],
            linewidth=1.9,
            marker="o",
            markersize=2.8,
            label=str(scenario_label),
        )

    axes[0].set_ylabel("Used slots")
    axes[0].set_title("Slot pressure over the day")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(frameon=True, ncol=min(len(scenario_order), 3))

    axes[1].set_xlabel("Quarter-hour bin index")
    axes[1].set_ylabel("Unused slots")
    axes[1].set_title("Remaining slot headroom")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)

    plt.tight_layout()
    return fig, axes


def plot_cumulative_day_energy(
    bin_table_all: pd.DataFrame,
    *,
    scenario_order: Iterable[str],
    bin_duration_s: float,
):
    scenario_order = list(scenario_order)
    day_bin_count = int(bin_table_all["bin_index"].max()) + 1
    fig, ax = plt.subplots(figsize=(12, 4.8))

    for scenario_label in scenario_order:
        scenario_bins = (
            bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)]
            .sort_values("bin_index")
            .reset_index(drop=True)
        )
        cumulative_energy_wh = np.nancumsum(scenario_bins["dc_total_w"].to_numpy(dtype=float) * float(bin_duration_s) / 3600.0)
        ax.plot(
            scenario_bins["bin_index"],
            cumulative_energy_wh,
            linewidth=2.1,
            label=str(scenario_label),
        )

    _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title("Scenario differences accumulate across the day")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def plot_scenario_delta(
    bin_table_all: pd.DataFrame,
    *,
    base_scenario: str,
    compare_scenario: str,
    value_column: str = "dc_total_w",
    ylabel: str = "Power difference (W)",
    title: str = "Per-bin total power difference",
):
    base = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(base_scenario), ["bin_index", value_column]]
        .rename(columns={value_column: "base_value"})
    )
    compare = (
        bin_table_all.loc[bin_table_all["scenario_label"].eq(compare_scenario), ["bin_index", value_column]]
        .rename(columns={value_column: "compare_value"})
    )
    delta_table = base.merge(compare, on="bin_index", how="inner")
    day_bin_count = int(delta_table["bin_index"].max()) + 1

    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(
        delta_table["bin_index"],
        delta_table["compare_value"] - delta_table["base_value"],
        width=0.9,
        color="#4c78a8",
        alpha=0.75,
    )
    _set_day_bin_boundaries(ax, day_bin_count=day_bin_count)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax
