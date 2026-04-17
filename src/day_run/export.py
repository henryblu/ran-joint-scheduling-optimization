from __future__ import annotations

"""Build and write the authoritative JSON export for one day-run simulation.

Keep the export layer as plain dict builders rather than a second schema stack.
The day-run JSON is the public artifact; these helpers only shape the already
resolved run results into that file format.
"""

import json
from pathlib import Path

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from configs.day_run import DAY_RUN_RESULT_FILENAME
from models import PASwitchPolicy
from models.day_run import BinRunResult, DayRunConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_day_run_result(
    *,
    config: DayRunConfig,
    user_tables_by_bin: dict[int, pd.DataFrame],
    bin_results: list[BinRunResult],
) -> None:
    """Write one authoritative JSON artifact for the completed day run."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    export_document = build_day_run_result_document(
        config=config,
        user_tables_by_bin=user_tables_by_bin,
        bin_results=bin_results,
    )
    with (config.output_dir / DAY_RUN_RESULT_FILENAME).open("w", encoding="utf-8") as output_file:
        json.dump(export_document, output_file, indent=2)


def build_day_run_result_document(
    *,
    config: DayRunConfig,
    user_tables_by_bin: dict[int, pd.DataFrame],
    bin_results: list[BinRunResult],
) -> dict[str, object]:
    """Build the hierarchical export document from demand rows and lean bin results."""

    result_by_bin = {int(result.bin_index): result for result in bin_results}
    pa_catalog = build_pa_catalog(SINGLE_USER_SEARCH_CONFIG.pa_data_csv)
    allowed_pa_ids = set(range(len(pa_catalog)))
    if config.switch_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        allowed_pa_ids = {
            pa_id
            for pa_id, pa in enumerate(pa_catalog)
            if str(pa.scenario_label) == "8W PA"
        }
        if not allowed_pa_ids and pa_catalog:
            max_p_max_w = max(float(pa.p_max_w) for pa in pa_catalog)
            allowed_pa_ids = {
                pa_id
                for pa_id, pa in enumerate(pa_catalog)
                if float(pa.p_max_w) == max_p_max_w
            }

    # Iterate over the projected bin tables so the export preserves the day-run
    # bin ordering even when the worker pool finishes bins out of order.
    bins = [
        _build_bin_document(
            bin_index=int(bin_index),
            user_table=user_table,
            result=result_by_bin[int(bin_index)],
        )
        for bin_index, user_table in user_tables_by_bin.items()
    ]
    return {
        "schema_version": "day_run_result_v1",
        "run": {
            "switch_policy": str(config.switch_policy.value),
            "load_curve_csv": _serialize_export_path(config.load_curve_csv),
            "day_bin_count": int(config.session_generation_config.day_bin_count),
            "bin_duration_s": float(config.session_generation_config.bin_duration_s),
        },
        "pa_lookup": [
            {
                "pa_id": int(pa_id),
                "pa_label": str(pa.scenario_label),
                "pa_name": str(pa.pa_name),
            }
            for pa_id, pa in enumerate(pa_catalog)
            if pa_id in allowed_pa_ids
        ],
        "bins": bins,
    }


def _build_bin_document(
    *,
    bin_index: int,
    user_table: pd.DataFrame,
    result: BinRunResult,
) -> dict[str, object]:
    """Build one bin-level export block from the demand rows and scheduler result."""

    # Preserve the original demand rows alongside the scheduler outcome so the
    # exported file explains both what was requested and what was scheduled.
    demand_users = [
        {
            "user_id": int(row.user_id),
            "distance_m": float(row.distance_m),
            "required_rate_bps": float(row.required_rate_bps),
        }
        for row in user_table.sort_values("user_id").itertuples(index=False)
    ]
    return {
        "bin_index": int(bin_index),
        "demand": {
            "user_count": int(len(user_table)),
            "requested_rate_sum_bps": float(user_table["required_rate_bps"].sum()),
            "users": demand_users,
        },
        "status": str(result.status),
        "timings_s": {
            "single_user": float(result.single_user_elapsed_s),
            "joint": float(result.joint_elapsed_s),
            "total": float(result.total_elapsed_s),
        },
        "schedule": None if result.best_schedule is None else _build_schedule_document(result.best_schedule),
    }


def _build_schedule_document(best_schedule: dict[str, object]) -> dict[str, object]:
    """Build the solved-bin schedule block from the public scheduler payload."""

    # Re-map the scheduler row names onto the slightly more descriptive export
    # names without widening the scheduler's own public contract.
    selected_allocations = [
        _build_selected_allocation(row)
        for row in best_schedule["rows"]
    ]
    active_dc_total_w = float(sum(allocation["p_dc_avg_frame_w"] for allocation in selected_allocations))
    dc_total_w = float(best_schedule["schedule_p_dc_total_avg_frame_w"])
    return {
        "slot_total": int(best_schedule["slot_total"]),
        "unused_slots": int(best_schedule["unused_slots"]),
        "delivered_rate_sum_bps": float(best_schedule["total_rate_bps"]),
        "power_w": {
            "dc_total": dc_total_w,
            "dc_active": active_dc_total_w,
            "dc_inactive": float(max(0.0, dc_total_w - active_dc_total_w)),
            "rf_total": float(best_schedule["schedule_p_out_total_avg_frame_w"]),
        },
        "selected_allocations": selected_allocations,
    }


def _build_selected_allocation(row: dict[str, object]) -> dict[str, object]:
    """Derive one export allocation block from the selected active row and slot count."""

    n_slots = int(row["n_slots"])
    slot_share = float(n_slots) / float(SINGLE_USER_SEARCH_CONFIG.frame_n_slots)
    frame_duration_s = float(SINGLE_USER_SEARCH_CONFIG.frame_n_slots) * float(SINGLE_USER_SEARCH_CONFIG.t_slot_s)
    return {
        "user_id": int(row["user_id"]),
        "pa_id": int(row["pa_id"]),
        "n_prb": int(row["n_prb"]),
        "layers": int(row["layers"]),
        "mcs": int(row["mcs"]),
        "n_slots": n_slots,
        "delivered_rate_bps": float(n_slots * float(row["bits_per_slot"]) / frame_duration_s),
        "p_dc_avg_frame_w": float(slot_share * float(row["p_dc_active_w"])),
        "p_out_avg_frame_w": float(slot_share * float(row["p_out_total_w"])),
    }


def _serialize_export_path(path: Path) -> str:
    """Serialize one export path relative to the repository when possible."""

    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved_path)


__all__ = [
    "build_day_run_result_document",
    "write_day_run_result",
]
