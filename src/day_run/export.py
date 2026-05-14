from __future__ import annotations

"""Build and write the authoritative JSON export for one day-run simulation.

Keep the export layer as plain dict builders rather than a second schema stack.
The day-run JSON is the public artifact; these helpers only shape the already
resolved run results into that file format.
"""

import json
import math
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
        "schema_version": "day_run_result_v2",
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
        "schedule": None if result.schedule_result is None else _build_schedule_document(result.schedule_result),
    }


def _build_schedule_document(schedule_result) -> dict[str, object]:
    """Build the solved-bin schedule block directly from the shared scheduler result."""

    return {
        "scheduler_mode": str(schedule_result.scheduler_mode.value),
        "frame_summary": {
            "frame_n_slots": int(len(schedule_result.slot_schedules)),
            "t_slot_s": float(SINGLE_USER_SEARCH_CONFIG.t_slot_s),
            "prb_max": int(
                math.floor(
                    float(SINGLE_USER_SEARCH_CONFIG.channel_bw_hz)
                    / (12.0 * float(SINGLE_USER_SEARCH_CONFIG.delta_f_hz))
                )
            ),
            "n_tx_chains": int(SINGLE_USER_SEARCH_CONFIG.n_tx_chains),
        },
        "power_summary": {
            "frame_energy_j": float(schedule_result.power_summary.frame_energy_j),
            "average_frame_dc_power_w": float(schedule_result.power_summary.average_frame_dc_power_w),
            "active_energy_j": float(schedule_result.power_summary.active_energy_j),
            "inactive_energy_j": float(schedule_result.power_summary.inactive_energy_j),
            "average_frame_rf_output_w": float(schedule_result.power_summary.average_frame_rf_output_w),
        },
        "user_summaries": [
            {
                "user_id": int(user_summary.user_id),
                "required_bits": float(user_summary.required_bits),
                "delivered_bits": float(user_summary.delivered_bits),
                "required_rate_bps": float(user_summary.required_rate_bps),
                "delivered_rate_bps": float(user_summary.delivered_rate_bps),
                "satisfied": bool(user_summary.satisfied),
            }
            for user_summary in schedule_result.user_summaries
        ],
        "slot_schedules": [
            {
                "slot_index": int(slot.slot_index),
                "active": bool(slot.active),
                "pa_id": None if slot.pa_id is None else int(slot.pa_id),
                "used_prbs": int(slot.used_prbs),
                "aggregate_p_out_w": float(slot.aggregate_p_out_w),
                "dc_power_w": float(slot.dc_power_w),
                "allocations": [
                    {
                        "user_id": int(allocation.user_id),
                        "pa_id": int(allocation.pa_id),
                        "n_prb": int(allocation.n_prb),
                        "layers": int(allocation.layers),
                        "mcs": int(allocation.mcs),
                        "bits_per_slot": float(allocation.bits_per_slot),
                        "p_out_total_w": float(allocation.p_out_total_w),
                        "p_dc_active_w": float(allocation.p_dc_active_w),
                    }
                    for allocation in slot.allocations
                ],
            }
            for slot in schedule_result.slot_schedules
        ],
        "solver_details": dict(schedule_result.solver_details),
    }


def _serialize_export_path(path: Path) -> str:
    """Serialize one export path relative to the repository when possible."""

    resolved_path = Path(path).resolve()
    if resolved_path.is_relative_to(REPO_ROOT):
        return str(resolved_path.relative_to(REPO_ROOT))
    return str(resolved_path)


__all__ = [
    "build_day_run_result_document",
    "write_day_run_result",
]
