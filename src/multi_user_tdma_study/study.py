import pandas as pd

from radio_core import (
    MULTI_USER_TDMA_PRESET,
    build_pa_catalog,
    build_multi_user_system_cfg,
    build_pa_characteristics_table,
    resolve_path_loss_db_values,
)

from .models import MultiUserTdmaScenario, MultiUserTdmaStudyResult
from .user_space import (
    build_active_candidate_summary_df,
    build_user_candidate_review_tables,
    build_user_candidate_spaces,
    enumerate_user_active_operating_tables,
    resolve_repeated_frame_requirement,
)


def build_multi_user_tdma_scenario(user_table):
    """Build the notebook-facing multi-user TDMA study scenario.

    Steps:
    1. Keep the notebook user table as the study-owned scenario table.
    2. Resolve the shared slot-level system view from the canonical preset.
    3. Load the preset PA catalog once for notebook summaries.
    4. Return the compact scenario reused by search and reporting helpers.
    """
    resolved_preset = MULTI_USER_TDMA_PRESET
    scenario_user_table = _resolve_scenario_user_table(
        user_table,
        link_constants=resolved_preset.model.link.__dict__,
    )
    resolved_system_cfg = build_multi_user_system_cfg(resolved_preset.model, resolved_preset.tdd)
    return MultiUserTdmaScenario(
        user_table=scenario_user_table,
        preset=resolved_preset,
        system_cfg=resolved_system_cfg,
        pa_catalog=build_pa_catalog(str(resolved_preset.model.pa_data_csv)),
    )


def run_multi_user_tdma_scenario(
    scenario,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Run the pre-scheduler multi-user TDMA study flow for one prepared scenario.

    Steps:
    1. Enumerate the exact active operating-point catalog for every user deployment.
    2. Resolve the minimum repeated-frame count required by slot quantization.
    3. Build the exact per-user TDMA candidate spaces for the resolved repeated-frame count.
    4. Return notebook-facing tables without imposing heuristic config-count cuts.
    """
    
    active_candidate_tables = enumerate_user_active_operating_tables(
        scenario.user_table,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )
    active_summary_df = build_active_candidate_summary_df(
        scenario.user_table,
        active_candidate_tables,
    )
    max_repeated_frames = int(scenario.preset.runtime.max_schedule_windows)
    repeated_frame_requirement = resolve_repeated_frame_requirement(
        scenario.user_table,
        active_candidate_tables,
        frame_slot_count=int(scenario.system_cfg["frame_slots"]),
        max_repeated_frames=max_repeated_frames,
    )
    status = str(repeated_frame_requirement["status"])
    if status == "missing_active_operating_points":
        raise RuntimeError(
            f"No feasible active operating points were found for user {int(repeated_frame_requirement['user_id'])}."
        )
    if status == "user_target_exceeds_active_rate":
        raise RuntimeError(
            f"User {int(repeated_frame_requirement['user_id'])} requires a higher average rate than any active operating point can deliver."
        )
    if status == "overloaded":
        raise RuntimeError(
            "The requested average rates are infeasible within one frame budget: "
            f"exact frame-share lower bound = {float(repeated_frame_requirement['exact_frame_share_sum']):.3f} > 1.0."
        )
    if status != "ok":
        raise RuntimeError(
            "Could not resolve a finite repeated-frame count within "
            f"{max_repeated_frames} repeated frames."
        )

    repeated_frames = int(repeated_frame_requirement["min_repeated_frames"])
    user_candidate_spaces, user_candidate_summary_df = build_user_candidate_spaces(
        scenario.user_table,
        active_candidate_tables,
        repeated_frames=repeated_frames,
        frame_slot_count=int(scenario.system_cfg["frame_slots"]),
    )
    user_candidate_review_tables = build_user_candidate_review_tables(user_candidate_spaces)
    return MultiUserTdmaStudyResult(
        repeated_frames=int(repeated_frames),
        repeated_period_slots=int(repeated_frames * int(scenario.system_cfg["frame_slots"])),
        active_candidate_tables=active_candidate_tables,
        active_summary_df=active_summary_df,
        frame_share_summary_df=repeated_frame_requirement.get("share_rows", pd.DataFrame()).copy(),
        user_candidate_spaces=user_candidate_spaces,
        user_candidate_review_tables=user_candidate_review_tables,
        user_candidate_summary_df=user_candidate_summary_df,
    )


def summarize_multi_user_tdma_scenario(scenario):
    """Return the notebook-facing tables that describe one prepared multi-user scenario."""
    system_summary = pd.DataFrame(
        [
            {
                "n_users": int(len(scenario.user_table)),
                "frame_slots": int(scenario.system_cfg["frame_slots"]),
                "slot_duration_ms": 1e3 * float(scenario.system_cfg["t_slot_s"]),
                "frame_duration_ms": 1e3
                * float(scenario.system_cfg["frame_slots"] * scenario.system_cfg["t_slot_s"]),
            }
        ]
    )
    active_search_summary = pd.DataFrame(
        [
            {
                "bandwidth_space_hz": tuple(float(v) for v in scenario.system_cfg["bandwidth_space_hz"]),
                "frame_slots": (int(scenario.system_cfg["frame_slots"]),),
                "layers_space": tuple(int(v) for v in scenario.system_cfg["layers_space"]),
                "mcs_min": int(min(scenario.system_cfg["mcs_space"])),
                "mcs_max": int(max(scenario.system_cfg["mcs_space"])),
                "prb_step": int(scenario.system_cfg["prb_step"]),
            }
        ]
    )
    return {
        "user_summary": scenario.user_table.copy(),
        "system_summary": system_summary,
        "pa_characteristics": build_pa_characteristics_table(scenario.pa_catalog),
        "active_search_summary": active_search_summary,
    }


def search_user_candidate_spaces(
    user_table,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Convenience wrapper that returns only the exact per-user TDMA candidate spaces."""

    scenario = build_multi_user_tdma_scenario(user_table)
    study_result = run_multi_user_tdma_scenario(
        scenario,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )
    return study_result.user_candidate_spaces


def _resolve_scenario_user_table(user_table, *, link_constants):
    """Resolve the scenario table into one concrete path-loss row per user."""

    if "path_loss_db" in user_table.columns:
        raise ValueError("user_table must not include path_loss_db; path loss is derived from distance in radio_core.")

    scenario_user_table = user_table.copy()
    scenario_user_table["distance_m"] = scenario_user_table["distance_m"].astype(float)
    scenario_user_table["path_loss_db"] = resolve_path_loss_db_values(
        link_constants,
        scenario_user_table["distance_m"].tolist(),
    )
    return scenario_user_table[["user_id", "distance_m", "required_rate_bps", "path_loss_db"]]
