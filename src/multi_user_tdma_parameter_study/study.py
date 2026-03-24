import pandas as pd

from pa_models import build_pa_catalog, build_pa_characteristics_table
from radio_configs import MULTI_USER_TDMA_CONFIG
from radio_models import PathLossModel, build_resolved_fingerprint
from single_user_search.models import SearchSpace

from .models import MultiUserSystemConfig, MultiUserTdmaScenario, MultiUserTdmaStudyResult
from .presets import MULTI_USER_TDMA_PRESET
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
    1. Normalize the notebook-facing user table into one resolved path-loss row per user.
    2. Resolve the canonical active-search radio state once at the study boundary.
    3. Build the shared slot-level system view and immutable PA catalog once.
    4. Return the compact scenario reused by search and reporting helpers.
    """

    resolved_preset = MULTI_USER_TDMA_PRESET
    active_search_model_inputs = MULTI_USER_TDMA_CONFIG
    active_search_shape = _build_search_space(active_search_model_inputs)
    scenario_user_table = _resolve_scenario_user_table(
        user_table,
        link_constants=active_search_model_inputs,
    )
    return MultiUserTdmaScenario(
        user_table=scenario_user_table,
        preset=resolved_preset,
        system_cfg=_build_system_cfg(active_search_model_inputs, resolved_preset.tdd),
        pa_catalog=tuple(build_pa_catalog(active_search_model_inputs.pa_data_csv)),
        active_search_model_inputs=active_search_model_inputs,
        active_search_shape=active_search_shape,
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
        system_cfg=scenario.system_cfg,
        model_inputs=scenario.active_search_model_inputs,
        search_shape=scenario.active_search_shape,
        pa_catalog=scenario.pa_catalog,
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
        frame_slot_count=int(scenario.system_cfg.frame_slots),
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
        frame_slot_count=int(scenario.system_cfg.frame_slots),
    )
    user_candidate_review_tables = build_user_candidate_review_tables(user_candidate_spaces)
    return MultiUserTdmaStudyResult(
        repeated_frames=int(repeated_frames),
        repeated_period_slots=int(repeated_frames * int(scenario.system_cfg.frame_slots)),
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
                "frame_slots": int(scenario.system_cfg.frame_slots),
                "slot_duration_ms": 1e3 * float(scenario.system_cfg.t_slot_s),
                "frame_duration_ms": 1e3 * float(scenario.system_cfg.frame_slots * scenario.system_cfg.t_slot_s),
            }
        ]
    )
    active_search_summary = pd.DataFrame(
        [
            {
                "bandwidth_space_hz": tuple(float(v) for v in scenario.active_search_shape.bandwidth_space_hz),
                "frame_slots": (int(scenario.system_cfg.frame_slots),),
                "layers_space": tuple(int(v) for v in scenario.active_search_shape.layers_space),
                "mcs_min": int(min(scenario.active_search_shape.mcs_space)),
                "mcs_max": int(max(scenario.active_search_shape.mcs_space)),
                "prb_step": int(scenario.active_search_shape.prb_step),
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

    scenario_user_table = user_table.copy()
    scenario_user_table["distance_m"] = scenario_user_table["distance_m"].astype(float)
    scenario_user_table["path_loss_db"] = _resolve_path_loss_db_values(
        link_constants,
        scenario_user_table["distance_m"].tolist(),
    )
    return scenario_user_table[["user_id", "distance_m", "required_rate_bps", "path_loss_db"]]


def _build_search_space(model_inputs):
    n_slots_on_space = tuple(range(1, int(model_inputs.n_slots_win) + 1))
    return SearchSpace(
        config=model_inputs,
        bandwidth_space_hz=model_inputs.bandwidth_space_hz,
        n_slots_on_space=n_slots_on_space,
        layers_space=model_inputs.layers_space,
        mcs_space=model_inputs.mcs_space,
        prb_step=model_inputs.prb_step,
        fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
        use_cache=True,
    )


def _resolve_path_loss_db_values(config, distance_values_m):
    """Resolve one concrete path-loss value per user distance for the TDMA study."""

    path_loss_model = PathLossModel(
        fc_hz=config.fc_hz,
        model=config.pl_model,
        g_tx_db=config.g_tx_db,
        g_rx_db=config.g_rx_db,
        shadow_margin_db=config.shadow_margin_db,
        h_bs_m=config.h_bs_m,
        h_ut_m=config.h_ut_m,
    )
    return [
        path_loss_model.effective_path_loss_db(float(distance_m))
        for distance_m in distance_values_m
    ]


def _build_system_cfg(config, tdd_config):
    """Build the TDMA-owned mixed-slot system view from radio config and TDD pattern."""

    _validate_mixed_slot_pattern(config, tdd_config)
    return MultiUserSystemConfig(
        fc_hz=config.fc_hz,
        channel_bw_hz=config.channel_bw_hz,
        bandwidth_space_hz=config.bandwidth_space_hz,
        total_prbs=int(config.channel_bw_hz // (12.0 * config.delta_f_hz)),
        frame_slots=config.n_slots_win,
        slot_dl_symbols=tdd_config.n_dl_symbols,
        slot_guard_symbols=tdd_config.n_guard_symbols,
        slot_ul_symbols=tdd_config.n_ul_symbols,
        slot_payload_symbols=tdd_config.n_dl_symbols - config.n_dmrs_sym,
        total_slots=config.n_slots_win,
        delta_f_hz=config.delta_f_hz,
        g_tx_db=config.g_tx_db,
        g_rx_db=config.g_rx_db,
        noise_density_dbm_per_hz=config.n0_dbm_per_hz,
        noise_figure_db=config.lna_noise_figure_db,
        impl_loss_db=config.l_impl_db,
        mi_n_samples=config.mi_n_samples,
        n_dmrs_sym=config.n_dmrs_sym,
        n_guard_sym=config.n_guard_sym,
        n_ul_sym=config.n_ul_sym,
        n_sym_data=config.n_sym_data,
        n_sym_total=config.n_sym_total,
        dft_size_N=config.dft_size_N,
        t_slot_s=config.t_slot_s,
        n_tx_chains=config.n_tx_chains,
        use_psd_constraint=config.use_psd_constraint,
        psd_max_w_per_hz=config.psd_max_w_per_hz,
        papr_db=config.papr_db,
        g_phi=config.g_phi,
        sigma_phi2=config.sigma_phi2,
        sigma_q2=config.sigma_q2,
        layers_space=config.layers_space,
        mcs_space=config.mcs_space,
        prb_step=config.prb_step,
    )


def _validate_mixed_slot_pattern(config, tdd_config):
    """Reject inconsistent mixed-slot TDD definitions before TDMA study code uses them."""

    if int(tdd_config.n_dl_symbols) != int(config.n_sym_data):
        raise ValueError("TDD DL-symbol count must match n_sym_data.")
    if int(tdd_config.n_guard_symbols) != int(config.n_guard_sym):
        raise ValueError("TDD guard-symbol count must match n_guard_sym.")
    if int(tdd_config.n_ul_symbols) != int(config.n_ul_sym):
        raise ValueError("TDD UL-symbol count must match n_ul_sym.")
    if int(tdd_config.n_dl_symbols) + int(tdd_config.n_guard_symbols) + int(tdd_config.n_ul_symbols) != int(config.n_sym_total):
        raise ValueError("TDD slot symbols must sum to n_sym_total.")
    if int(config.n_dmrs_sym) > int(tdd_config.n_dl_symbols):
        raise ValueError("DMRS symbols cannot exceed the DL-symbol region in one slot.")
