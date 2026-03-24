from copy import deepcopy
from dataclasses import asdict

from .model_types import DeploymentParams
from .path_loss_models import PathLossModel


def build_model_inputs(model_preset):
    """Convert a frozen preset into notebook/engine-ready dictionaries."""
    return {
        "link_constants": asdict(model_preset.link),
        "phy_constants": asdict(model_preset.phy),
        "scheduler_sweep": {
            "bandwidth_space_hz": tuple(float(v) for v in model_preset.scheduler.bandwidth_space_hz),
            "layers_space": list(int(v) for v in model_preset.scheduler.layers_space),
            "mcs_space": list(int(v) for v in model_preset.scheduler.mcs_space),
            "prb_step": int(model_preset.scheduler.prb_step),
        },
        "mcs_table": deepcopy(model_preset.mcs_table),
        "pa_data_csv": str(model_preset.pa_data_csv),
    }


def resolve_path_loss_db(link_constants, distance_m):
    """Resolve one deployment's concrete path loss from distance and radio configuration."""

    return float(
        PathLossModel(
            fc_hz=link_constants["fc_hz"],
            model=link_constants.get("pl_model", "fspl"),
            g_tx_db=link_constants.get("g_tx_db", 0.0),
            g_rx_db=link_constants.get("g_rx_db", 0.0),
            shadow_margin_db=link_constants.get("shadow_margin_db", 0.0),
            h_bs_m=link_constants.get("h_bs_m", 10.0),
            h_ut_m=link_constants.get("h_ut_m", 1.5),
        ).effective_path_loss_db(distance_m)
    )


def resolve_path_loss_db_values(link_constants, distance_values_m):
    """Resolve one concrete path-loss value per distance entry."""

    return [
        resolve_path_loss_db(link_constants, distance_m=float(distance_m))
        for distance_m in distance_values_m
    ]


def build_single_user_deployment(link_constants, phy_constants, distance_m):
    """Build a deployment object from user-level scenario inputs."""
    resolved_path_loss_db = resolve_path_loss_db(
        link_constants,
        distance_m=float(distance_m),
    )
    return DeploymentParams(
        fc_hz=float(link_constants["fc_hz"]),
        channel_bw_hz=float(phy_constants["channel_bw_hz"]),
        distance_m=float(distance_m),
        path_loss_db=float(resolved_path_loss_db),
        g_tx_db=float(link_constants["g_tx_db"]),
        g_rx_db=float(link_constants["g_rx_db"]),
        n0_dbm_per_hz=float(link_constants["n0_dbm_per_hz"]),
        lna_noise_figure_db=float(link_constants["lna_noise_figure_db"]),
        l_impl_db=float(phy_constants["l_impl_db"]),
        mi_n_samples=int(phy_constants["mi_n_samples"]),
        n_dmrs_sym=int(phy_constants["n_dmrs_sym"]),
        n_guard_sym=int(phy_constants["n_guard_sym"]),
        n_ul_sym=int(phy_constants["n_ul_sym"]),
        dft_size_N=int(phy_constants["dft_size_N"]),
        n_slots_win=int(phy_constants["n_slots_win"]),
        t_slot_s=float(phy_constants["t_slot_s"]),
        n_sym_data=int(phy_constants["n_sym_data"]),
        n_sym_total=int(phy_constants["n_sym_total"]),
        use_psd_constraint=bool(phy_constants["use_psd_constraint"]),
        psd_max_w_per_hz=float(phy_constants["psd_max_w_per_hz"]),
        papr_db=float(phy_constants["papr_db"]),
        g_phi=float(phy_constants["g_phi"]),
        sigma_phi2=float(phy_constants["sigma_phi2"]),
        sigma_q2=float(phy_constants["sigma_q2"]),
        n_tx_chains=int(phy_constants["n_tx_chains"]),
    )


def build_multi_user_system_cfg(model_preset, tdd_config):
    """Build the mixed-slot TDMA/system view from canonical radio config.

    Steps:
    1. Resolve the frame length directly from the shared PHY window.
    2. Validate that the mixed-slot TDD pattern matches the PHY symbol accounting.
    3. Expose one schedulable slot per frame slot, with reduced DL payload carried in the PHY symbols.
    4. Return the notebook/module-facing system dictionary used by multi-user studies.
    """
    link = model_preset.link
    phy = model_preset.phy
    scheduler = model_preset.scheduler

    frame_slots = int(phy.n_slots_win)
    _validate_mixed_slot_pattern(phy, tdd_config)
    total_slots = int(frame_slots)

    return {
        "fc_hz": float(link.fc_hz),
        "channel_bw_hz": float(phy.channel_bw_hz),
        "bandwidth_space_hz": tuple(float(v) for v in scheduler.bandwidth_space_hz),
        "total_prbs": int(float(phy.channel_bw_hz) // (12.0 * float(phy.delta_f_hz))),
        "frame_slots": int(frame_slots),
        "slot_dl_symbols": int(tdd_config.n_dl_symbols),
        "slot_guard_symbols": int(tdd_config.n_guard_symbols),
        "slot_ul_symbols": int(tdd_config.n_ul_symbols),
        "slot_payload_symbols": int(tdd_config.n_dl_symbols - int(phy.n_dmrs_sym)),
        "total_slots": int(total_slots),
        "delta_f_hz": float(phy.delta_f_hz),
        "g_tx_db": float(link.g_tx_db),
        "g_rx_db": float(link.g_rx_db),
        "noise_density_dbm_per_hz": float(link.n0_dbm_per_hz),
        "noise_figure_db": float(link.lna_noise_figure_db),
        "impl_loss_db": float(phy.l_impl_db),
        "mi_n_samples": int(phy.mi_n_samples),
        "n_dmrs_sym": int(phy.n_dmrs_sym),
        "n_guard_sym": int(phy.n_guard_sym),
        "n_ul_sym": int(phy.n_ul_sym),
        "n_sym_data": int(phy.n_sym_data),
        "n_sym_total": int(phy.n_sym_total),
        "dft_size_N": int(phy.dft_size_N),
        "t_slot_s": float(phy.t_slot_s),
        "n_tx_chains": int(phy.n_tx_chains),
        "use_psd_constraint": bool(phy.use_psd_constraint),
        "psd_max_w_per_hz": float(phy.psd_max_w_per_hz),
        "papr_db": float(phy.papr_db),
        "g_phi": float(phy.g_phi),
        "sigma_phi2": float(phy.sigma_phi2),
        "sigma_q2": float(phy.sigma_q2),
        "layers_space": list(int(v) for v in scheduler.layers_space),
        "mcs_space": list(int(v) for v in scheduler.mcs_space),
        "prb_step": int(scheduler.prb_step),
    }


def _validate_mixed_slot_pattern(phy, tdd_config):
    """Reject inconsistent mixed-slot TDD definitions before study code uses them."""

    if int(tdd_config.n_dl_symbols) != int(phy.n_sym_data):
        raise ValueError("TDD DL-symbol count must match phy.n_sym_data.")
    if int(tdd_config.n_guard_symbols) != int(phy.n_guard_sym):
        raise ValueError("TDD guard-symbol count must match phy.n_guard_sym.")
    if int(tdd_config.n_ul_symbols) != int(phy.n_ul_sym):
        raise ValueError("TDD UL-symbol count must match phy.n_ul_sym.")
    if int(tdd_config.n_dl_symbols) + int(tdd_config.n_guard_symbols) + int(tdd_config.n_ul_symbols) != int(phy.n_sym_total):
        raise ValueError("TDD slot symbols must sum to phy.n_sym_total.")
    if int(phy.n_dmrs_sym) > int(tdd_config.n_dl_symbols):
        raise ValueError("DMRS symbols cannot exceed the DL-symbol region in one slot.")


def build_multi_user_runtime_cfg(runtime_config):
    """Convert runtime policy into notebook-ready values."""
    return {
        "switch_policy": runtime_config.switch_policy,
        "max_configs_per_user": int(runtime_config.max_configs_per_user),
        "max_schedule_windows": int(runtime_config.max_schedule_windows),
    }


__all__ = [
    "build_model_inputs",
    "resolve_path_loss_db",
    "resolve_path_loss_db_values",
    "build_single_user_deployment",
    "build_multi_user_runtime_cfg",
    "build_multi_user_system_cfg",
]
