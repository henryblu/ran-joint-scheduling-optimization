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

def build_single_user_deployment(link_constants, phy_constants, distance_m, *, path_loss_db=None):
    """Build a deployment object from user-level scenario inputs."""
    resolved_path_loss_db = (
        PathLossModel(
            fc_hz=link_constants["fc_hz"],
            model=link_constants.get("pl_model", "fspl"),
            g_tx_db=link_constants.get("g_tx_db", 0.0),
            g_rx_db=link_constants.get("g_rx_db", 0.0),
            shadow_margin_db=link_constants.get("shadow_margin_db", 0.0),
            h_bs_m=link_constants.get("h_bs_m", 10.0),
            h_ut_m=link_constants.get("h_ut_m", 1.5),
        ).effective_path_loss_db(distance_m)
        if path_loss_db is None
        else float(path_loss_db)
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
    """Build the derived TDMA/system view from canonical radio config."""
    link = model_preset.link
    phy = model_preset.phy
    scheduler = model_preset.scheduler

    frame_slots = int(round(10e-3 / float(phy.t_slot_s)))
    total_slots = int((frame_slots // int(tdd_config.tdd_pattern_slots)) * (int(tdd_config.tdd_pattern_slots) - int(tdd_config.ul_slots)))

    return {
        "fc_hz": float(link.fc_hz),
        "channel_bw_hz": float(phy.channel_bw_hz),
        "bandwidth_space_hz": tuple(float(v) for v in scheduler.bandwidth_space_hz),
        "total_prbs": int(float(phy.channel_bw_hz) // (12.0 * float(phy.delta_f_hz))),
        "frame_slots": int(frame_slots),
        "tdd_pattern_slots": int(tdd_config.tdd_pattern_slots),
        "ul_slots": int(tdd_config.ul_slots),
        "total_slots": int(total_slots),
        "delta_f_hz": float(phy.delta_f_hz),
        "g_tx_db": float(link.g_tx_db),
        "g_rx_db": float(link.g_rx_db),
        "noise_density_dbm_per_hz": float(link.n0_dbm_per_hz),
        "noise_figure_db": float(link.lna_noise_figure_db),
        "impl_loss_db": float(phy.l_impl_db),
        "mi_n_samples": int(phy.mi_n_samples),
        "n_dmrs_sym": int(phy.n_dmrs_sym),
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


def build_multi_user_runtime_cfg(runtime_config):
    """Convert runtime policy into notebook-ready values."""
    return {
        "switch_policy": runtime_config.switch_policy,
        "max_configs_per_user": int(runtime_config.max_configs_per_user),
        "max_schedule_windows": int(runtime_config.max_schedule_windows),
    }


__all__ = [
    "build_model_inputs",
    "build_single_user_deployment",
    "build_multi_user_runtime_cfg",
    "build_multi_user_system_cfg",
]
