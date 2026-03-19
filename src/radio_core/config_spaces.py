from copy import deepcopy
from pathlib import Path

from .pa_models import PASwitchPolicy


COMMON_LINK_CONSTANTS = {
    "pl_model": "umi_sc_nlos",
    "fc_hz": 3.5e9,
    "g_tx_db": 8.0,
    "g_rx_db": 0.0,
    "n0_dbm_per_hz": -174.0,
    "lna_noise_figure_db": 5.0,
    "shadow_margin_db": 4.0,
}

COMMON_PHY_CONSTANTS = {
    "channel_bw_hz": 100e6,
    "l_impl_db": 3.0,
    "mi_n_samples": 1500,
    "papr_db": 8.0,
    "g_phi": 1.0,
    "sigma_phi2": 0.0,
    "sigma_q2": 0.0,
    "n_dmrs_sym": 2,
    "n_sym_data": 12,
    "n_sym_total": 14,
    "dft_size_N": 4096,
    "t_slot_s": 0.5e-3,
    "n_tx_chains": 4,
    "use_psd_constraint": True,
    "psd_max_w_per_hz": 8e-7,
    "delta_f_hz": 30e3,
}

COMMON_SCHEDULER_SWEEP = {
    "bandwidth_space_hz": (100e6, 50e6),
    "layers_space": [1, 2, 4],
    "mcs_space": list(range(0, 29)),
    "prb_step": 5,
}

DEFAULT_MCS_TABLE = {
    0: {"qm": 2, "r": 120, "eta": 0.2344},
    1: {"qm": 2, "r": 157, "eta": 0.3066},
    2: {"qm": 2, "r": 193, "eta": 0.3770},
    3: {"qm": 2, "r": 251, "eta": 0.4902},
    4: {"qm": 2, "r": 308, "eta": 0.6016},
    5: {"qm": 2, "r": 379, "eta": 0.7402},
    6: {"qm": 2, "r": 449, "eta": 0.8770},
    7: {"qm": 2, "r": 526, "eta": 1.0273},
    8: {"qm": 2, "r": 602, "eta": 1.1758},
    9: {"qm": 2, "r": 679, "eta": 1.3262},
    10: {"qm": 4, "r": 340, "eta": 1.3281},
    11: {"qm": 4, "r": 378, "eta": 1.4766},
    12: {"qm": 4, "r": 434, "eta": 1.6953},
    13: {"qm": 4, "r": 490, "eta": 1.9141},
    14: {"qm": 4, "r": 553, "eta": 2.1602},
    15: {"qm": 4, "r": 616, "eta": 2.4063},
    16: {"qm": 4, "r": 658, "eta": 2.5703},
    17: {"qm": 6, "r": 438, "eta": 2.5664},
    18: {"qm": 6, "r": 466, "eta": 2.7305},
    19: {"qm": 6, "r": 517, "eta": 3.0293},
    20: {"qm": 6, "r": 567, "eta": 3.3223},
    21: {"qm": 6, "r": 616, "eta": 3.6094},
    22: {"qm": 6, "r": 666, "eta": 3.9023},
    23: {"qm": 6, "r": 719, "eta": 4.2129},
    24: {"qm": 6, "r": 772, "eta": 4.5234},
    25: {"qm": 6, "r": 822, "eta": 4.8164},
    26: {"qm": 6, "r": 873, "eta": 5.1152},
    27: {"qm": 6, "r": 910, "eta": 5.3320},
    28: {"qm": 6, "r": 948, "eta": 5.5547},
}

DEFAULT_PA_DATA_CSV = str(Path("PA models") / "3.5Ghz_pas" / "4W_8W_NR_combined_NR_carrier.csv")


def _clone_mapping(mapping):
    return deepcopy(mapping)


def _build_base_notebook_config(*, n_slots_win):
    phy_constants = _clone_mapping(COMMON_PHY_CONSTANTS)
    phy_constants["n_slots_win"] = int(n_slots_win)
    return {
        "link_constants": _clone_mapping(COMMON_LINK_CONSTANTS),
        "phy_constants": phy_constants,
        "scheduler_sweep": _clone_mapping(COMMON_SCHEDULER_SWEEP),
        "mcs_table": _clone_mapping(DEFAULT_MCS_TABLE),
        "pa_data_csv": DEFAULT_PA_DATA_CSV,
    }


def get_single_user_resource_model_config():
    """Reviewer-facing single-user walkthrough preset."""
    return _build_base_notebook_config(n_slots_win=7)


def get_single_user_power_optimization_config():
    """Single-user sweep/optimization notebook preset."""
    return _build_base_notebook_config(n_slots_win=20)


def get_multi_user_tdma_config():
    """TDMA notebook preset with shared PHY space and notebook-local runtime knobs."""
    config = _build_base_notebook_config(n_slots_win=20)
    link_constants = config["link_constants"]
    phy_constants = config["phy_constants"]
    scheduler_sweep = config["scheduler_sweep"]

    frame_slots = int(round(10e-3 / phy_constants["t_slot_s"]))
    tdd_pattern_slots = 10
    ul_slots = 3
    total_slots = int((frame_slots // tdd_pattern_slots) * (tdd_pattern_slots - ul_slots))

    config["system_cfg"] = {
        "fc_hz": float(link_constants["fc_hz"]),
        "channel_bw_hz": float(phy_constants["channel_bw_hz"]),
        "bandwidth_space_hz": tuple(float(v) for v in scheduler_sweep["bandwidth_space_hz"]),
        "total_prbs": int(phy_constants["channel_bw_hz"] // (12.0 * phy_constants["delta_f_hz"])),
        "frame_slots": int(frame_slots),
        "tdd_pattern_slots": int(tdd_pattern_slots),
        "ul_slots": int(ul_slots),
        "total_slots": int(total_slots),
        "delta_f_hz": float(phy_constants["delta_f_hz"]),
        "g_tx_db": float(link_constants["g_tx_db"]),
        "g_rx_db": float(link_constants["g_rx_db"]),
        "noise_density_dbm_per_hz": float(link_constants["n0_dbm_per_hz"]),
        "noise_figure_db": float(link_constants["lna_noise_figure_db"]),
        "impl_loss_db": float(phy_constants["l_impl_db"]),
        "mi_n_samples": int(phy_constants["mi_n_samples"]),
        "n_dmrs_sym": int(phy_constants["n_dmrs_sym"]),
        "n_sym_data": int(phy_constants["n_sym_data"]),
        "n_sym_total": int(phy_constants["n_sym_total"]),
        "dft_size_N": int(phy_constants["dft_size_N"]),
        "t_slot_s": float(phy_constants["t_slot_s"]),
        "n_tx_chains": int(phy_constants["n_tx_chains"]),
        "use_psd_constraint": bool(phy_constants["use_psd_constraint"]),
        "psd_max_w_per_hz": float(phy_constants["psd_max_w_per_hz"]),
        "papr_db": float(phy_constants["papr_db"]),
        "g_phi": float(phy_constants["g_phi"]),
        "sigma_phi2": float(phy_constants["sigma_phi2"]),
        "sigma_q2": float(phy_constants["sigma_q2"]),
        "layers_space": list(scheduler_sweep["layers_space"]),
        "mcs_space": list(scheduler_sweep["mcs_space"]),
        "prb_step": int(scheduler_sweep["prb_step"]),
    }
    config["runtime"] = {
        "switch_policy": PASwitchPolicy.STANDBY,
        "max_configs_per_user": 300,
        "max_schedule_windows": 32,
    }
    return config
