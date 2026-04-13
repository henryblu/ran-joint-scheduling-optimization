"""Shared radio defaults, immutable presets, and source references."""

from models import RadioConfig, freeze_mcs_table

from .pa import DEFAULT_PA_DATA_CSV


DEFAULT_NR_MCS_TABLE = {
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


COMMON_RADIO_CONFIG = RadioConfig(
    pl_model="umi_sc_nlos",
    fc_hz=3.5e9,
    g_tx_db=8.0,
    g_rx_db=0.0,
    n0_dbm_per_hz=-174.0,
    lna_noise_figure_db=5.0,
    shadow_margin_db=4.0,
    h_bs_m=10.0,
    h_ut_m=1.5,
    channel_bw_hz=100e6,
    l_impl_db=3.0,
    mi_n_samples=1500,
    papr_db=8.0,
    g_phi=1.0,
    sigma_phi2=0.0,
    sigma_q2=0.0,
    n_dmrs_sym=2,
    n_guard_sym=1,
    n_ul_sym=3,
    n_sym_data=10,
    n_sym_total=14,
    dft_size_N=4096,
    t_slot_s=0.5e-3,
    n_tx_chains=4,
    use_psd_constraint=True,
    psd_max_w_per_hz=8e-6,
    delta_f_hz=30e3,
    frame_n_slots=20,

    # Spaces for parameter sweeps inside one fixed 100 MHz carrier.
    layers_space=(1, 2, 3, 4),
    mcs_space=tuple(range(0, 29)),
    prb_step=5,
    mcs_table=freeze_mcs_table(DEFAULT_NR_MCS_TABLE),
    pa_data_csv=DEFAULT_PA_DATA_CSV,
)


SINGLE_USER_SEARCH_CONFIG = COMMON_RADIO_CONFIG
MULTI_USER_TDMA_CONFIG = COMMON_RADIO_CONFIG


def get_scenario_config(name: str):
    """Return the shared static radio config for a known scenario alias."""

    if str(name) in {"single_user_search", "multi_user_tdma"}:
        return COMMON_RADIO_CONFIG
    raise KeyError(f"Unknown radio scenario config: {name}")


__all__ = [
    "COMMON_RADIO_CONFIG",
    "DEFAULT_NR_MCS_TABLE",
    "MULTI_USER_TDMA_CONFIG",
    "SINGLE_USER_SEARCH_CONFIG",
    "get_scenario_config",
]
