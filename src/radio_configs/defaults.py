"""Shared static radio defaults and source references."""

from pathlib import Path

from .mcs_tables import DEFAULT_NR_MCS_TABLE
from .types import RadioConfig, freeze_mcs_table


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PA_DATA_CSV = str(
    REPO_ROOT / "PA models" / "3.5Ghz_pas" / "4W_8W_NR_combined_NR_carrier.csv"
)

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
    psd_max_w_per_hz=8e-7,
    delta_f_hz=30e3,
    n_slots_win=20,

    # Spaces for parameter sweeps.
    bandwidth_space_hz=(100e6, 50e6),
    layers_space=(1, 2, 3, 4),
    mcs_space=tuple(range(0, 29)),
    prb_step=5,
    mcs_table=freeze_mcs_table(DEFAULT_NR_MCS_TABLE),
    pa_data_csv=DEFAULT_PA_DATA_CSV,
)

__all__ = [
    "COMMON_RADIO_CONFIG",
    "DEFAULT_PA_DATA_CSV",
]
