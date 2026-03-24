from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkConstantsConfig:
    pl_model: str
    fc_hz: float
    g_tx_db: float
    g_rx_db: float
    n0_dbm_per_hz: float
    lna_noise_figure_db: float
    shadow_margin_db: float
    h_bs_m: float = 10.0
    h_ut_m: float = 1.5


@dataclass(frozen=True)
class PhyConstantsConfig:
    channel_bw_hz: float
    l_impl_db: float
    mi_n_samples: int
    papr_db: float
    g_phi: float
    sigma_phi2: float
    sigma_q2: float
    n_dmrs_sym: int
    n_guard_sym: int
    n_ul_sym: int
    n_sym_data: int
    n_sym_total: int
    dft_size_N: int
    t_slot_s: float
    n_tx_chains: int
    use_psd_constraint: bool
    psd_max_w_per_hz: float
    delta_f_hz: float
    n_slots_win: int

    def __post_init__(self):
        if int(self.n_slots_win) < 1:
            raise ValueError("PhyConstantsConfig requires n_slots_win >= 1.")
        if int(self.n_tx_chains) < 1:
            raise ValueError("PhyConstantsConfig requires n_tx_chains >= 1.")


@dataclass(frozen=True)
class SchedulerSpaceConfig:
    bandwidth_space_hz: tuple[float, ...]
    layers_space: tuple[int, ...]
    mcs_space: tuple[int, ...]
    prb_step: int

    def __post_init__(self):
        bandwidth_space_hz = tuple(float(value) for value in self.bandwidth_space_hz)
        layers_space = tuple(int(value) for value in self.layers_space)
        mcs_space = tuple(int(value) for value in self.mcs_space)
        prb_step = int(self.prb_step)

        if not bandwidth_space_hz:
            raise ValueError("SchedulerSpaceConfig requires at least one bandwidth value.")
        if not layers_space:
            raise ValueError("SchedulerSpaceConfig requires at least one layer value.")
        if not mcs_space:
            raise ValueError("SchedulerSpaceConfig requires at least one MCS value.")
        if prb_step < 1:
            raise ValueError("SchedulerSpaceConfig requires prb_step >= 1.")

        object.__setattr__(self, "bandwidth_space_hz", bandwidth_space_hz)
        object.__setattr__(self, "layers_space", layers_space)
        object.__setattr__(self, "mcs_space", mcs_space)
        object.__setattr__(self, "prb_step", prb_step)


COMMON_LINK_CONSTANTS = LinkConstantsConfig(
    pl_model="umi_sc_nlos",
    fc_hz=3.5e9,
    g_tx_db=8.0,
    g_rx_db=0.0,
    n0_dbm_per_hz=-174.0,
    lna_noise_figure_db=5.0,
    shadow_margin_db=4.0,
    h_bs_m=10.0,
    h_ut_m=1.5,
)

COMMON_PHY_CONSTANTS = PhyConstantsConfig(
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
)

COMMON_SCHEDULER_SPACE = SchedulerSpaceConfig(
    bandwidth_space_hz=(100e6, 50e6),
    layers_space=(1, 2, 3, 4),
    mcs_space=tuple(range(0, 29)),
    prb_step=5,
)

DEFAULT_PA_DATA_CSV = str(Path("PA models") / "3.5Ghz_pas" / "4W_8W_NR_combined_NR_carrier.csv")

__all__ = [
    "COMMON_LINK_CONSTANTS",
    "COMMON_PHY_CONSTANTS",
    "COMMON_SCHEDULER_SPACE",
    "DEFAULT_PA_DATA_CSV",
    "LinkConstantsConfig",
    "PhyConstantsConfig",
    "SchedulerSpaceConfig",
]
