"""Static radio configuration types and immutable shared config values."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


FrozenMcsTable = Mapping[int, Mapping[str, float]]


@dataclass(frozen=True)
class RadioConfig:
    pl_model: str
    fc_hz: float
    g_tx_db: float
    g_rx_db: float
    n0_dbm_per_hz: float
    lna_noise_figure_db: float
    shadow_margin_db: float
    h_bs_m: float
    h_ut_m: float
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
    bandwidth_space_hz: tuple[float, ...]
    layers_space: tuple[int, ...]
    mcs_space: tuple[int, ...]
    prb_step: int
    mcs_table: FrozenMcsTable
    pa_data_csv: str


def freeze_mcs_table(mcs_table: Mapping[int, Mapping[str, float]]) -> FrozenMcsTable:
    """Freeze one nested MCS table into read-only mappings."""

    frozen_rows = {}
    for mcs, row in mcs_table.items():
        frozen_rows[int(mcs)] = MappingProxyType(dict(row))
    return MappingProxyType(frozen_rows)


__all__ = [
    "FrozenMcsTable",
    "RadioConfig",
    "freeze_mcs_table",
]
