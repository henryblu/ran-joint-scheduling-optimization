from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config_values import LinkConstantsConfig, PhyConstantsConfig, SchedulerSpaceConfig


@dataclass(frozen=True)
class ResolvedModelInputs:
    """Frozen preset-derived radio inputs trusted by engine code."""

    fingerprint: str
    link: LinkConstantsConfig
    phy: PhyConstantsConfig
    scheduler: SchedulerSpaceConfig
    mcs_table: dict[int, dict[str, float]]
    pa_data_csv: str


@dataclass(frozen=True)
class ResolvedSearchShape:
    """Concrete tuple-based scheduler dimensions used by grid search."""

    bandwidth_space_hz: tuple[float, ...]
    n_slots_on_space: tuple[int, ...]
    layers_space: tuple[int, ...]
    mcs_space: tuple[int, ...]
    prb_step: int
    fingerprint: str
    use_cache: bool = True


SearchSpace = ResolvedSearchShape


@dataclass(frozen=True)
class ResolvedMultiUserSystemCfg:
    """Resolved mixed-slot TDMA system view built from canonical radio config."""

    fc_hz: float
    channel_bw_hz: float
    bandwidth_space_hz: tuple[float, ...]
    total_prbs: int
    frame_slots: int
    slot_dl_symbols: int
    slot_guard_symbols: int
    slot_ul_symbols: int
    slot_payload_symbols: int
    total_slots: int
    delta_f_hz: float
    g_tx_db: float
    g_rx_db: float
    noise_density_dbm_per_hz: float
    noise_figure_db: float
    impl_loss_db: float
    mi_n_samples: int
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
    papr_db: float
    g_phi: float
    sigma_phi2: float
    sigma_q2: float
    layers_space: tuple[int, ...]
    mcs_space: tuple[int, ...]
    prb_step: int


@dataclass(frozen=True)
class DeploymentParams:
    """Deployment / physical environment parameters for one scenario."""

    fc_hz: float
    channel_bw_hz: float
    distance_m: float
    path_loss_db: float
    g_tx_db: float
    g_rx_db: float
    n0_dbm_per_hz: float
    lna_noise_figure_db: float
    l_impl_db: float
    mi_n_samples: int
    n_dmrs_sym: int
    n_guard_sym: int
    n_ul_sym: int
    dft_size_N: int
    n_slots_win: int
    t_slot_s: float
    n_sym_data: int
    n_sym_total: int
    use_psd_constraint: bool
    psd_max_w_per_hz: float
    papr_db: float
    g_phi: float
    sigma_phi2: float
    sigma_q2: float
    n_tx_chains: int


@dataclass(frozen=True)
class PAParams:
    """Measured PA representation."""

    p_max_w: float
    p_idle_w: float
    eta_max: float
    g_pa_eff_linear: float
    kappa_distortion: float
    backoff_db: float
    pa_name: str = ""
    curve_pout_w: np.ndarray | None = None
    curve_pdc_w: np.ndarray | None = None
    curve_pin_w: np.ndarray | None = None
    source_csv: str = ""


@dataclass(frozen=True)
class RRCParams:
    """RRC/BWP envelope for one bandwidth and PA pairing."""

    bwp_bw_hz: float
    bwp_index: int
    delta_f_hz: float
    prb_max_bwp: int
    max_layers: int
    max_mcs: int
    active_pa_id: int


@dataclass(frozen=True)
class ModelOptions:
    """Execution and optimization policy."""

    tie_break_keys: tuple[str, ...] = ("p_dc_avg_total_w", "bandwidth_hz", "n_prb", "n_slots_on")


@dataclass(frozen=True)
class Problem:
    """Complete single-user problem definition."""

    deployment: DeploymentParams
    pa_catalog: tuple[PAParams, ...]
    rrc_catalog: tuple[RRCParams, ...]
    search_space: ResolvedSearchShape
    options: ModelOptions
    rrc_lookup: Mapping[tuple[int, int], RRCParams] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class SchedulerVars:
    """Scheduler variables for evaluating one configuration."""

    n_prb: int
    n_slots_on: int
    layers: int
    mcs: int


@dataclass(frozen=True)
class Candidate:
    """One discrete scheduler/RRC/PA candidate."""

    pa_id: int
    bwp_idx: int
    n_prb: int
    n_slots_on: int
    layers: int
    mcs: int
