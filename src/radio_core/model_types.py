from dataclasses import dataclass

import numpy as np


@dataclass
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


@dataclass
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


@dataclass
class RRCParams:
    """RRC/BWP envelope for one bandwidth and PA pairing."""

    bwp_bw_hz: float
    bwp_index: int
    delta_f_hz: float
    prb_max_bwp: int
    max_layers: int
    max_mcs: int
    active_pa_id: int


@dataclass
class SearchSpace:
    """Discrete scheduler search dimensions."""

    n_slots_on_space: list
    layers_space: list
    n_active_tx_space: list
    mcs_space: list
    prb_step: int = 1


@dataclass
class ModelOptions:
    """Execution and optimization policy."""

    tie_break_keys: tuple = ("p_dc_avg_total_w", "bandwidth_hz", "n_prb", "n_slots_on")
    fast_mode: bool = False


@dataclass
class Problem:
    """Complete single-user problem definition."""

    deployment: DeploymentParams
    pa_catalog: list
    rrc_catalog: list
    search_space: SearchSpace
    options: ModelOptions


@dataclass
class SchedulerVars:
    """Scheduler variables for evaluating one configuration."""

    n_prb: int
    n_slots_on: int
    layers: int
    n_active_tx: int
    mcs: int


@dataclass
class Candidate:
    """One discrete scheduler/RRC/PA candidate."""

    pa_id: int
    bwp_idx: int
    n_prb: int
    n_slots_on: int
    layers: int
    n_active_tx: int
    mcs: int
