from dataclasses import dataclass, field

import pandas as pd

from pa_models import PAParams
from radio_configs import RadioConfig
from single_user_search.models import SearchSpace

from .presets import MultiUserPreset


@dataclass(frozen=True)
class MultiUserSystemConfig:
    """TDMA-owned mixed-slot system view derived from radio config and TDD pattern."""

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
class MultiUserTdmaScenario:
    """Prepared multi-user TDMA study scenario used by notebook-facing helpers."""

    user_table: pd.DataFrame
    preset: MultiUserPreset
    system_cfg: MultiUserSystemConfig
    pa_catalog: tuple[PAParams, ...]
    active_search_model_inputs: RadioConfig
    active_search_shape: SearchSpace


@dataclass
class MultiUserTdmaStudyResult:
    """Notebook-facing result tables for one multi-user pre-scheduler study run."""

    repeated_frames: int = 0
    repeated_period_slots: int = 0
    active_candidate_tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    active_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame_share_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_candidate_spaces: dict[int, pd.DataFrame] = field(default_factory=dict)
    user_candidate_review_tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    user_candidate_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
