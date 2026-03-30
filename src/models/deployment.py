"""Shared derived deployment state built from radio config and link distance."""

from dataclasses import dataclass

from .path_loss import PathLossModel


@dataclass(frozen=True)
class DeploymentParams:
    """Physical deployment parameters derived from shared radio config and distance."""

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


def build_deployment(config, distance_m):
    """Build one concrete deployment from shared radio config and link distance."""

    distance_m = float(distance_m)
    path_loss_db = PathLossModel(
        fc_hz=config.fc_hz,
        model=config.pl_model,
        g_tx_db=config.g_tx_db,
        g_rx_db=config.g_rx_db,
        shadow_margin_db=config.shadow_margin_db,
        h_bs_m=config.h_bs_m,
        h_ut_m=config.h_ut_m,
    ).effective_path_loss_db(distance_m)
    return DeploymentParams(
        fc_hz=config.fc_hz,
        channel_bw_hz=config.channel_bw_hz,
        distance_m=distance_m,
        path_loss_db=path_loss_db,
        g_tx_db=config.g_tx_db,
        g_rx_db=config.g_rx_db,
        n0_dbm_per_hz=config.n0_dbm_per_hz,
        lna_noise_figure_db=config.lna_noise_figure_db,
        l_impl_db=config.l_impl_db,
        mi_n_samples=config.mi_n_samples,
        n_dmrs_sym=config.n_dmrs_sym,
        n_guard_sym=config.n_guard_sym,
        n_ul_sym=config.n_ul_sym,
        dft_size_N=config.dft_size_N,
        n_slots_win=config.n_slots_win,
        t_slot_s=config.t_slot_s,
        n_sym_data=config.n_sym_data,
        n_sym_total=config.n_sym_total,
        use_psd_constraint=config.use_psd_constraint,
        psd_max_w_per_hz=config.psd_max_w_per_hz,
        papr_db=config.papr_db,
        g_phi=config.g_phi,
        sigma_phi2=config.sigma_phi2,
        sigma_q2=config.sigma_q2,
        n_tx_chains=config.n_tx_chains,
    )


__all__ = [
    "DeploymentParams",
    "build_deployment",
]
