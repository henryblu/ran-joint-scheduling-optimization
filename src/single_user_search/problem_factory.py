from pa_models import build_pa_catalog
from radio_models import DeploymentParams, PathLossModel, build_resolved_fingerprint

from .candidate_space import build_search_catalog
from .models import PreparedSingleUserContext


def prepare_single_user_problem(request, model_inputs, search_shape, *, pa_catalog=None):
    """Build the reusable single-user context from resolved engine state."""

    config = getattr(model_inputs, "config", model_inputs)
    resolved_pa_catalog = (
        tuple(build_pa_catalog(config.pa_data_csv))
        if pa_catalog is None
        else tuple(pa_catalog)
    )
    search_catalog = build_search_catalog(
        model_inputs=config,
        pa_catalog=resolved_pa_catalog,
        search_shape=search_shape,
    )
    deployment = _build_deployment(config, request.distance_m)
    static_catalog_key = build_resolved_fingerprint(
        {
            "model_inputs": build_resolved_fingerprint(config),
            "search_shape": search_shape.fingerprint,
            "pa_catalog": build_resolved_fingerprint(resolved_pa_catalog),
        }
    )
    active_table_key = build_resolved_fingerprint(
        {
            "static_catalog": static_catalog_key,
            "deployment": deployment,
        }
    )
    return PreparedSingleUserContext(
        request=request,
        model_inputs=config,
        deployment=deployment,
        search_catalog=search_catalog,
        static_catalog_key=static_catalog_key,
        active_table_key=active_table_key,
    )


def clear_problem_factory_cache():
    """Compatibility no-op after deleting the legacy radio-core config caches."""

    return None


def _build_deployment(config, distance_m):
    """Build the concrete single-user deployment owned by the single-user layer."""

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
