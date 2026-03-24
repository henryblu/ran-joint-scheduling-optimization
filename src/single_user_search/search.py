import pandas as pd

from downlink_candidate_evaluation import CandidatePowerModel, CandidateRateModel
from downlink_candidate_evaluation.mcs_requirements import McsRequirementModel

from .candidate_space import iter_candidates, resolve_candidate_context
from .models import SingleUserStaticCandidateCatalog, StaticCandidateSpec


ACTIVE_RESULT_COLUMNS = [
    "distance_m",
    "path_loss_db",
    "pa_id",
    "pa_name",
    "bwp_idx",
    "bandwidth_hz",
    "n_prb",
    "n_slots_on",
    "layers",
    "mcs",
    "p_dc_avg_total_w",
    "p_out_total_w",
    "ps_total_w",
    "rate_ach_bps",
    "gamma_req_lin",
    "gamma_req_db",
    "gamma_achieved",
    "rho_ach_raw_linear",
    "n_streams",
    "g_bf_linear",
    "sigma_e2",
]


_ACTIVE_TABLE_CACHE = {}
_STATIC_CANDIDATE_CATALOG_CACHE = {}
_CANDIDATE_BATCH_SIZE = 2048


def enumerate_active_candidates_from_context(context):
    """Build full feasible active candidate table."""

    cache_key = context.active_table_key if context.options.use_cache else None
    if cache_key in _ACTIVE_TABLE_CACHE:
        return _ACTIVE_TABLE_CACHE[cache_key].copy()

    static_candidates = _get_static_candidates(context)
    table = _evaluate_active_candidates(context, static_candidates)

    if cache_key is not None:
        _ACTIVE_TABLE_CACHE[cache_key] = table.copy()

    return table


def search_candidates_from_context(context, required_rate_bps):
    """Build active table only for candidates meeting target rate."""

    static_candidates = [
        sc for sc in _get_static_candidates(context)
        if sc.rate_ach_bps >= float(required_rate_bps)
    ]
    return _evaluate_active_candidates(context, static_candidates)


def filter_rate_feasible_candidates(active_candidate_table, required_rate_bps):
    """Filter existing table by rate."""

    return (
        active_candidate_table[
            active_candidate_table["rate_ach_bps"] >= float(required_rate_bps)
        ]
        .copy()
        .reset_index(drop=True)
        .reindex(columns=ACTIVE_RESULT_COLUMNS)
    )


def _get_static_candidates(context):
    """Enumerate and cache static candidate metadata."""

    cached = _STATIC_CANDIDATE_CATALOG_CACHE.get(context.static_catalog_key)
    if cached is not None:
        return cached.candidates

    rate_model = CandidateRateModel(context.mcs_table)
    mcs_model = McsRequirementModel(context.mcs_table)
    sinr_table = mcs_model.get_required_sinr_table(context.deployment)

    candidates = []
    for ordinal, candidate in enumerate(iter_candidates(context.search_catalog)):
        rrc, _pa = resolve_candidate_context(context.search_catalog, candidate)
        rate = rate_model.compute_candidate_rate(context.deployment, rrc, candidate)
        gamma = sinr_table[candidate.mcs]
        candidates.append(
            StaticCandidateSpec(
                candidate_ordinal=ordinal,
                candidate=candidate,
                rate_ach_bps=rate.rate_ach_bps,
                gamma_req_lin=gamma["rho_req_linear"],
                gamma_req_db=gamma["rho_req_db"],
            )
        )

    frozen_candidates = tuple(
        sorted(
            candidates,
            key=lambda c: (-c.rate_ach_bps, c.gamma_req_lin, c.candidate_ordinal),
        )
    )
    _STATIC_CANDIDATE_CATALOG_CACHE[context.static_catalog_key] = SingleUserStaticCandidateCatalog(
        candidates=frozen_candidates
    )
    return frozen_candidates


def _evaluate_active_candidates(context, static_candidates):
    """Evaluate candidate feasibility and assemble the normalized active table."""

    power_model = CandidatePowerModel(context.mcs_table)
    feasible = {}

    batch = []
    for static_candidate in static_candidates:
        batch.append(static_candidate)
        if len(batch) < _CANDIDATE_BATCH_SIZE:
            continue
        _evaluate_batch(batch, power_model, context, feasible)
        batch = []

    if batch:
        _evaluate_batch(batch, power_model, context, feasible)

    rows = []
    for static_candidate in static_candidates:
        result = feasible.get(static_candidate.candidate_ordinal)
        if result is None:
            continue
        candidate = static_candidate.candidate
        rrc, pa = resolve_candidate_context(context.search_catalog, candidate)
        rows.append(
            {
                "distance_m": context.deployment.distance_m,
                "path_loss_db": context.deployment.path_loss_db,
                "pa_id": candidate.pa_id,
                "pa_name": pa.pa_name,
                "bwp_idx": candidate.bwp_idx,
                "bandwidth_hz": rrc.bwp_bw_hz,
                "n_prb": candidate.n_prb,
                "n_slots_on": candidate.n_slots_on,
                "layers": candidate.layers,
                "mcs": candidate.mcs,
                "p_dc_avg_total_w": result.p_dc_avg_total_w,
                "p_out_total_w": result.p_out_total_w,
                "ps_total_w": result.ps_total_w,
                "rate_ach_bps": static_candidate.rate_ach_bps,
                "gamma_req_lin": static_candidate.gamma_req_lin,
                "gamma_req_db": static_candidate.gamma_req_db,
                "gamma_achieved": result.gamma_achieved,
                "rho_ach_raw_linear": result.rho_ach_raw_linear,
                "n_streams": result.n_streams,
                "g_bf_linear": result.g_bf_linear,
                "sigma_e2": result.sigma_e2,
            }
        )

    return pd.DataFrame.from_records(rows, columns=ACTIVE_RESULT_COLUMNS).reset_index(drop=True)


def _evaluate_batch(batch, power_model, context, feasible):
    """Inline batch evaluation using the normalized candidate shape."""

    for static_candidate in batch:
        candidate = static_candidate.candidate
        rrc, pa = resolve_candidate_context(context.search_catalog, candidate)
        result = power_model.solve_candidate_power(
            context.deployment,
            rrc,
            candidate,
            pa,
            gamma_req_lin=static_candidate.gamma_req_lin,
        )
        if result.is_feasible:
            feasible[static_candidate.candidate_ordinal] = result
