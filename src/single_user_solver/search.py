import pandas as pd

from downlink_candidate_evaluation import CandidatePowerModel, CandidateRateModel
from downlink_candidate_evaluation.mcs_requirements import McsRequirementModel

from .candidate_space import iter_candidates, resolve_candidate_context
from .models import StaticCandidateSpec


# Keep the runtime active table narrow: candidate identity, achieved rate,
# scheduler-side power terms, and one compact SINR sanity check.
ACTIVE_RESULT_COLUMNS = [
    "pa_id",
    "n_prb",
    "n_slots_on",
    "layers",
    "mcs",
    "p_dc_avg_total_w",
    "p_out_total_w",
    "rate_ach_bps",
    "gamma_req_lin",
    "gamma_achieved",
]


_STATIC_CANDIDATE_CATALOG_CACHE = {}


def enumerate_active_candidates_from_context(context):
    """Build full feasible active candidate table."""

    return _evaluate_active_candidates(
        context,
        _get_static_candidates(context, use_cache=context.options.use_cache),
    )


def search_candidates_from_context(context, required_rate_bps):
    """Build active table only for candidates meeting target rate."""

    static_candidates = tuple(
        static_candidate
        for static_candidate in _get_static_candidates(
            context,
            use_cache=context.options.use_cache,
        )
        if static_candidate.rate_ach_bps >= float(required_rate_bps)
    )
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


def _get_static_candidates(context, *, use_cache):
    """Enumerate the deployment-independent candidate metadata for one search shape."""

    if use_cache and context.static_catalog_key in _STATIC_CANDIDATE_CATALOG_CACHE:
        return _STATIC_CANDIDATE_CATALOG_CACHE[context.static_catalog_key]

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
            )
        )

    frozen_candidates = tuple(
        sorted(
            candidates,
            key=lambda c: (-c.rate_ach_bps, c.gamma_req_lin, c.candidate_ordinal),
        )
    )
    if use_cache:
        _STATIC_CANDIDATE_CATALOG_CACHE[context.static_catalog_key] = frozen_candidates
    return frozen_candidates


def _evaluate_active_candidates(context, static_candidates):
    """Evaluate each static candidate and assemble the normalized active table."""

    power_model = CandidatePowerModel(context.mcs_table)

    rows = []
    for static_candidate in static_candidates:
        candidate = static_candidate.candidate
        rrc, pa = resolve_candidate_context(context.search_catalog, candidate)
        result = power_model.solve_candidate_power(
            context.deployment,
            rrc,
            candidate,
            pa,
            gamma_req_lin=static_candidate.gamma_req_lin,
        )
        if not result.is_feasible:
            continue
        rows.append(
            {
                "pa_id": candidate.pa_id,
                "n_prb": candidate.n_prb,
                "n_slots_on": candidate.n_slots_on,
                "layers": candidate.layers,
                "mcs": candidate.mcs,
                "p_dc_avg_total_w": result.p_dc_avg_total_w,
                "p_out_total_w": result.p_out_total_w,
                "rate_ach_bps": static_candidate.rate_ach_bps,
                "gamma_req_lin": static_candidate.gamma_req_lin,
                "gamma_achieved": result.gamma_achieved,
            }
        )

    return pd.DataFrame.from_records(rows, columns=ACTIVE_RESULT_COLUMNS).reset_index(drop=True)
