import hashlib
import json
from dataclasses import asdict, is_dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd

from downlink_candidate_evaluation import CandidatePowerModel, CandidatePowerResult, CandidateRateModel
from downlink_candidate_evaluation.mcs_requirements import McsRequirementModel

from .candidate_space import iter_candidates, resolve_candidate_context
from .models import SingleUserSearchOptions, SingleUserStaticCandidateCatalog, StaticCandidateSpec
from .problem_factory import clear_problem_factory_cache


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
    "n_active_tx",
    "mcs",
    "alpha_f",
    "p_dc_avg_total_w",
    "p_rf_out_active_w",
    "p_out_total_w",
    "p_sig_out_active_w",
    "p_sig_out_total_w",
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


def enumerate_active_candidates_from_context(context, *, options=None):
    """Build the full feasible active candidate table for one prepared deployment."""

    resolved_options = _resolve_run_options(context.options, options)
    cache_key = _build_active_table_cache_key_if_enabled(context, resolved_options)
    if cache_key is not None:
        cached_active_table = _get_cached_active_table(cache_key)
        if cached_active_table is not None:
            return cached_active_table

    static_catalog = _build_static_candidate_catalog(context)
    active_table = _build_active_candidate_table(context, static_catalog.candidates)
    if cache_key is not None:
        _store_cached_active_table(cache_key, active_table)
    return active_table


def search_candidates_from_context(context, required_rate_bps, *, options=None):
    """Filter a prepared deployment's active table down to one user's target-rate space."""

    resolved_options = _resolve_run_options(context.options, options)
    static_catalog = _build_static_candidate_catalog(context)
    filtered_candidates = _filter_static_candidates_by_rate(
        static_catalog.candidates,
        required_rate_bps=float(required_rate_bps),
    )
    return _build_active_candidate_table(context, filtered_candidates)


def clear_cache():
    """Clear all memoized single-user caches for the current Python process."""

    _ACTIVE_TABLE_CACHE.clear()
    _STATIC_CANDIDATE_CATALOG_CACHE.clear()
    clear_problem_factory_cache()


def _build_static_candidate_catalog(context):
    """Build and cache the scenario-invariant candidate catalog for one search-space shape."""

    cache_key = _build_static_catalog_cache_key(context)
    cached_catalog = _STATIC_CANDIDATE_CATALOG_CACHE.get(cache_key)
    if cached_catalog is not None:
        return cached_catalog

    static_catalog = SingleUserStaticCandidateCatalog(
        candidates=_build_static_candidate_specs(context),
    )
    _STATIC_CANDIDATE_CATALOG_CACHE[cache_key] = static_catalog
    return static_catalog


def _build_static_candidate_specs(context):
    """Enumerate and sort the static candidate metadata reused across deployments."""

    rate_model = CandidateRateModel(context.mcs_table)
    mcs_model = McsRequirementModel(context.mcs_table)
    sinr_requirement_table = mcs_model.get_required_sinr_table(context.deployment)

    candidates = []
    for candidate_ordinal, candidate in enumerate(iter_candidates(context.built_problem)):
        rrc, sched, pa = resolve_candidate_context(context.built_problem, candidate)
        if rrc is None or pa is None:
            continue

        rate_result = rate_model.compute_candidate_rate(context.deployment, rrc, sched)
        gamma_req = sinr_requirement_table[sched.mcs]
        candidates.append(
            StaticCandidateSpec(
                candidate_ordinal=int(candidate_ordinal),
                candidate=replace(
                    candidate,
                    pa_id=int(candidate.pa_id),
                    bwp_idx=int(candidate.bwp_idx),
                    n_prb=int(candidate.n_prb),
                    n_slots_on=int(candidate.n_slots_on),
                    layers=int(candidate.layers),
                    n_active_tx=int(candidate.n_active_tx),
                    mcs=int(candidate.mcs),
                ),
                pa_name=str(pa.pa_name),
                bandwidth_hz=float(rrc.bwp_bw_hz),
                alpha_f=float(sched.n_prb / max(rrc.prb_max_bwp, 1)),
                rate_ach_bps=float(rate_result.rate_ach_bps),
                gamma_req_lin=float(gamma_req["rho_req_linear"]),
                gamma_req_db=float(gamma_req["rho_req_db"]),
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.rate_ach_bps,
                candidate.gamma_req_lin,
                candidate.candidate_ordinal,
            ),
        )
    )


def _filter_static_candidates_by_rate(static_candidates, required_rate_bps):
    """Keep only static candidates that satisfy the requested target rate."""

    return tuple(
        candidate
        for candidate in static_candidates
        if float(candidate.rate_ach_bps) >= float(required_rate_bps)
    )


def _build_active_candidate_table(context, static_candidates):
    """Evaluate a static candidate slice and assemble the feasible active table."""

    power_model = CandidatePowerModel(context.mcs_table)
    dynamic_results = []
    for candidate_batch in _iter_candidate_batches(static_candidates, batch_size=_CANDIDATE_BATCH_SIZE):
        dynamic_results.extend(
            _evaluate_candidate_batch(power_model, context.built_problem, candidate_batch)
        )
    return _assemble_active_candidate_table(
        static_candidates,
        dynamic_results,
        deployment=context.deployment,
    )


def _assemble_active_candidate_table(static_candidates, dynamic_results, *, deployment):
    """Assemble the feasible active candidate table from evaluated dynamic results."""

    feasible_result_by_ordinal = {
        int(result["candidate_ordinal"]): result["power_result"]
        for result in dynamic_results
        if bool(result["power_result"].is_feasible)
    }

    rows = []
    for static_candidate in static_candidates:
        dynamic_result = feasible_result_by_ordinal.get(int(static_candidate.candidate_ordinal))
        if dynamic_result is None:
            continue
        rows.append(_build_active_candidate_row(static_candidate, dynamic_result, deployment=deployment))
    return _finalize_active_candidate_table(rows)


def filter_rate_feasible_candidates(active_candidate_table, required_rate_bps):
    """Filter an active candidate table down to rows that meet the target rate."""

    if active_candidate_table.empty:
        return active_candidate_table.copy()

    filtered_candidate_table = active_candidate_table[
        active_candidate_table["rate_ach_bps"] >= float(required_rate_bps)
    ].copy()
    return filtered_candidate_table.reset_index(drop=True).reindex(columns=ACTIVE_RESULT_COLUMNS)


def _iter_candidate_batches(candidates, *, batch_size):
    """Yield fixed-size evaluation batches while preserving candidate order."""

    batch = []
    for candidate in candidates:
        batch.append(candidate)
        if len(batch) < batch_size:
            continue
        yield batch
        batch = []
    if batch:
        yield batch


def _evaluate_candidate_batch(power_model, problem, candidates):
    """Evaluate one execution batch using a shared power model instance."""

    rows = []
    for static_candidate in candidates:
        rrc, sched, pa = resolve_candidate_context(problem, static_candidate.candidate)
        if rrc is None or pa is None:
            power_result = CandidatePowerResult(
                is_feasible=False,
                infeasibility_reason="rrc_not_found",
                gamma_req_lin=float(static_candidate.gamma_req_lin),
                gamma_req_db=float(static_candidate.gamma_req_db),
            )
        else:
            power_result = power_model.solve_candidate_power(
                problem.deployment,
                rrc,
                sched,
                pa,
                gamma_req_lin=float(static_candidate.gamma_req_lin),
            )
        rows.append(
            {
                "candidate_ordinal": int(static_candidate.candidate_ordinal),
                "power_result": power_result,
            }
        )
    return rows


def _build_active_candidate_row(static_candidate, dynamic_result, *, deployment):
    """Merge one static candidate spec with one feasible dynamic result."""

    return {
        "distance_m": float(deployment.distance_m),
        "path_loss_db": float(deployment.path_loss_db),
        "pa_id": int(static_candidate.candidate.pa_id),
        "pa_name": str(static_candidate.pa_name),
        "bwp_idx": int(static_candidate.candidate.bwp_idx),
        "bandwidth_hz": float(static_candidate.bandwidth_hz),
        "n_prb": int(static_candidate.candidate.n_prb),
        "n_slots_on": int(static_candidate.candidate.n_slots_on),
        "layers": int(static_candidate.candidate.layers),
        "n_active_tx": int(static_candidate.candidate.n_active_tx),
        "mcs": int(static_candidate.candidate.mcs),
        "alpha_f": float(static_candidate.alpha_f),
        "p_dc_avg_total_w": float(dynamic_result.p_dc_avg_total_w),
        "p_rf_out_active_w": float(dynamic_result.p_out_total_w),
        "p_out_total_w": float(dynamic_result.p_out_total_w),
        "p_sig_out_active_w": float(dynamic_result.p_sig_out_total_w),
        "p_sig_out_total_w": float(dynamic_result.p_sig_out_total_w),
        "ps_total_w": float(dynamic_result.ps_total_w),
        "rate_ach_bps": float(static_candidate.rate_ach_bps),
        "gamma_req_lin": float(static_candidate.gamma_req_lin),
        "gamma_req_db": float(static_candidate.gamma_req_db),
        "gamma_achieved": float(dynamic_result.gamma_achieved),
        "rho_ach_raw_linear": float(dynamic_result.rho_ach_raw_linear),
        "n_streams": int(dynamic_result.n_streams),
        "g_bf_linear": float(dynamic_result.g_bf_linear),
        "sigma_e2": float(dynamic_result.sigma_e2),
    }


def _finalize_active_candidate_table(rows):
    """Normalize the canonical active-candidate table schema and row order."""

    candidate_table = pd.DataFrame(rows)
    if candidate_table.empty:
        return pd.DataFrame(columns=ACTIVE_RESULT_COLUMNS)
    return candidate_table.reindex(columns=ACTIVE_RESULT_COLUMNS).reset_index(drop=True)


def _build_active_table_cache_key_if_enabled(context, options):
    """Build the active-table memoization key only when caching is enabled."""

    if not options.use_cache:
        return None

    cache_context = replace(
        context,
        options=replace(
            context.options,
            use_cache=False,
        ),
    )
    return _build_active_table_cache_key(cache_context)


def _build_static_catalog_cache_key(context):
    """Build the cache key for one search-space-shaped static catalog."""

    payload = {
        "model_inputs": _normalize_cache_value(context.model_inputs),
        "options": _normalize_cache_value(_build_static_cache_options(context.options)),
        "pa_catalog": _normalize_cache_value(context.pa_catalog),
    }
    return _build_hash_key(payload)


def _build_active_table_cache_key(context):
    """Build the cache key for one deployment-specific active candidate table."""

    payload = {
        "deployment": _normalize_cache_value(context.deployment),
        "model_inputs": _normalize_cache_value(context.model_inputs),
        "options": _normalize_cache_value(_build_static_cache_options(context.options)),
        "pa_catalog": _normalize_cache_value(context.pa_catalog),
    }
    return _build_hash_key(payload)


def _get_cached_active_table(cache_key):
    """Return a copy of the cached active table, if present."""

    cached_active_table = _ACTIVE_TABLE_CACHE.get(cache_key)
    return None if cached_active_table is None else cached_active_table.copy()


def _store_cached_active_table(cache_key, active_table):
    """Store a copy of the computed active candidate table."""

    _ACTIVE_TABLE_CACHE[cache_key] = active_table.copy()


def _build_static_cache_options(options):
    """Keep only the search-space-shaping options in cache keys."""

    return SingleUserSearchOptions(
        fast_mode=options.fast_mode,
        prb_step=options.prb_step,
        bandwidth_space_hz=options.bandwidth_space_hz,
        n_slots_on_space=options.n_slots_on_space,
        use_cache=False,
    )


def _build_hash_key(payload):
    """Build a stable SHA256 cache key from the normalized payload."""

    raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def _normalize_cache_value(value):
    """Convert nested search inputs into stable JSON-serializable primitives."""

    if is_dataclass(value):
        return _normalize_cache_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_cache_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def _resolve_run_options(problem_options, options):
    """Merge execution overrides onto the prepared context's stored search options."""

    if options is None:
        return problem_options
    return replace(
        problem_options,
        fast_mode=options.fast_mode,
        prb_step=options.prb_step,
        bandwidth_space_hz=options.bandwidth_space_hz,
        n_slots_on_space=options.n_slots_on_space,
        use_cache=options.use_cache,
    )
