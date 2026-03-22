import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd

from downlink_candidate_evaluation import DownlinkCandidateEvaluator, DownlinkProblemSpace
from downlink_candidate_evaluation.mcs_requirements import McsRequirementModel

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
    "alpha_t",
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
_WORKER_EVALUATOR = None
_WORKER_PROBLEM = None


def enumerate_active_candidates_from_context(context, *, options=None):
    """Build the full feasible active candidate table for one prepared deployment.

    Steps:
    1. Resolve execution overrides on top of the prepared context options.
    2. Reuse a cached active table when the deployment and search space already match.
    3. Build or fetch the static single-user candidate catalog for that search space.
    4. Evaluate every catalog candidate against the concrete deployment.
    5. Assemble only feasible rows into the canonical flat active-candidate table.
    """

    resolved_options = _resolve_run_options(context.options, options)
    cache_key = _build_active_table_cache_key_if_enabled(context, resolved_options)
    if cache_key is not None:
        cached_active_table = _get_cached_active_table(cache_key)
        if cached_active_table is not None:
            return cached_active_table

    static_catalog = _build_static_candidate_catalog(context)
    dynamic_results = _evaluate_dynamic_candidate_slice(
        context,
        static_catalog.candidates,
        parallel=resolved_options.parallel,
        max_workers=resolved_options.max_workers,
    )
    active_table = _assemble_active_candidate_table(static_catalog.candidates, dynamic_results)
    if cache_key is not None:
        _store_cached_active_table(cache_key, active_table)
    return active_table


def search_candidates_from_context(context, required_rate_bps, *, options=None):
    """Filter a prepared deployment's active table down to one user's target-rate space."""

    active_table = enumerate_active_candidates_from_context(context, options=options)
    return filter_rate_feasible_candidates(
        active_table,
        required_rate_bps=float(required_rate_bps),
    )


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

    candidate_space = DownlinkProblemSpace(context.mcs_table)
    grid_model = candidate_space.grid_model
    mcs_model = McsRequirementModel(context.mcs_table)
    sinr_requirement_table = mcs_model.get_required_sinr_table(context.deployment)

    candidates = []
    for candidate_ordinal, candidate in enumerate(candidate_space.iter_candidates(context.built_problem)):
        rrc = context.rrc_lookup.get((int(candidate.pa_id), int(candidate.bwp_idx)))
        if rrc is None:
            continue

        sched = grid_model.build_scheduler_vars(candidate)
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
                pa_name=str(context.pa_catalog[candidate.pa_id].pa_name),
                bandwidth_hz=float(rrc.bwp_bw_hz),
                alpha_f=float(sched.n_prb / max(rrc.prb_max_bwp, 1)),
                alpha_t=float(grid_model.scheduler_duty_cycle(context.deployment, sched)),
                rate_ach_bps=float(grid_model.compute_rate(context.deployment, rrc, sched)),
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


def _evaluate_dynamic_candidate_slice(
    context,
    static_candidates,
    *,
    parallel=False,
    max_workers=None,
):
    """Evaluate the dynamic power feasibility for one static candidate slice."""

    candidate_batch_builder = _build_candidate_batch_builder(static_candidates)
    return _evaluate_candidate_batches(
        context.mcs_table,
        context.built_problem,
        candidate_batch_builder,
        parallel=parallel,
        max_workers=max_workers,
    )


def _assemble_active_candidate_table(static_candidates, dynamic_results):
    """Assemble the feasible active candidate table from evaluated dynamic results."""

    feasible_result_by_ordinal = {
        int(result.candidate_ordinal): result
        for result in dynamic_results
        if bool(result.is_feasible)
    }

    rows = []
    for static_candidate in static_candidates:
        dynamic_result = feasible_result_by_ordinal.get(int(static_candidate.candidate_ordinal))
        if dynamic_result is None:
            continue
        rows.append(_build_active_candidate_row(static_candidate, dynamic_result))
    return _finalize_active_candidate_table(rows)


def filter_rate_feasible_candidates(active_candidate_table, required_rate_bps):
    """Filter an active candidate table down to rows that meet the target rate."""

    if active_candidate_table.empty:
        return active_candidate_table.copy()

    filtered_candidate_table = active_candidate_table[
        active_candidate_table["rate_ach_bps"] >= float(required_rate_bps)
    ].copy()
    return filtered_candidate_table.reset_index(drop=True).reindex(columns=ACTIVE_RESULT_COLUMNS)


def _build_candidate_batch_builder(candidates):
    """Return a generator factory for fixed-size batches over a candidate slice."""

    return lambda: _iter_candidate_batches(candidates, batch_size=_CANDIDATE_BATCH_SIZE)


def _iter_candidate_batches(candidates, *, batch_size):
    """Yield fixed-size evaluation batches while preserving candidate order."""

    batch_id = 0
    batch = []
    for candidate in candidates:
        batch.append(candidate)
        if len(batch) < batch_size:
            continue
        yield batch_id, batch
        batch_id += 1
        batch = []
    if batch:
        yield batch_id, batch


def _evaluate_candidate_batches(mcs_table, problem, candidate_batch_builder, *, parallel=False, max_workers=None):
    """Evaluate candidate batches either in worker processes or in-process."""

    if parallel:
        parallel_rows = _run_batched_parallel(
            mcs_table,
            problem,
            candidate_batch_builder(),
            max_workers=max_workers,
        )
        if parallel_rows is not None:
            return parallel_rows
    return _run_batched_serial(mcs_table, problem, candidate_batch_builder())


def _run_batched_parallel(mcs_table, problem, candidate_batches, max_workers=None):
    """Try process-based evaluation and fall back to serial mode when unavailable."""

    futures = {}
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_worker_state,
            initargs=(mcs_table, problem),
        ) as executor:
            for batch_id, candidates in candidate_batches:
                futures[executor.submit(_evaluate_candidate_batch_worker, (batch_id, candidates))] = batch_id
            if not futures:
                return []

            batch_rows = {}
            for future in as_completed(futures):
                batch_id, rows = future.result()
                batch_rows[batch_id] = rows
    except (OSError, PermissionError):
        return None
    return _flatten_batch_rows(batch_rows)


def _run_batched_serial(mcs_table, problem, candidate_batches):
    """Evaluate candidate batches in-process while preserving enumeration order."""

    evaluator = DownlinkCandidateEvaluator(mcs_table)
    rows = []
    for _batch_id, candidates in candidate_batches:
        rows.extend(_evaluate_candidate_batch(evaluator, problem, candidates))
    return rows


def _evaluate_candidate_batch(evaluator, problem, candidates):
    """Evaluate one execution batch using a shared evaluator instance."""

    return [evaluator.evaluate_static_candidate(problem, candidate) for candidate in candidates]


def _initialize_worker_state(mcs_table, problem):
    """Initialize read-only worker state once per process."""

    global _WORKER_EVALUATOR, _WORKER_PROBLEM
    _WORKER_EVALUATOR = DownlinkCandidateEvaluator(mcs_table)
    _WORKER_PROBLEM = problem


def _evaluate_candidate_batch_worker(payload):
    """Worker entry point for one execution batch."""

    batch_id, candidates = payload
    rows = _evaluate_candidate_batch(_WORKER_EVALUATOR, _WORKER_PROBLEM, candidates)
    return batch_id, rows


def _flatten_batch_rows(batch_rows):
    """Flatten completed batch rows back into canonical batch order."""

    rows = []
    for batch_id in sorted(batch_rows):
        rows.extend(batch_rows[batch_id])
    return rows


def _build_active_candidate_row(static_candidate, dynamic_result):
    """Merge one static candidate spec with one feasible dynamic result."""

    return {
        "distance_m": float(dynamic_result.distance_m),
        "path_loss_db": float(dynamic_result.path_loss_db),
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
        "alpha_t": float(static_candidate.alpha_t),
        "p_dc_avg_total_w": float(dynamic_result.p_dc_avg_total_w),
        "p_rf_out_active_w": float(dynamic_result.p_rf_out_active_w),
        "p_out_total_w": float(dynamic_result.p_out_total_w),
        "p_sig_out_active_w": float(dynamic_result.p_sig_out_active_w),
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
            parallel=False,
            max_workers=None,
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
        parallel=False,
        max_workers=None,
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
        parallel=options.parallel,
        max_workers=options.max_workers,
        use_cache=options.use_cache,
    )
