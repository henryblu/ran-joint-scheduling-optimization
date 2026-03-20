from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from downlink_candidate_evaluation import DownlinkCandidateEvaluator, DownlinkProblemSpace

from .cache import build_cache_key, get_cached_result, store_cached_result
from .models import SingleUserSearchResult


RESULT_COLUMNS = [
    "distance_m",
    "path_loss_db",
    "is_feasible",
    "infeasibility_reason",
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
    "meets_rate_target",
    "gamma_req_lin",
    "gamma_achieved",
    "rho_ach_raw_linear",
    "n_streams",
    "g_bf_linear",
    "sigma_e2",
]


def run_single_user_search(problem, *, options=None):
    """Enumerate and evaluate the full candidate ledger for one built problem.

    Steps:
    1. Resolve the execution options and load a cached result when enabled.
    2. Enumerate the built candidate space in its natural traversal order.
    3. Evaluate each candidate and preserve that same order in the result ledger.
    4. Attach rate-target metadata and return the canonical candidate table.
    """
    resolved_options = _resolve_run_options(problem.options, options)
    cache_key = _build_cache_key_if_enabled(problem, resolved_options)
    if cache_key is not None:
        cached = get_cached_result(cache_key)
        if cached is not None:
            return cached

    candidate_table = _build_candidate_table(
        problem,
        include_infeasible=resolved_options.include_infeasible,
        parallel=resolved_options.parallel,
        max_workers=resolved_options.max_workers,
    )
    candidate_table = _annotate_rate_target(
        candidate_table,
        required_rate_bps=problem.request.required_rate_bps,
    )
    result = SingleUserSearchResult(
        request=problem.request,
        candidate_table=candidate_table,
    )
    if cache_key is not None:
        store_cached_result(cache_key, result)
    return result


def _build_candidate_table(problem, include_infeasible, parallel=False, max_workers=None):
    """Evaluate the full candidate space and normalize the result ledger."""
    candidate_space = DownlinkProblemSpace(problem.model_inputs["mcs_table"])
    grouped_candidates = _collect_candidate_groups(problem, candidate_space)
    rows = _evaluate_candidate_groups(
        problem.model_inputs["mcs_table"],
        problem.problem,
        grouped_candidates,
        include_infeasible,
        parallel=parallel,
        max_workers=max_workers,
    )
    return _finalize_candidate_table(rows)


def _collect_candidate_groups(problem, candidate_space):
    """Group enumerated candidates by RRC key while preserving traversal order."""
    grouped_candidates = {}
    for candidate in candidate_space.iter_candidates(problem.problem):
        grouped_candidates.setdefault((candidate.pa_id, candidate.bwp_idx), []).append(candidate)
    return list(grouped_candidates.values())


def _evaluate_candidate_groups(mcs_table, problem, grouped_candidates, include_infeasible, parallel=False, max_workers=None):
    """Evaluate candidate groups either in worker processes or serial fallback mode."""
    if not grouped_candidates:
        return []
    if parallel and len(grouped_candidates) > 1:
        parallel_rows = _evaluate_candidate_groups_parallel(
            mcs_table,
            problem,
            grouped_candidates,
            include_infeasible,
            max_workers=max_workers,
        )
        if parallel_rows is not None:
            return parallel_rows
    return _evaluate_candidate_groups_serial(mcs_table, problem, grouped_candidates, include_infeasible)


def _evaluate_candidate_groups_parallel(mcs_table, problem, grouped_candidates, include_infeasible, max_workers=None):
    """Try process-based evaluation and fall back to serial mode when unavailable."""
    worker_payloads = [
        (mcs_table, problem, candidates, include_infeasible)
        for candidates in grouped_candidates
    ]
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for group_rows in executor.map(_evaluate_candidate_group_worker, worker_payloads):
                rows.extend(group_rows)
    except (OSError, PermissionError):
        return None
    return rows


def _evaluate_candidate_groups_serial(mcs_table, problem, grouped_candidates, include_infeasible):
    """Evaluate candidate groups in-process while preserving group order."""
    evaluator = DownlinkCandidateEvaluator(mcs_table)
    rows = []
    for candidates in grouped_candidates:
        rows.extend(
            _evaluate_candidate_list(
                evaluator,
                problem,
                candidates,
                include_infeasible=include_infeasible,
            )
        )
    return rows


def _evaluate_candidate_list(evaluator, problem, candidates, include_infeasible):
    """Evaluate one candidate list using a shared evaluator instance."""
    rows = []
    for candidate in candidates:
        row = evaluator.evaluate_candidate(problem, candidate, include_infeasible=include_infeasible)
        if row is not None:
            rows.append(row)
    return rows


def _finalize_candidate_table(rows):
    """Reindex columns and fill default feasibility metadata for the result ledger."""
    candidate_table = pd.DataFrame(rows)
    if candidate_table.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    if "is_feasible" not in candidate_table.columns:
        candidate_table["is_feasible"] = True
    if "infeasibility_reason" not in candidate_table.columns:
        candidate_table["infeasibility_reason"] = "ok"
    return candidate_table.reindex(columns=RESULT_COLUMNS).reset_index(drop=True)


def _annotate_rate_target(candidate_table, required_rate_bps):
    """Mark whether each candidate reaches the request-level rate target."""
    if candidate_table.empty:
        return candidate_table.copy()
    annotated = candidate_table.copy()
    annotated["meets_rate_target"] = False
    if required_rate_bps is None:
        annotated["meets_rate_target"] = annotated["rate_ach_bps"].notna()
        return annotated.reindex(columns=RESULT_COLUMNS)
    annotated.loc[
        annotated["rate_ach_bps"].notna(),
        "meets_rate_target",
    ] = annotated.loc[
        annotated["rate_ach_bps"].notna(),
        "rate_ach_bps",
    ] >= float(required_rate_bps)
    return annotated.reindex(columns=RESULT_COLUMNS)


def _evaluate_candidate_group_worker(payload):
    """Worker entry point for one candidate group in process-based evaluation."""
    mcs_table, problem, candidates, include_infeasible = payload
    evaluator = DownlinkCandidateEvaluator(mcs_table)
    return _evaluate_candidate_list(
        evaluator,
        problem,
        candidates,
        include_infeasible=include_infeasible,
    )


def _build_cache_key_if_enabled(problem, options):
    """Build the memoization key only when caching is enabled."""
    if not options.use_cache:
        return None
    return build_cache_key(
        problem.request,
        problem.model_inputs,
        options,
        pa_catalog=problem.problem.pa_catalog,
    )


def _resolve_run_options(problem_options, options):
    """Merge execution overrides onto the problem's stored search options."""
    if options is None:
        return problem_options
    return replace(
        problem_options,
        include_infeasible=options.include_infeasible,
        parallel=options.parallel,
        max_workers=options.max_workers,
        use_cache=options.use_cache,
    )
