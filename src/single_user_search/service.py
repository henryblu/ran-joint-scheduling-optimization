from .problem import (
    iter_requests,
    prepare_single_user_problem,
    prepare_single_user_problem_from_deployment,
)
from .search import run_single_user_search


def build_single_user_context(request, preset, *, pa_catalog=None, options=None):
    """Compatibility wrapper for the refactored single-user problem builder."""
    return prepare_single_user_problem(
        request,
        preset,
        pa_catalog=pa_catalog,
        options=options,
    )


def build_single_user_context_from_deployment(deployment, required_rate_bps, preset, *, pa_catalog=None, options=None):
    """Compatibility wrapper for deployment-based problem construction."""
    return prepare_single_user_problem_from_deployment(
        deployment,
        required_rate_bps,
        preset,
        pa_catalog=pa_catalog,
        options=options,
    )


def search_feasible_configurations(request, preset, *, pa_catalog=None, options=None):
    """Compatibility wrapper for the refactored candidate-ledger search flow."""
    problem = prepare_single_user_problem(
        request,
        preset,
        pa_catalog=pa_catalog,
        options=options,
    )
    return run_single_user_search(problem, options=options)
