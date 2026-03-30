from single_user_solver.api import enumerate_active_candidates, search_candidates

from .batch import (
    search_candidate_space_for_request,
    search_candidate_spaces as _build_batch_user_parameter_space,
)
from .scenario import (
    build_single_user_pa_curve_table,
    build_single_user_scenario,
    preview_single_user_candidates,
    run_single_user_scenario,
    summarize_single_user_scenario,
)


def build_batch_user_parameter_space(user_table):
    """Build the trusted batch single-user parameter-space artifact for one user table."""

    return _build_batch_user_parameter_space(user_table)


def search_candidate_spaces(user_table):
    """Backward-compatible alias for the batch single-user parameter-space entry point."""

    return build_batch_user_parameter_space(user_table)


__all__ = [
    "build_batch_user_parameter_space",
    "build_single_user_pa_curve_table",
    "build_single_user_scenario",
    "enumerate_active_candidates",
    "preview_single_user_candidates",
    "run_single_user_scenario",
    "search_candidate_space_for_request",
    "search_candidate_spaces",
    "search_candidates",
    "summarize_single_user_scenario",
]
