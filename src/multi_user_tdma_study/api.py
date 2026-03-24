from .study import (
    build_multi_user_tdma_scenario as build_multi_user_tdma_scenario_from_study,
    run_multi_user_tdma_scenario as run_multi_user_tdma_scenario_from_study,
    search_user_candidate_spaces as search_user_candidate_spaces_from_study,
    summarize_multi_user_tdma_scenario as summarize_multi_user_tdma_scenario_from_study,
)
from .user_space import (
    build_user_candidate_review_table as build_user_candidate_review_table_from_user_space,
    build_user_candidate_review_tables as build_user_candidate_review_tables_from_user_space,
    enumerate_user_active_operating_tables as enumerate_user_active_operating_tables_from_user_space,
)


def build_multi_user_tdma_scenario(user_table):
    """Expose the prepared multi-user TDMA scenario builder through the study API."""

    return build_multi_user_tdma_scenario_from_study(user_table)


def run_multi_user_tdma_scenario(
    scenario,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Expose the prepared multi-user TDMA study run through the study API."""

    return run_multi_user_tdma_scenario_from_study(
        scenario,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )


def summarize_multi_user_tdma_scenario(scenario):
    """Expose the notebook-facing multi-user summary tables through the study API."""

    return summarize_multi_user_tdma_scenario_from_study(scenario)


def enumerate_user_active_operating_tables(
    user_table,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Convenience wrapper for enumerating user burst operating catalogs without a scenario."""

    return enumerate_user_active_operating_tables_from_user_space(
        user_table,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )


def build_user_candidate_review_table(candidate_table, *, top_n=None):
    """Return the compact reviewer-facing TDMA table for one user."""

    return build_user_candidate_review_table_from_user_space(
        candidate_table,
        top_n=top_n,
    )


def build_user_candidate_review_tables(user_candidate_spaces, *, top_n=None):
    """Return one compact reviewer-facing TDMA table per user."""

    return build_user_candidate_review_tables_from_user_space(
        user_candidate_spaces,
        top_n=top_n,
    )


def search_user_candidate_spaces(
    user_table,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Convenience wrapper that returns only the exact per-user TDMA candidate spaces."""

    return search_user_candidate_spaces_from_study(
        user_table,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )
