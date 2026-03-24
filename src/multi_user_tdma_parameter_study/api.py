from .study import (
    build_multi_user_tdma_scenario,
    run_multi_user_tdma_scenario,
    search_user_candidate_spaces,
    summarize_multi_user_tdma_scenario,
)
from .user_space import (
    build_user_candidate_review_table,
    build_user_candidate_review_tables,
    enumerate_user_active_operating_tables as enumerate_user_active_operating_tables_from_scenario,
)


def enumerate_user_active_operating_tables(
    user_table,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Convenience boundary that resolves the canonical multi-user scenario once."""

    scenario = build_multi_user_tdma_scenario(user_table)
    return enumerate_user_active_operating_tables_from_scenario(
        scenario.user_table,
        system_cfg=scenario.system_cfg,
        model_inputs=scenario.active_search_model_inputs,
        search_shape=scenario.active_search_shape,
        pa_catalog=scenario.pa_catalog,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )

