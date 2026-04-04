from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_solver.api import enumerate_active_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

from .models import DISTANCE_BIN_GRID_M, DistanceBinnedCandidateTable
from .pruning import prune_candidate_frontier


@dataclass(frozen=True)
class _SingleUserEngineState:
    """Shared single-user engine state resolved once for candidate-table generation."""

    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


def build_distance_binned_candidate_table() -> DistanceBinnedCandidateTable:
    """Build the full fixed-grid candidate table used by later user assignment.

    Steps:
    1. Resolve the canonical single-user engine state once.
    2. Enumerate the active feasible table for each configured distance bin.
    3. Keep only full-frame scheduler-facing rows.
    4. Strict-prune dominated rows within each PA family.
    """

    engine_state = _resolve_default_single_user_engine_state()
    return DistanceBinnedCandidateTable(
        frontiers_by_distance_m={
            int(distance_m): build_candidate_frontier_for_distance(
                int(distance_m),
                engine_state=engine_state,
            )
            for distance_m in DISTANCE_BIN_GRID_M
        }
    )


def build_candidate_frontier_for_distance(
    distance_m: int,
    *,
    engine_state: _SingleUserEngineState,
) -> pd.DataFrame:
    """Build one strict-pruned scheduler-facing frontier for a fixed distance bin."""

    active_table = _enumerate_active_candidates_for_distance(
        int(distance_m),
        engine_state=engine_state,
    )
    candidate_table = _project_active_table_to_candidate_frontier(
        active_table,
        frame_n_slots=int(engine_state.model_inputs.n_slots_win),
    )
    return prune_candidate_frontier(candidate_table)


def _resolve_default_single_user_engine_state() -> _SingleUserEngineState:
    """Resolve the canonical single-user engine state for the distance-bin sweep."""

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    search_shape = _build_search_space(model_inputs)
    pa_catalog = tuple(build_pa_catalog(model_inputs.pa_data_csv))
    return _SingleUserEngineState(
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )


def _build_search_space(model_inputs) -> SearchSpace:
    """Build the shared search-space shape used during candidate-table generation."""

    n_slots_on_space = tuple(range(1, int(model_inputs.n_slots_win) + 1))
    return SearchSpace(
        config=model_inputs,
        n_slots_on_space=n_slots_on_space,
        layers_space=model_inputs.layers_space,
        mcs_space=model_inputs.mcs_space,
        prb_step=model_inputs.prb_step,
        fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
        use_cache=True,
    )


def _enumerate_active_candidates_for_distance(
    distance_m: int,
    *,
    engine_state: _SingleUserEngineState,
) -> pd.DataFrame:
    """Enumerate the full active feasible table for one fixed distance bin."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=engine_state.model_inputs,
        search_shape=engine_state.search_shape,
        pa_catalog=engine_state.pa_catalog,
    )
    return enumerate_active_candidates(context)


def _project_active_table_to_candidate_frontier(
    active_table: pd.DataFrame,
    *,
    frame_n_slots: int,
) -> pd.DataFrame:
    """Project one raw active table onto the stored scheduler-facing row contract."""

    if active_table.empty or "n_slots_on" not in active_table.columns:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table = active_table[
        active_table["n_slots_on"].astype(int) == int(frame_n_slots)
    ].copy()
    if full_frame_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table["rate_active_bps"] = full_frame_table["rate_ach_bps"].astype(float)
    full_frame_table["p_dc_active_w"] = full_frame_table["p_dc_avg_total_w"].astype(float)
    return (
        full_frame_table[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(
            ["pa_id", "p_dc_active_w", "n_prb", "mcs", "layers"],
            ascending=[True, True, True, True, True],
        )
        .reset_index(drop=True)
    )


__all__ = [
    "build_distance_binned_candidate_table",
]
