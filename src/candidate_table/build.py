from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import logging

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_solver.api import enumerate_active_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

from .artifact import CANDIDATE_FRONTIER_SORT_COLUMNS
from .logging import emit_candidate_table_console_log
from .models import DISTANCE_BIN_GRID_M, DistanceBinnedCandidateTable
from .pruning import prune_candidate_frontier


@dataclass(frozen=True)
class _SingleUserEngineState:
    """Shared single-user engine state resolved once for candidate-table generation."""

    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


@lru_cache(maxsize=1)
def _get_single_user_engine_state() -> _SingleUserEngineState:
    """Resolve the static single-user engine state once per process."""

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    n_slots_on_space = (1,)
    return _SingleUserEngineState(
        model_inputs=model_inputs,
        search_shape=SearchSpace(
            config=model_inputs,
            n_slots_on_space=n_slots_on_space,
            layers_space=model_inputs.layers_space,
            mcs_space=model_inputs.mcs_space,
            prb_step=model_inputs.prb_step,
            fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
            use_cache=True,
        ),
        pa_catalog=tuple(build_pa_catalog(model_inputs.pa_data_csv)),
    )


def build_candidate_table(
    *,
    max_workers: int | None = None,
) -> DistanceBinnedCandidateTable:
    """Build the full fixed-grid candidate table used by later user assignment.

    Steps:
    1. Resolve the canonical single-user engine state once.
    2. Enumerate the active feasible table for each configured distance bin.
    3. Project the one-slot scheduler-facing rows.
    4. Strict-prune dominated rows within each PA family.
    """

    _get_single_user_engine_state.cache_clear()
    engine_state = _get_single_user_engine_state()
    frontiers_by_distance_m = {}
    if max_workers is None or int(max_workers) <= 1:
        for distance_m in DISTANCE_BIN_GRID_M:
            frontier = build_candidate_frontier_for_distance(
                int(distance_m),
                engine_state=engine_state,
            )
            frontiers_by_distance_m[int(distance_m)] = frontier
            emit_candidate_table_console_log(
                level=logging.DEBUG,
                stage="build",
                event="bin",
                fields=[
                    ("dist_m", str(int(distance_m))),
                    ("rows", str(int(len(frontier)))),
                ],
            )
        return DistanceBinnedCandidateTable(frontiers_by_distance_m=frontiers_by_distance_m)

    with ProcessPoolExecutor(max_workers=min(int(max_workers), len(DISTANCE_BIN_GRID_M))) as executor:
        future_by_distance = {
            executor.submit(
                _build_candidate_frontier_for_distance_in_worker,
                int(distance_m),
            ): int(distance_m)
            for distance_m in DISTANCE_BIN_GRID_M
        }
        for future in as_completed(future_by_distance):
            distance_m = future_by_distance[future]
            frontier = future.result()
            frontiers_by_distance_m[int(distance_m)] = frontier
            emit_candidate_table_console_log(
                level=logging.DEBUG,
                stage="build",
                event="bin",
                fields=[
                    ("dist_m", str(int(distance_m))),
                    ("rows", str(int(len(frontier)))),
                ],
            )
    return DistanceBinnedCandidateTable(
        frontiers_by_distance_m=dict(sorted(frontiers_by_distance_m.items()))
    )


def build_candidate_frontier_for_distance(
    distance_m: int,
    *,
    engine_state: _SingleUserEngineState | None = None,
) -> pd.DataFrame:
    """Build one strict-pruned slot-normalized frontier for a fixed distance bin."""

    resolved_engine_state = _get_single_user_engine_state() if engine_state is None else engine_state
    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=resolved_engine_state.model_inputs,
        search_shape=resolved_engine_state.search_shape,
        pa_catalog=resolved_engine_state.pa_catalog,
    )
    active_table = enumerate_active_candidates(context)
    if active_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    candidate_table = (
        active_table.assign(
            bits_per_slot=active_table["bits_per_slot"].astype(float),
            p_dc_active_w=active_table["p_dc_active_total_w"].astype(float),
        )[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(CANDIDATE_FRONTIER_SORT_COLUMNS)
        .reset_index(drop=True)
    )
    return prune_candidate_frontier(candidate_table)


def _build_candidate_frontier_for_distance_in_worker(distance_m: int) -> pd.DataFrame:
    """Build one distance-bin frontier inside a worker without parent-state pickling."""

    return build_candidate_frontier_for_distance(int(distance_m))


__all__ = [
    "build_candidate_frontier_for_distance",
    "build_candidate_table",
]
