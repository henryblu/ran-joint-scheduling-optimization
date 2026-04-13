from __future__ import annotations

"""Minimal candidate-table inspection helpers for Notebook 3."""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from candidate_table_generation import DISTANCE_BIN_GRID_M
from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_solver.api import enumerate_active_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem


@dataclass(frozen=True)
class _CandidateTableEngineState:
    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


def _build_full_frame_candidate_table(
    distance_m: int,
    *,
    engine_state: _CandidateTableEngineState | None = None,
) -> pd.DataFrame:
    """Enumerate the full-frame rows for one fixed distance bin before pruning."""

    resolved_engine_state = (
        _resolve_candidate_table_engine_state()
        if engine_state is None
        else engine_state
    )
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

    return (
        active_table.loc[
            active_table["n_slots_on"].astype(int).eq(int(resolved_engine_state.model_inputs.frame_n_slots))
        ]
        .assign(
            rate_active_bps=lambda table: table["rate_ach_bps"].astype(float),
            p_dc_active_w=lambda table: table["p_dc_avg_total_w"].astype(float),
        )[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(["pa_id", "p_dc_active_w", "n_prb", "mcs", "layers"])
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _resolve_candidate_table_engine_state() -> _CandidateTableEngineState:
    model_inputs = SINGLE_USER_SEARCH_CONFIG
    n_slots_on_space = tuple(range(1, int(model_inputs.frame_n_slots) + 1))
    return _CandidateTableEngineState(
        model_inputs=model_inputs,
        search_shape=SearchSpace(
            config=model_inputs,
            n_slots_on_space=n_slots_on_space,
            layers_space=tuple(int(value) for value in model_inputs.layers_space),
            mcs_space=tuple(int(value) for value in model_inputs.mcs_space),
            prb_step=int(model_inputs.prb_step),
            fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
            use_cache=True,
        ),
        pa_catalog=tuple(build_pa_catalog(model_inputs.pa_data_csv)),
    )


def _select_distance_bin(distance_m: int) -> int:
    """Snap one notebook distance onto the configured candidate-table grid."""

    for supported_distance_m in DISTANCE_BIN_GRID_M:
        if int(supported_distance_m) >= int(distance_m):
            return int(supported_distance_m)
    return int(DISTANCE_BIN_GRID_M[-1])


__all__ = [
    "_build_full_frame_candidate_table",
    "_resolve_candidate_table_engine_state",
    "_select_distance_bin",
]
