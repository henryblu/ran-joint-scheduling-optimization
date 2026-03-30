from dataclasses import dataclass

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG, USER_REQUIREMENT_COLUMNS, build_pa_catalog
from models import build_resolved_fingerprint
from single_user_solver.api import search_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

from .models import (
    BATCH_USER_REQUIREMENT_COLUMNS,
    BatchUserParameterSpace,
    BATCH_USER_PARAMETER_SPACE_COLUMNS,
)


@dataclass(frozen=True)
class _SingleUserEngineState:
    """Shared single-user engine state resolved once for one batch request."""

    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


def search_candidate_space_for_request(
    request,
    *,
    model_inputs,
    search_shape,
    pa_catalog,
):
    """Solve one single-user candidate space from an explicit user request."""

    context = prepare_single_user_problem(
        request=request,
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )
    return search_candidates(
        context,
        required_rate_bps=float(request.required_rate_bps),
    )


def search_candidate_spaces(user_table):
    """Build the trusted batch parameter-space artifact for one user table.

    Steps:
    1. Normalize the user request table and reject duplicate user ids.
    2. Resolve the canonical single-user engine state once for the whole batch.
    3. Solve each unique `(distance_m, required_rate_bps)` request once.
    4. Project the scheduler-owned full-frame feasible spaces from the raw user tables.
    5. Return one trusted artifact that carries only the shared runtime state each downstream layer owns.
    """

    users = _normalize_user_table(user_table)
    engine_state = _resolve_default_single_user_engine_state()
    user_parameter_spaces = _collect_batch_user_parameter_spaces(
        users,
        engine_state,
    )
    frame_n_slots = int(engine_state.model_inputs.n_slots_win)
    return BatchUserParameterSpace(
        user_requirements=users[BATCH_USER_REQUIREMENT_COLUMNS].copy(),
        user_parameter_spaces=user_parameter_spaces,
        frame_n_slots=frame_n_slots,
        n_tx_chains=int(engine_state.model_inputs.n_tx_chains),
        pa_catalog=engine_state.pa_catalog,
    )


def _collect_batch_user_parameter_spaces(users, engine_state):
    """Solve each unique request once and build one scheduler-facing space per user."""

    frame_n_slots = int(engine_state.model_inputs.n_slots_win)
    grouped_candidate_spaces = {}
    request_groups = {
        (float(user_row.distance_m), float(user_row.required_rate_bps)): SingleUserRequest(
            distance_m=float(user_row.distance_m),
            required_rate_bps=float(user_row.required_rate_bps),
        )
        for user_row in users.itertuples(index=False)
    }
    for group_key, request in request_groups.items():
        raw_candidate_table = search_candidate_space_for_request(
            request,
            model_inputs=engine_state.model_inputs,
            search_shape=engine_state.search_shape,
            pa_catalog=engine_state.pa_catalog,
        )
        grouped_candidate_spaces[group_key] = _project_batch_user_parameter_space(
            raw_candidate_table,
            frame_n_slots=frame_n_slots,
        )

    return {
        int(user_row.user_id): grouped_candidate_spaces[
            (float(user_row.distance_m), float(user_row.required_rate_bps))
        ].copy()
        for user_row in users.itertuples(index=False)
    }


def _normalize_user_table(user_table):
    """Normalize the batch user table and validate its required schema."""

    if not isinstance(user_table, pd.DataFrame):
        raise TypeError("user_table must be a pandas DataFrame.")

    required_columns = set(USER_REQUIREMENT_COLUMNS)
    missing_columns = sorted(required_columns.difference(user_table.columns))
    if missing_columns:
        raise ValueError(f"user_table is missing required columns: {missing_columns}")
    if "path_loss_db" in user_table.columns:
        raise ValueError(
            "user_table must not include path_loss_db; path loss is derived from distance in the shared radio model."
        )

    users = user_table.copy()
    users["user_id"] = users["user_id"].astype(int)
    if users["user_id"].duplicated().any():
        duplicate_ids = sorted(users.loc[users["user_id"].duplicated(), "user_id"].unique())
        raise ValueError(f"user_table contains duplicate user_id values: {duplicate_ids}")

    users["distance_m"] = users["distance_m"].astype(float)
    users["required_rate_bps"] = users["required_rate_bps"].astype(float)
    return users[USER_REQUIREMENT_COLUMNS]


def _project_batch_user_parameter_space(candidate_table, *, frame_n_slots):
    """Build the batch-owned full-frame feasible table for one user."""

    if candidate_table.empty or "n_slots_on" not in candidate_table.columns:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table = candidate_table[
        candidate_table["n_slots_on"].astype(int) == int(frame_n_slots)
    ].copy()
    if full_frame_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table["rate_active_bps"] = full_frame_table["rate_ach_bps"].astype(float)
    full_frame_table["p_dc_active_w"] = full_frame_table["p_dc_avg_total_w"].astype(float)
    full_frame_table["p_out_total_w"] = full_frame_table["p_out_total_w"].astype(float)
    return (
        full_frame_table[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(
            ["p_dc_active_w", "bandwidth_hz", "n_prb", "mcs", "layers", "pa_id"],
            ascending=[True, True, True, True, True, True],
        )
        .reset_index(drop=True)
    )


def _resolve_default_single_user_engine_state():
    """Resolve the canonical single-user engine state owned by the batch layer."""

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    search_shape = _build_search_space(model_inputs)
    pa_catalog = tuple(build_pa_catalog(model_inputs.pa_data_csv))
    return _SingleUserEngineState(
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )


def _build_search_space(model_inputs):
    """Build the shared search-space shape used by the batch parameter-space layer."""

    n_slots_on_space = tuple(range(1, int(model_inputs.n_slots_win) + 1))
    return SearchSpace(
        config=model_inputs,
        bandwidth_space_hz=model_inputs.bandwidth_space_hz,
        n_slots_on_space=n_slots_on_space,
        layers_space=model_inputs.layers_space,
        mcs_space=model_inputs.mcs_space,
        prb_step=model_inputs.prb_step,
        fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
        use_cache=True,
    )


__all__ = [
    "search_candidate_space_for_request",
    "search_candidate_spaces",
]
