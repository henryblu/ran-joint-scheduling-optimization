from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import islice

import numpy as np
import pandas as pd

from pa_models import build_pa_catalog, build_pa_characteristics_table
from radio_configs import SINGLE_USER_SEARCH_CONFIG
from radio_models import build_resolved_fingerprint
from single_user_search.api import enumerate_active_candidates, search_candidates
from single_user_search.candidate_space import count_candidates_for_rrc, iter_candidates
from single_user_search.models import SearchSpace, SingleUserRequest
from single_user_search.problem_factory import prepare_single_user_problem
from single_user_search.search import (
    filter_rate_feasible_candidates,
)

from .models import SingleUserScenario


def search_candidate_spaces(
    user_table,
    *,
    outer_parallel=False,
    max_workers=None,
):
    """Build many user candidate tables from the notebook-facing study layer.

    Steps:
    1. Normalize the notebook user table and reject duplicate user ids.
    2. Resolve the canonical single-user engine state once for the whole batch.
    3. Build one active table per unique deployment, either serially or in outer processes.
    4. Filter each shared active table by user target rate and return `user_id -> table`.
    """

    normalized_users = _normalize_user_table(user_table)
    model_inputs, search_shape, pa_catalog = _resolve_default_single_user_engine_state()

    group_requests = {}
    for user_row in normalized_users.itertuples(index=False):
        group_key = float(user_row.distance_m)
        if group_key in group_requests:
            continue
        group_requests[group_key] = float(user_row.distance_m)

    grouped_active_tables = {}
    if bool(outer_parallel) and len(group_requests) > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_user_group_worker,
                        group_key,
                        distance_m,
                        model_inputs,
                        search_shape,
                        pa_catalog,
                    ): group_key
                    for group_key, distance_m in group_requests.items()
                }
                for future in as_completed(futures):
                    group_key, active_table = future.result()
                    grouped_active_tables[group_key] = active_table
        except (OSError, PermissionError, BrokenProcessPool, RuntimeError):
            grouped_active_tables = {}

    if not grouped_active_tables:
        for group_key, distance_m in group_requests.items():
            grouped_active_tables[group_key] = _build_active_table_for_distance(
                distance_m,
                model_inputs=model_inputs,
                search_shape=search_shape,
                pa_catalog=pa_catalog,
            )

    user_candidate_spaces = {}
    for user_row in normalized_users.itertuples(index=False):
        active_table = grouped_active_tables[float(user_row.distance_m)]
        user_candidate_spaces[int(user_row.user_id)] = filter_rate_feasible_candidates(
            active_table,
            required_rate_bps=float(user_row.required_rate_bps),
        )
    return user_candidate_spaces


def build_single_user_scenario(
    distance_m,
    required_rate_bps,
):
    """Build the notebook-facing scenario context for one user deployment.

    Steps:
    1. Normalize the scalar notebook inputs into the strict request model.
    2. Resolve the canonical preset-owned engine state once at the study boundary.
    3. Build the reusable single-user context once for the deployment.
    4. Return the scenario object reused by search and reporting helpers.
    """

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=float(required_rate_bps),
    )
    model_inputs, search_shape, pa_catalog = _resolve_default_single_user_engine_state()
    context = prepare_single_user_problem(
        request=request,
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )
    return SingleUserScenario(request=request, context=context)


def run_single_user_scenario(scenario):
    """Run the candidate-space engine for one prepared study scenario."""

    return search_candidates(
        scenario.context,
        required_rate_bps=float(scenario.request.required_rate_bps),
    )


def summarize_single_user_scenario(scenario, scenario_count=1):
    """Return the notebook-facing tables that describe one prepared scenario.

    Steps:
        1. Resolve the prepared study context owned by the notebook facade.
        2. Build one raw candidate-space table that explains the structural search domain.
        3. Solve the single-user scenario once and choose the deterministic illustrative candidate.
        4. Return only the non-overlapping notebook tables plus PA characteristics.
    """

    context = scenario.context
    feasible_table = run_single_user_scenario(scenario)
    example_candidate_row = _select_example_candidate_row(feasible_table)
    return {
        "candidate_space_view": _build_candidate_space_view(
            context,
            scenario_count=int(scenario_count),
        ),
        "example_candidate_view": _build_example_candidate_view(context, example_candidate_row),
        "pa_characteristics": build_pa_characteristics_table(context.pa_catalog),
    }


def preview_single_user_candidates(scenario, limit=5):
    """Return the first few discrete candidates from one prepared scenario."""

    preview_rows = [
        candidate.__dict__
        for candidate in islice(iter_candidates(scenario.context.search_catalog), int(limit))
    ]
    return pd.DataFrame(preview_rows)


def build_single_user_pa_curve_table(scenario):
    """Return one row per measured PA curve point for notebook plotting."""

    rows = []
    for pa_id, pa in enumerate(scenario.context.pa_catalog):
        pout_values = np.asarray(getattr(pa, "curve_pout_w", []), dtype=float)
        pin_values = np.asarray(getattr(pa, "curve_pin_w", []), dtype=float)
        pdc_values = np.asarray(getattr(pa, "curve_pdc_w", []), dtype=float)
        for pin_w, pout_w, pdc_w in zip(pin_values, pout_values, pdc_values):
            rows.append(
                {
                    "pa_id": int(pa_id),
                    "scenario_label": str(pa.scenario_label),
                    "pa_name": str(pa.pa_name),
                    "pin_w": float(pin_w),
                    "pout_w": float(pout_w),
                    "pdc_w": float(pdc_w),
                }
            )
    return pd.DataFrame(rows)


def _normalize_user_table(user_table):
    """Normalize the notebook batch table and validate its required schema."""

    if not isinstance(user_table, pd.DataFrame):
        raise TypeError("user_table must be a pandas DataFrame.")

    required_columns = {"user_id", "distance_m", "required_rate_bps"}
    missing_columns = sorted(required_columns.difference(user_table.columns))
    if missing_columns:
        raise ValueError(f"user_table is missing required columns: {missing_columns}")
    if "path_loss_db" in user_table.columns:
        raise ValueError("user_table must not include path_loss_db; path loss is derived from distance in the shared radio model.")

    normalized = user_table.copy()
    normalized["user_id"] = normalized["user_id"].astype(int)
    if normalized["user_id"].duplicated().any():
        duplicate_ids = sorted(normalized.loc[normalized["user_id"].duplicated(), "user_id"].unique())
        raise ValueError(f"user_table contains duplicate user_id values: {duplicate_ids}")

    normalized["distance_m"] = normalized["distance_m"].astype(float)
    normalized["required_rate_bps"] = normalized["required_rate_bps"].astype(float)
    return normalized[["user_id", "distance_m", "required_rate_bps"]]


def _resolve_default_single_user_engine_state():
    """Resolve the canonical single-user engine state owned by the study boundary."""

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    search_shape = _build_search_space(model_inputs)
    pa_catalog = tuple(build_pa_catalog(model_inputs.pa_data_csv))
    return model_inputs, search_shape, pa_catalog


def _build_active_table_for_distance(distance_m, *, model_inputs, search_shape, pa_catalog):
    """Build one active candidate table for a resolved deployment distance."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )
    return enumerate_active_candidates(context)


def _evaluate_user_group_worker(group_key, distance_m, model_inputs, search_shape, pa_catalog):
    """Build one shared active table inside an outer worker process."""

    active_table = _build_active_table_for_distance(
        distance_m,
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )
    return group_key, active_table


def _build_candidate_space_view(context, *, scenario_count):
    """Return the compact definition and size of the raw structural candidate space."""

    pa_labels = tuple(str(pa.scenario_label) for pa in context.pa_catalog)
    bandwidth_options_hz = tuple(sorted({float(rrc.bwp_bw_hz) for rrc in context.rrc_catalog}))
    max_prbs_by_bwp = tuple(
        (
            str(context.pa_catalog[int(rrc.active_pa_id)].scenario_label),
            int(rrc.bwp_index),
            int(rrc.prb_max_bwp),
        )
        for rrc in sorted(
            context.rrc_catalog,
            key=lambda item: (int(item.active_pa_id), float(item.bwp_bw_hz), int(item.bwp_index)),
        )
    )
    per_pa_counts = []
    for pa_id in range(len(context.pa_catalog)):
        rrc_space = [rrc for rrc in context.rrc_catalog if rrc.active_pa_id == pa_id]
        per_pa_counts.append(
            (
                str(context.pa_catalog[int(pa_id)].scenario_label),
                int(sum(count_candidates_for_rrc(context.search_catalog, rrc) for rrc in rrc_space)),
            )
        )

    raw_candidate_count_total = int(sum(count for _label, count in per_pa_counts))
    return pd.DataFrame(
        [
            {
                "pa_labels": pa_labels,
                "bandwidth_options_hz": bandwidth_options_hz,
                "max_prbs_by_bwp": max_prbs_by_bwp,
                "slot_domain": (1, int(context.deployment.n_slots_win)),
                "layer_domain": (
                    int(min(context.search_shape.layers_space)),
                    int(max(context.search_shape.layers_space)),
                ),
                "mcs_domain": (
                    int(min(context.search_shape.mcs_space)),
                    int(max(context.search_shape.mcs_space)),
                ),
                "prb_step": int(context.search_shape.prb_step),
                "raw_candidate_count_per_pa": tuple(per_pa_counts),
                "raw_candidate_count_total": raw_candidate_count_total,
                "raw_candidate_count_across_scenarios": int(raw_candidate_count_total * scenario_count),
            }
        ]
    )


def _build_example_candidate_view(context, example_candidate_row):
    """Return the deterministic illustrative feasible candidate with its envelope."""

    selected_rrc = next(
        rrc
        for rrc in context.rrc_catalog
        if int(rrc.active_pa_id) == int(example_candidate_row["pa_id"])
        and int(rrc.bwp_index) == int(example_candidate_row["bwp_idx"])
    )
    return pd.DataFrame(
        [
            {
                "scenario_label": str(example_candidate_row["scenario_label"]),
                "pa_name": str(example_candidate_row["pa_name"]),
                "bandwidth_hz": float(example_candidate_row["bandwidth_hz"]),
                "bwp_idx": int(example_candidate_row["bwp_idx"]),
                "allocated_prbs": int(example_candidate_row["n_prb"]),
                "available_prbs": int(selected_rrc.prb_max_bwp),
                "allocated_slots": int(example_candidate_row["n_slots_on"]),
                "available_slots": int(context.deployment.n_slots_win),
                "allocated_layers": int(example_candidate_row["layers"]),
                "available_layers": int(selected_rrc.max_layers),
                "mcs": int(example_candidate_row["mcs"]),
                "rate_ach_bps": float(example_candidate_row["rate_ach_bps"]),
                "window_avg_total_pa_dc_w": float(example_candidate_row["p_dc_avg_total_w"]),
            }
        ]
    )


def _select_example_candidate_row(feasible_table):
    """Return the stable illustrative candidate from the feasible cloud."""

    if feasible_table.empty:
        raise ValueError("Cannot build an example candidate view from an empty feasible table.")
    return (
        feasible_table.sort_values(
            [
                "p_dc_avg_total_w",
                "bandwidth_hz",
                "n_prb",
                "n_slots_on",
                "layers",
                "mcs",
                "pa_id",
                "bwp_idx",
            ]
        )
        .reset_index(drop=True)
        .iloc[0]
    )


def _build_search_space(model_inputs):
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
    "build_single_user_pa_curve_table",
    "enumerate_active_candidates",
    "search_candidate_spaces",
    "search_candidates",
    "build_single_user_scenario",
    "preview_single_user_candidates",
    "run_single_user_scenario",
    "summarize_single_user_scenario",
]
