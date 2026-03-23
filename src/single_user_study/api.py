from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import islice

import numpy as np
import pandas as pd

from radio_core import SINGLE_USER_SEARCH_PRESET, build_pa_characteristics_table
from single_user_search.api import (
    enumerate_active_candidates as enumerate_active_candidates_from_engine,
    search_candidates as search_candidates_from_engine,
)
from single_user_search.candidate_space import count_candidates_for_rrc, iter_candidates
from single_user_search.models import SingleUserRequest, SingleUserSearchOptions
from single_user_search.problem_factory import prepare_single_user_problem
from single_user_search.search import filter_rate_feasible_candidates, search_candidates_from_context

from .models import SingleUserScenario


def enumerate_active_candidates(
    distance_m,
    *,
    path_loss_db=None,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Expose the single-user active-table engine through the notebook-facing study module."""

    return enumerate_active_candidates_from_engine(
        distance_m,
        path_loss_db=path_loss_db,
        preset=preset,
        pa_catalog=pa_catalog,
        options=options,
    )


def search_candidates(
    distance_m,
    required_rate_bps,
    *,
    path_loss_db=None,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Expose the single-user rate-filtered search through the notebook-facing study module."""

    return search_candidates_from_engine(
        distance_m,
        required_rate_bps,
        path_loss_db=path_loss_db,
        preset=preset,
        pa_catalog=pa_catalog,
        options=options,
    )


def search_candidate_spaces(
    user_table,
    *,
    preset=None,
    pa_catalog=None,
    options=None,
    outer_parallel=False,
    max_workers=None,
):
    """Build many user candidate tables from the notebook-facing study layer.

    Steps:
    1. Normalize the notebook user table and reject duplicate user ids.
    2. Group users by deployment-equivalent active-table inputs.
    3. Build one active table per unique deployment, either serially or in outer processes.
    4. Filter each shared active table by user target rate and return `user_id -> table`.
    """

    normalized_users = _normalize_user_table(user_table)
    resolved_preset = SINGLE_USER_SEARCH_PRESET if preset is None else preset
    resolved_options = SingleUserSearchOptions(use_cache=True) if options is None else options

    group_requests = {}
    for user_row in normalized_users.itertuples(index=False):
        group_key = _build_user_group_key(user_row)
        if group_key in group_requests:
            continue
        group_requests[group_key] = {
            "distance_m": float(user_row.distance_m),
            "path_loss_db": user_row.path_loss_db,
        }

    grouped_active_tables = {}
    if bool(outer_parallel) and len(group_requests) > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_user_group_worker,
                        group_key,
                        request["distance_m"],
                        request["path_loss_db"],
                        resolved_preset,
                        pa_catalog,
                        resolved_options,
                    ): group_key
                    for group_key, request in group_requests.items()
                }
                for future in as_completed(futures):
                    group_key, active_table = future.result()
                    grouped_active_tables[group_key] = active_table
        except (OSError, PermissionError, BrokenProcessPool, RuntimeError):
            grouped_active_tables = {}

    if not grouped_active_tables:
        for group_key, request in group_requests.items():
            grouped_active_tables[group_key] = enumerate_active_candidates(
                request["distance_m"],
                path_loss_db=request["path_loss_db"],
                preset=resolved_preset,
                pa_catalog=pa_catalog,
                options=resolved_options,
            )

    user_candidate_spaces = {}
    for user_row in normalized_users.itertuples(index=False):
        active_table = grouped_active_tables[_build_user_group_key(user_row)]
        user_candidate_spaces[int(user_row.user_id)] = filter_rate_feasible_candidates(
            active_table,
            required_rate_bps=float(user_row.required_rate_bps),
        )
    return user_candidate_spaces


def build_single_user_scenario(
    distance_m,
    required_rate_bps,
    *,
    path_loss_db=None,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Build the notebook-facing scenario context for one user deployment.

    Steps:
    1. Normalize the scalar notebook inputs into the strict request model.
    2. Resolve the canonical preset and API-owned default runtime policy.
    3. Build the reusable single-user context once for the deployment.
    4. Return the scenario object reused by search and reporting helpers.
    """

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=float(required_rate_bps),
        path_loss_db=None if path_loss_db is None else float(path_loss_db),
    )
    context = prepare_single_user_problem(
        request=request,
        preset=SINGLE_USER_SEARCH_PRESET if preset is None else preset,
        pa_catalog=pa_catalog,
        options=SingleUserSearchOptions(use_cache=True) if options is None else options,
    )
    return SingleUserScenario(request=request, context=context)


def run_single_user_scenario(scenario, *, options=None):
    """Run the candidate-space engine for one prepared study scenario."""

    return search_candidates_from_context(
        scenario.context,
        required_rate_bps=float(scenario.request.required_rate_bps),
        options=options,
    )


def summarize_single_user_scenario(scenario, scenario_count=1):
    """Return the notebook-facing tables that describe one prepared scenario.

    Steps:
    1. Resolve the prepared study context owned by the notebook facade.
    2. Build the deployment, RRC, and search-space summary tables owned by the study layer.
    3. Add API-owned PA and search-space detail views required by notebooks.
    4. Return stable tables without exposing search-package internals.
    """

    context = scenario.context
    return {
        "deployment_summary": _build_deployment_summary_table(context.built_problem),
        "pa_characteristics": build_pa_characteristics_table(context.pa_catalog),
        "rrc_catalog": _build_rrc_catalog_table(context.built_problem),
        "search_space_detail": _build_search_space_detail(context),
        "search_space_summary": _build_search_space_summary_table(
            context.built_problem,
            scenario_count=int(scenario_count),
        ),
    }


def preview_single_user_candidates(scenario, limit=5):
    """Return the first few discrete candidates from one prepared scenario."""

    preview_rows = [
        candidate.__dict__
        for candidate in islice(iter_candidates(scenario.context.built_problem), int(limit))
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
                    "pa_name": str(pa.pa_name),
                    "pin_w": float(pin_w),
                    "pout_w": float(pout_w),
                    "pdc_w": float(pdc_w),
                }
            )
    return pd.DataFrame(rows)


def build_single_user_plot_domains(scenario):
    """Return discrete axis metadata for notebook plots."""

    built_problem = scenario.context.built_problem
    prb_values = sorted(
        {
            int(n_prb)
            for rrc in built_problem.rrc_catalog
            for n_prb in range(
                1,
                int(rrc.prb_max_bwp) + 1,
                max(1, int(built_problem.search_space.prb_step)),
            )
        }
    )
    return {
        "frame_slot_count": int(built_problem.deployment.n_slots_win),
        "layers": _build_integer_axis_config(built_problem.search_space.layers_space),
        "mcs": _build_integer_axis_config(built_problem.search_space.mcs_space),
        "n_prb": _build_integer_axis_config(prb_values),
        "n_slots_on": _build_integer_axis_config(built_problem.search_space.n_slots_on_space),
    }


def _normalize_user_table(user_table):
    """Normalize the notebook batch table and validate its required schema."""

    if not isinstance(user_table, pd.DataFrame):
        raise TypeError("user_table must be a pandas DataFrame.")

    required_columns = {"user_id", "distance_m", "required_rate_bps"}
    missing_columns = sorted(required_columns.difference(user_table.columns))
    if missing_columns:
        raise ValueError(f"user_table is missing required columns: {missing_columns}")

    normalized = user_table.copy()
    if "path_loss_db" not in normalized.columns:
        normalized["path_loss_db"] = None

    normalized["user_id"] = normalized["user_id"].astype(int)
    if normalized["user_id"].duplicated().any():
        duplicate_ids = sorted(normalized.loc[normalized["user_id"].duplicated(), "user_id"].unique())
        raise ValueError(f"user_table contains duplicate user_id values: {duplicate_ids}")

    normalized["distance_m"] = normalized["distance_m"].astype(float)
    normalized["required_rate_bps"] = normalized["required_rate_bps"].astype(float)
    normalized["path_loss_db"] = normalized["path_loss_db"].apply(_normalize_optional_float)
    return normalized[["user_id", "distance_m", "required_rate_bps", "path_loss_db"]]


def _build_user_group_key(user_row):
    """Build the deployment-equivalent key for one notebook batch row."""

    return (
        float(user_row.distance_m),
        _normalize_optional_float(user_row.path_loss_db),
    )


def _evaluate_user_group_worker(group_key, distance_m, path_loss_db, preset, pa_catalog, options):
    """Build one shared active table inside an outer worker process."""

    active_table = enumerate_active_candidates(
        float(distance_m),
        path_loss_db=path_loss_db,
        preset=preset,
        pa_catalog=pa_catalog,
        options=options,
    )
    return group_key, active_table


def _build_deployment_summary_table(built_problem):
    """Build the deployment summary table used by the study notebooks."""

    return pd.DataFrame(
        [
            {
                "distance_m": float(built_problem.deployment.distance_m),
                "path_loss_db": float(built_problem.deployment.path_loss_db),
                "fc_hz": float(built_problem.deployment.fc_hz),
                "n_tx_chains": int(built_problem.deployment.n_tx_chains),
                "n_slots_win": int(built_problem.deployment.n_slots_win),
                "delta_f_hz": (
                    float(built_problem.rrc_catalog[0].delta_f_hz) if built_problem.rrc_catalog else np.nan
                ),
            }
        ]
    )


def _build_rrc_catalog_table(built_problem):
    """Build the notebook-facing RRC envelope table."""

    return pd.DataFrame(
        [
            {
                "pa_id": int(rrc.active_pa_id),
                "bwp_idx": int(rrc.bwp_index),
                "bandwidth_hz": float(rrc.bwp_bw_hz),
                "prb_max_bwp": int(rrc.prb_max_bwp),
                "max_layers": int(rrc.max_layers),
                "max_mcs": int(rrc.max_mcs),
            }
            for rrc in built_problem.rrc_catalog
        ]
    ).sort_values(["pa_id", "bandwidth_hz"]).reset_index(drop=True)


def _build_search_space_summary_table(built_problem, *, scenario_count):
    """Summarize the raw combinatorial search size for notebook inspection."""

    per_pa_counts = []
    for pa_id in range(len(built_problem.pa_catalog)):
        rrc_space = [rrc for rrc in built_problem.rrc_catalog if rrc.active_pa_id == pa_id]
        per_pa_counts.append(
            sum(count_candidates_for_rrc(built_problem, rrc) for rrc in rrc_space)
        )

    raw_configs_per_scenario = sum(per_pa_counts)
    return pd.DataFrame(
        [
            {
                "pa_count": int(len(built_problem.pa_catalog)),
                "scenario_count": int(scenario_count),
                "raw_configs_per_pa_per_scenario": int(per_pa_counts[0]) if per_pa_counts else 0,
                "raw_configs_per_scenario": int(raw_configs_per_scenario),
                "raw_total_configs": int(raw_configs_per_scenario * scenario_count),
                "n_slots_on_values": len(built_problem.search_space.n_slots_on_space),
                "layers_values": len(built_problem.search_space.layers_space),
                "n_active_tx_values": len(built_problem.search_space.n_active_tx_space),
                "mcs_values": len(built_problem.search_space.mcs_space),
                "prb_step": int(built_problem.search_space.prb_step),
            }
        ]
    )


def _build_search_space_detail(context):
    """Return the explicit discrete search dimensions behind the candidate ledger."""

    built_problem = context.built_problem
    bandwidth_space_hz = tuple(sorted({float(rrc.bwp_bw_hz) for rrc in built_problem.rrc_catalog}))
    mcs_space = tuple(int(value) for value in built_problem.search_space.mcs_space)
    return pd.DataFrame(
        [
            {
                "bandwidth_space_hz": bandwidth_space_hz,
                "layers_space": tuple(int(value) for value in built_problem.search_space.layers_space),
                "mcs_min": int(min(mcs_space)) if mcs_space else np.nan,
                "mcs_max": int(max(mcs_space)) if mcs_space else np.nan,
                "prb_step": int(built_problem.search_space.prb_step),
                "mcs_entry_count": int(len(context.mcs_table)),
            }
        ]
    )


def _build_integer_axis_config(values, max_ticks=9, dense_span=20):
    """Build y-axis limits and ticks for a discrete scheduler dimension."""

    normalized_values = sorted({int(value) for value in values})
    if not normalized_values:
        raise ValueError("Discrete axis config requires at least one value.")

    lower = int(normalized_values[0])
    upper = int(normalized_values[-1])
    if lower == upper:
        tick_values = [lower]
    elif upper - lower <= int(dense_span):
        tick_values = list(range(lower, upper + 1))
    else:
        step_candidates = [
            b - a for a, b in zip(normalized_values, normalized_values[1:]) if b > a
        ]
        base_step = max(1, min(step_candidates) if step_candidates else 1)
        approx_step = max(base_step, int(np.ceil((upper - lower) / max(max_ticks - 1, 1))))
        tick_step = int(base_step * np.ceil(approx_step / base_step))
        tick_values = list(range(lower, upper + 1, tick_step))
        if tick_values[-1] != upper:
            tick_values.append(upper)
    return {
        "limits": (float(lower), float(upper)),
        "ticks": tick_values,
    }


def _normalize_optional_float(value):
    """Normalize pandas missing values into `None` for batch-group keys."""

    if pd.isna(value):
        return None
    return float(value)


__all__ = [
    "build_single_user_pa_curve_table",
    "build_single_user_plot_domains",
    "enumerate_active_candidates",
    "search_candidate_spaces",
    "search_candidates",
    "build_single_user_scenario",
    "preview_single_user_candidates",
    "run_single_user_scenario",
    "summarize_single_user_scenario",
]
