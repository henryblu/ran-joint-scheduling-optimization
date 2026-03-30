from itertools import islice

import numpy as np
import pandas as pd

from configs import build_pa_characteristics_table
from single_user_solver.candidate_space import count_candidates_for_rrc, iter_candidates
from single_user_solver.models import SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

from .batch import _resolve_default_single_user_engine_state, search_candidate_space_for_request
from .models import SingleUserScenario


def build_single_user_scenario(
    distance_m,
    required_rate_bps,
):
    """Build the notebook-facing scenario context for one user deployment.

    Steps:
    1. Normalize the scalar notebook inputs into the strict request model.
    2. Resolve the canonical single-user engine state owned by the batch-study core.
    3. Build the reusable single-user context once for the deployment.
    4. Return the scenario object reused by the story and reporting helpers.
    """

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=float(required_rate_bps),
    )
    engine_state = _resolve_default_single_user_engine_state()
    context = prepare_single_user_problem(
        request=request,
        model_inputs=engine_state.model_inputs,
        search_shape=engine_state.search_shape,
        pa_catalog=engine_state.pa_catalog,
    )
    return SingleUserScenario(request=request, context=context)


def run_single_user_scenario(scenario):
    """Run the targeted candidate-space engine for one prepared study scenario."""

    return search_candidate_space_for_request(
        scenario.request,
        model_inputs=scenario.context.model_inputs,
        search_shape=scenario.context.search_shape,
        pa_catalog=scenario.context.pa_catalog,
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

    selected_pa = context.pa_catalog[int(example_candidate_row["pa_id"])]
    selected_rrc = next(
        rrc
        for rrc in context.rrc_catalog
        if int(rrc.active_pa_id) == int(example_candidate_row["pa_id"])
        and int(rrc.bwp_index) == int(example_candidate_row["bwp_idx"])
    )
    return pd.DataFrame(
        [
            {
                "scenario_label": str(selected_pa.scenario_label),
                "pa_name": str(selected_pa.pa_name),
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


__all__ = [
    "build_single_user_pa_curve_table",
    "build_single_user_scenario",
    "preview_single_user_candidates",
    "run_single_user_scenario",
    "summarize_single_user_scenario",
]
