from __future__ import annotations

"""Notebook-facing single-user study setup built on top of the solver packages."""

from dataclasses import dataclass

import pandas as pd

from configs import (
    SINGLE_USER_SEARCH_CONFIG,
    build_pa_catalog,
    build_pa_characteristics_table,
)
from models import build_resolved_fingerprint
from single_user_solver.api import search_candidates
from single_user_solver.candidate_space import count_candidates_for_rrc
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem


NOTEBOOK_CONFIG = SINGLE_USER_SEARCH_CONFIG
NOTEBOOK_PA_CATALOG = tuple(build_pa_catalog(NOTEBOOK_CONFIG.pa_data_csv))
NOTEBOOK_SEARCH_SHAPE = SearchSpace(
    config=NOTEBOOK_CONFIG,
    n_slots_on_space=tuple(range(1, int(NOTEBOOK_CONFIG.frame_n_slots) + 1)),
    layers_space=tuple(int(value) for value in NOTEBOOK_CONFIG.layers_space),
    mcs_space=tuple(int(value) for value in NOTEBOOK_CONFIG.mcs_space),
    prb_step=int(NOTEBOOK_CONFIG.prb_step),
    fingerprint=build_resolved_fingerprint(
        {
            "channel_bw_hz": float(NOTEBOOK_CONFIG.channel_bw_hz),
            "n_slots_on_space": tuple(range(1, int(NOTEBOOK_CONFIG.frame_n_slots) + 1)),
            "layers_space": tuple(int(value) for value in NOTEBOOK_CONFIG.layers_space),
            "mcs_space": tuple(int(value) for value in NOTEBOOK_CONFIG.mcs_space),
            "prb_step": int(NOTEBOOK_CONFIG.prb_step),
        }
    ),
    use_cache=True,
)
NOTEBOOK_RESULT_COLUMNS = [
    "pa_id",
    "pa_name",
    "bandwidth_hz",
    "bwp_idx",
    "n_prb",
    "n_slots_on",
    "layers",
    "mcs",
    "rate_ach_bps",
    "p_dc_avg_total_w",
    "p_out_total_w",
    "gamma_req_lin",
]
NOTEBOOK_SORT_COLUMNS = [
    "p_dc_avg_total_w",
    "bandwidth_hz",
    "n_prb",
    "n_slots_on",
    "layers",
    "mcs",
    "pa_id",
    "bwp_idx",
]


@dataclass(frozen=True)
class SingleUserNotebookScenario:
    """Prepared single-user notebook scenario backed by the production solver."""

    request: SingleUserRequest
    context: object


def build_single_user_scenario(distance_m, required_rate_bps) -> SingleUserNotebookScenario:
    """Prepare one fixed-carrier single-user study case for the notebooks."""

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=float(required_rate_bps),
    )
    context = prepare_single_user_problem(
        request=request,
        model_inputs=NOTEBOOK_CONFIG,
        search_shape=NOTEBOOK_SEARCH_SHAPE,
        pa_catalog=NOTEBOOK_PA_CATALOG,
    )
    return SingleUserNotebookScenario(
        request=request,
        context=context,
    )


def run_single_user_scenario(scenario: SingleUserNotebookScenario) -> pd.DataFrame:
    """Evaluate one notebook scenario and return the deterministic feasible table."""

    return _add_notebook_columns(
        search_candidates(
            scenario.context,
            required_rate_bps=float(scenario.request.required_rate_bps),
        ),
        scenario,
    )


def summarize_single_user_scenario(scenario: SingleUserNotebookScenario) -> dict[str, pd.DataFrame]:
    """Build the small notebook views reused across the scenario walkthroughs."""

    return {
        "candidate_space_view": _build_candidate_space_view(scenario),
        "example_candidate_view": _build_example_candidate_view(scenario),
        "pa_characteristics": build_pa_characteristics_table(scenario.context.pa_catalog),
    }


def build_single_user_pa_curve_table(scenario: SingleUserNotebookScenario) -> pd.DataFrame:
    """Build PA curve rows for the scenario PA catalog, including the quiescent reference point."""

    rows = []
    for pa_id, pa in enumerate(scenario.context.pa_catalog):
        curve_pin_w = tuple(getattr(pa, "curve_pin_w", ()))
        curve_pout_w = tuple(getattr(pa, "curve_pout_w", ()))
        curve_pdc_w = tuple(getattr(pa, "curve_pdc_w", ()))
        rows.append(
            {
                "pa_id": int(pa_id),
                "scenario_label": str(pa.scenario_label),
                "pa_name": str(pa.pa_name),
                "operating_state": "idle",
                "pin_w": 0.0,
                "pout_w": 0.0,
                "pdc_w": float(pa.p_idle_w),
            }
        )
        for pin_w, pout_w, pdc_w in zip(curve_pin_w, curve_pout_w, curve_pdc_w, strict=False):
            rows.append(
                {
                    "pa_id": int(pa_id),
                    "scenario_label": str(pa.scenario_label),
                    "pa_name": str(pa.pa_name),
                    "operating_state": "active",
                    "pin_w": float(pin_w),
                    "pout_w": float(pout_w),
                    "pdc_w": float(pdc_w),
                }
            )

    return pd.DataFrame(rows)


def _build_candidate_space_view(scenario: SingleUserNotebookScenario) -> pd.DataFrame:
    context = scenario.context
    return pd.DataFrame(
        [
            {
                "available_slots": int(context.model_inputs.frame_n_slots),
                "available_prbs": int(max(rrc.prb_max for rrc in context.rrc_catalog)),
                "available_layers": int(context.deployment.n_tx_chains),
                "mcs_values": f"{min(context.search_shape.mcs_space)}-{max(context.search_shape.mcs_space)}",
                "rrc_regions": int(len(context.rrc_catalog)),
                "candidate_count": int(
                    sum(
                        count_candidates_for_rrc(
                            context.search_catalog,
                            rrc,
                        )
                        for rrc in context.rrc_catalog
                    )
                    * len(context.pa_catalog)
                ),
            }
        ]
    )


def _build_example_candidate_view(scenario: SingleUserNotebookScenario) -> pd.DataFrame:
    context = scenario.context
    return pd.DataFrame(
        [
            {
                "available_slots": int(context.model_inputs.frame_n_slots),
                "available_prbs": int(max(rrc.prb_max for rrc in context.rrc_catalog)),
                "available_layers": int(context.deployment.n_tx_chains),
            }
        ]
    )


def _add_notebook_columns(
    candidate_table: pd.DataFrame,
    scenario: SingleUserNotebookScenario,
) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame(columns=NOTEBOOK_RESULT_COLUMNS)

    pa_name_by_id = {
        int(pa_id): str(pa.pa_name)
        for pa_id, pa in enumerate(scenario.context.pa_catalog)
    }
    return (
        candidate_table.assign(
            pa_name=lambda table: table["pa_id"].map(pa_name_by_id),
        )[NOTEBOOK_RESULT_COLUMNS]
        .sort_values(NOTEBOOK_SORT_COLUMNS)
        .reset_index(drop=True)
    )


__all__ = [
    "SingleUserNotebookScenario",
    "build_single_user_pa_curve_table",
    "build_single_user_scenario",
    "run_single_user_scenario",
    "summarize_single_user_scenario",
]
