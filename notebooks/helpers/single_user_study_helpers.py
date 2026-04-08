from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from configs import (
    SINGLE_USER_SEARCH_CONFIG,
    build_pa_catalog,
    build_pa_characteristics_table,
)
from models import build_resolved_fingerprint
from single_user_solver import search_candidates
from single_user_solver.candidate_space import count_candidates_for_rrc
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem


NOTEBOOK_CONFIG = SINGLE_USER_SEARCH_CONFIG
NOTEBOOK_PA_CATALOG = tuple(build_pa_catalog(NOTEBOOK_CONFIG.pa_data_csv))
NOTEBOOK_N_SLOTS_ON_SPACE = tuple(range(1, int(NOTEBOOK_CONFIG.n_slots_win) + 1))
NOTEBOOK_SEARCH_SHAPE = SearchSpace(
    config=NOTEBOOK_CONFIG,
    n_slots_on_space=NOTEBOOK_N_SLOTS_ON_SPACE,
    layers_space=tuple(int(value) for value in NOTEBOOK_CONFIG.layers_space),
    mcs_space=tuple(int(value) for value in NOTEBOOK_CONFIG.mcs_space),
    prb_step=int(NOTEBOOK_CONFIG.prb_step),
    fingerprint=build_resolved_fingerprint(
        {
            "channel_bw_hz": float(NOTEBOOK_CONFIG.channel_bw_hz),
            "n_slots_on_space": NOTEBOOK_N_SLOTS_ON_SPACE,
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


def build_single_user_scenario(distance_m, required_rate_bps):
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
    return SimpleNamespace(request=request, context=context)


def run_single_user_scenario(scenario):
    """Evaluate one notebook scenario and return the deterministic feasible table."""

    candidate_table = search_candidates(
        scenario.context,
        required_rate_bps=float(scenario.request.required_rate_bps),
    )
    return _add_notebook_columns(candidate_table, scenario)


def preview_single_user_candidates(scenario, limit=5):
    """Return a small deterministic preview of the feasible single-user rows."""

    return run_single_user_scenario(scenario).head(int(limit)).reset_index(drop=True)


def summarize_single_user_scenario(scenario):
    """Build the small scenario views reused across the notebook discussion cells."""

    return {
        "candidate_space_view": _build_candidate_space_view(scenario),
        "example_candidate_view": _build_example_candidate_view(scenario),
        "pa_characteristics": build_pa_characteristics_table(scenario.context.pa_catalog),
    }


def build_single_user_pa_curve_table(scenario):
    """Build PA curve rows for the scenario PA catalog, including idle points."""

    rows = []
    for pa_id, pa in enumerate(scenario.context.pa_catalog):
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
        curve_pin_w = getattr(pa, "curve_pin_w", None)
        curve_pout_w = getattr(pa, "curve_pout_w", None)
        curve_pdc_w = getattr(pa, "curve_pdc_w", None)
        curve_points = zip(
            () if curve_pin_w is None else curve_pin_w,
            () if curve_pout_w is None else curve_pout_w,
            () if curve_pdc_w is None else curve_pdc_w,
        )
        for pin_w, pout_w, pdc_w in curve_points:
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


def _build_candidate_space_view(scenario) -> pd.DataFrame:
    """Describe the fixed single-carrier search envelope shown in Notebook 2."""

    context = scenario.context
    per_pa_counts = tuple(
        (
            str(context.pa_catalog[int(rrc.active_pa_id)].scenario_label),
            int(count_candidates_for_rrc(context.search_catalog, rrc)),
        )
        for rrc in context.rrc_catalog
    )
    max_prbs = max(int(rrc.prb_max) for rrc in context.rrc_catalog)

    return pd.DataFrame(
        [
            {
                "pa_labels": tuple(str(pa.scenario_label) for pa in context.pa_catalog),
                "channel_bandwidth_mhz": float(context.deployment.channel_bw_hz) / 1e6,
                "max_prbs": int(max_prbs),
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
                "raw_candidate_count_per_pa": per_pa_counts,
                "raw_candidate_count_total": int(sum(count for _label, count in per_pa_counts)),
            }
        ]
    )


def _build_example_candidate_view(scenario) -> pd.DataFrame:
    """Build the single-row envelope summary used by the allocation figure."""

    context = scenario.context
    max_prbs = max(int(rrc.prb_max) for rrc in context.rrc_catalog)
    return pd.DataFrame(
        [
            {
                "distance_m": float(scenario.request.distance_m),
                "required_rate_mbps": float(scenario.request.required_rate_bps) / 1e6,
                "resolved_path_loss_db": float(context.deployment.path_loss_db),
                "channel_bandwidth_mhz": float(context.deployment.channel_bw_hz) / 1e6,
                "available_slots": int(context.deployment.n_slots_win),
                "available_prbs": int(max_prbs),
                "available_layers": int(context.deployment.n_tx_chains),
            }
        ]
    )


def _add_notebook_columns(candidate_table, scenario):
    """Attach the few notebook-only compatibility columns to the active table."""

    if candidate_table.empty:
        return pd.DataFrame(columns=NOTEBOOK_RESULT_COLUMNS)

    pa_name_by_id = {
        int(pa_id): str(pa.pa_name)
        for pa_id, pa in enumerate(scenario.context.pa_catalog)
    }
    bandwidth_hz = float(scenario.context.deployment.channel_bw_hz)

    return (
        candidate_table.assign(
            pa_name=candidate_table["pa_id"].map(pa_name_by_id),
            bandwidth_hz=bandwidth_hz,
            bwp_idx=0,
        )
        .sort_values(NOTEBOOK_SORT_COLUMNS)
        .reset_index(drop=True)
        .loc[:, NOTEBOOK_RESULT_COLUMNS]
    )


__all__ = [
    "PROJECT_ROOT",
    "build_single_user_pa_curve_table",
    "build_single_user_scenario",
    "preview_single_user_candidates",
    "run_single_user_scenario",
    "summarize_single_user_scenario",
]
