from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from candidate_table_generation import DISTANCE_BIN_GRID_M, build_distance_binned_candidate_table
from candidate_table_generation.pruning import prune_candidate_frontier
from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_solver import enumerate_active_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem


@dataclass(frozen=True)
class _CandidateTableEngineState:
    """Shared engine state reused while explaining candidate-table generation."""

    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


def build_candidate_table_generation_artifacts(
    *,
    distance_m: int = 200,
) -> SimpleNamespace:
    """Build the before-and-after views used in Notebook 3.

    Steps:
    1. Build the full distance-binned stored table once across the configured distance grid.
    2. Rebuild one selected distance slice before pruning so the notebook can explain the reduction.
    3. Return the compact summaries and plot-ready tables for that one slice.
    """

    engine_state = _resolve_candidate_table_engine_state()
    selected_distance_m = _select_distance_bin(int(distance_m))
    full_frame_candidate_table = _build_full_frame_candidate_table(
        int(selected_distance_m),
        engine_state=engine_state,
    )
    pruned_frontier_table = prune_candidate_frontier(full_frame_candidate_table)
    pa_label_map = {
        int(pa_id): str(pa.scenario_label)
        for pa_id, pa in enumerate(engine_state.pa_catalog)
    }

    distance_binned_table = build_distance_binned_candidate_table()
    return SimpleNamespace(
        distance_binned_table=distance_binned_table,
        distance_summary=_build_distance_frontier_summary(distance_binned_table),
        selected_distance_m=int(selected_distance_m),
        full_frame_candidate_table=full_frame_candidate_table,
        pruned_frontier_table=pruned_frontier_table,
        pruning_summary=_build_pruning_summary(
            full_frame_candidate_table,
            pruned_frontier_table,
            pa_label_map=pa_label_map,
        ),
        pa_label_map=pa_label_map,
    )


def plot_frontier_compaction(pruning_summary: pd.DataFrame):
    """Plot the row-count reduction caused by the strict frontier pruning step."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_positions = range(len(pruning_summary))
    width = 0.35

    ax.bar(
        [position - width / 2 for position in x_positions],
        pruning_summary["rows_before_pruning"],
        width=width,
        label="Full-frame rows",
    )
    ax.bar(
        [position + width / 2 for position in x_positions],
        pruning_summary["rows_after_pruning"],
        width=width,
        label="Stored frontier rows",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(pruning_summary["pa_label"].tolist())
    ax.set_ylabel("Row count")
    ax.set_title("Strict pruning keeps only non-dominated full-frame rows")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def plot_pruned_frontier(pruned_frontier_table: pd.DataFrame, *, pa_label_map: dict[int, str]):
    """Plot the stored scheduler-facing frontier for one distance bin."""

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for pa_id, pa_rows in pruned_frontier_table.groupby("pa_id", sort=True):
        ax.scatter(
            pa_rows["rate_active_bps"] / 1e6,
            pa_rows["p_dc_active_w"],
            s=32,
            alpha=0.8,
            label=pa_label_map[int(pa_id)],
        )

    ax.set_xlabel("Active rate for the full frame (Mbps)")
    ax.set_ylabel("Active PA DC power (W)")
    ax.set_title("Stored candidate frontier for one distance bin")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def _build_distance_frontier_summary(distance_binned_table) -> pd.DataFrame:
    """Return one compact summary row per precomputed distance bin."""

    rows = []
    for distance_m, frontier_table in sorted(distance_binned_table.frontiers_by_distance_m.items()):
        if frontier_table.empty:
            rows.append(
                {
                    "distance_bin_m": int(distance_m),
                    "stored_rows": 0,
                    "pa_families": 0,
                    "min_rate_mbps": float("nan"),
                    "max_rate_mbps": float("nan"),
                    "min_active_power_w": float("nan"),
                    "max_active_power_w": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "distance_bin_m": int(distance_m),
                "stored_rows": int(len(frontier_table)),
                "pa_families": int(frontier_table["pa_id"].nunique()),
                "min_rate_mbps": float(frontier_table["rate_active_bps"].min()) / 1e6,
                "max_rate_mbps": float(frontier_table["rate_active_bps"].max()) / 1e6,
                "min_active_power_w": float(frontier_table["p_dc_active_w"].min()),
                "max_active_power_w": float(frontier_table["p_dc_active_w"].max()),
            }
        )
    return pd.DataFrame(rows)


def _build_pruning_summary(
    full_frame_candidate_table: pd.DataFrame,
    pruned_frontier_table: pd.DataFrame,
    *,
    pa_label_map: dict[int, str],
) -> pd.DataFrame:
    """Return the per-PA row-count reduction introduced by the pruning step."""

    rows = []
    pa_ids = sorted(
        {
            *full_frame_candidate_table["pa_id"].dropna().astype(int).tolist(),
            *pruned_frontier_table["pa_id"].dropna().astype(int).tolist(),
        }
    )
    for pa_id in pa_ids:
        rows.append(
            {
                "pa_id": int(pa_id),
                "pa_label": pa_label_map[int(pa_id)],
                "rows_before_pruning": int(
                    full_frame_candidate_table["pa_id"].eq(int(pa_id)).sum()
                ),
                "rows_after_pruning": int(
                    pruned_frontier_table["pa_id"].eq(int(pa_id)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_full_frame_candidate_table(
    distance_m: int,
    *,
    engine_state: _CandidateTableEngineState,
) -> pd.DataFrame:
    """Project one distance slice onto the stored scheduler-facing full-frame rows."""

    active_table = _enumerate_active_candidates_for_distance(
        int(distance_m),
        engine_state=engine_state,
    )
    if active_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    frame_n_slots = int(engine_state.model_inputs.n_slots_win)
    full_frame_table = active_table.loc[
        active_table["n_slots_on"].astype(int).eq(frame_n_slots)
    ].copy()
    if full_frame_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table["rate_active_bps"] = full_frame_table["rate_ach_bps"].astype(float)
    full_frame_table["p_dc_active_w"] = full_frame_table["p_dc_avg_total_w"].astype(float)
    full_frame_table["p_out_total_w"] = full_frame_table["p_out_total_w"].astype(float)
    return (
        full_frame_table[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(
            ["pa_id", "n_prb", "p_dc_active_w", "rate_active_bps", "mcs", "layers"],
            ascending=[True, True, True, False, True, True],
        )
        .reset_index(drop=True)
    )


def _enumerate_active_candidates_for_distance(
    distance_m: int,
    *,
    engine_state: _CandidateTableEngineState,
) -> pd.DataFrame:
    """Enumerate the active single-user table for one fixed distance bin."""

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


def _resolve_candidate_table_engine_state() -> _CandidateTableEngineState:
    """Resolve the shared single-user engine state reused by table generation."""

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    n_slots_on_space = tuple(range(1, int(model_inputs.n_slots_win) + 1))
    search_shape = SearchSpace(
        config=model_inputs,
        n_slots_on_space=n_slots_on_space,
        layers_space=model_inputs.layers_space,
        mcs_space=model_inputs.mcs_space,
        prb_step=model_inputs.prb_step,
        fingerprint=build_resolved_fingerprint({"n_slots_on_space": n_slots_on_space}),
        use_cache=True,
    )
    pa_catalog = tuple(build_pa_catalog(model_inputs.pa_data_csv))
    return _CandidateTableEngineState(
        model_inputs=model_inputs,
        search_shape=search_shape,
        pa_catalog=pa_catalog,
    )


def _select_distance_bin(distance_m: int) -> int:
    """Snap one requested distance onto the configured precomputed distance grid."""

    return min(DISTANCE_BIN_GRID_M, key=lambda value: abs(int(value) - int(distance_m)))


__all__ = [
    "PROJECT_ROOT",
    "build_candidate_table_generation_artifacts",
    "plot_frontier_compaction",
    "plot_pruned_frontier",
]
