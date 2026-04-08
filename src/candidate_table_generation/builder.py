from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import pandas as pd

from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_solver.api import enumerate_active_candidates
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

from .console_logging import emit_candidate_table_console_log
from .models import DISTANCE_BIN_GRID_M, DistanceBinnedCandidateTable
from .pruning import prune_candidate_frontier


_DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "distance_binned_candidate_table.json"
)
_CANDIDATE_FRONTIER_DTYPE_MAP = {
    "pa_id": "int64",
    "n_prb": "int64",
    "layers": "int64",
    "mcs": "int64",
    "rate_active_bps": "float64",
    "p_dc_active_w": "float64",
    "p_out_total_w": "float64",
}
_CANDIDATE_FRONTIER_SORT_COLUMNS = ["pa_id", "p_dc_active_w", "n_prb", "mcs", "layers"]


@dataclass(frozen=True)
class _SingleUserEngineState:
    """Shared single-user engine state resolved once for candidate-table generation."""

    model_inputs: object
    search_shape: SearchSpace
    pa_catalog: tuple


def load_or_build_distance_binned_candidate_table(
    path: str | Path | None = None,
    *,
    max_workers: int | None = None,
) -> DistanceBinnedCandidateTable:
    """Load the persisted candidate table when present, otherwise build and save it."""

    artifact_path = _DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    if artifact_path.exists():
        return load_distance_binned_candidate_table(artifact_path)

    candidate_table = build_distance_binned_candidate_table(max_workers=max_workers)
    save_distance_binned_candidate_table(candidate_table, artifact_path)
    return candidate_table


def load_distance_binned_candidate_table(
    path: str | Path | None = None,
) -> DistanceBinnedCandidateTable:
    """Load the persisted candidate table JSON and restore typed frontier tables."""

    artifact_path = _DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frontiers_by_distance_m = {}
    for distance_m, rows in payload["frontiers_by_distance_m"].items():
        frontiers_by_distance_m[int(distance_m)] = (
            pd.DataFrame(rows, columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
            .astype(_CANDIDATE_FRONTIER_DTYPE_MAP)
            .sort_values(_CANDIDATE_FRONTIER_SORT_COLUMNS)
            .reset_index(drop=True)
        )

    return DistanceBinnedCandidateTable(frontiers_by_distance_m=frontiers_by_distance_m)


def save_distance_binned_candidate_table(
    candidate_table: DistanceBinnedCandidateTable,
    path: str | Path | None = None,
) -> Path:
    """Write the distance-binned candidate table JSON artifact to disk."""

    artifact_path = _DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    frontiers_by_distance_m = {}
    for distance_m, frontier in sorted(candidate_table.frontiers_by_distance_m.items()):
        normalized_frontier = (
            frontier.reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
            .astype(_CANDIDATE_FRONTIER_DTYPE_MAP)
            .sort_values(_CANDIDATE_FRONTIER_SORT_COLUMNS)
            .reset_index(drop=True)
        )
        frontiers_by_distance_m[str(int(distance_m))] = [
            [
                int(row.pa_id),
                int(row.n_prb),
                int(row.layers),
                int(row.mcs),
                float(row.rate_active_bps),
                float(row.p_dc_active_w),
                float(row.p_out_total_w),
            ]
            for row in normalized_frontier.itertuples(index=False)
        ]

    payload = {
        "distance_bin_grid_m": [int(distance_m) for distance_m in sorted(candidate_table.frontiers_by_distance_m)],
        "row_columns": list(BATCH_USER_PARAMETER_SPACE_COLUMNS),
        "frontiers_by_distance_m": frontiers_by_distance_m,
    }
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    emit_candidate_table_console_log(
        level=logging.DEBUG,
        stage="store",
        event="save",
        fields=[
            ("file", artifact_path.name),
            ("bins", str(int(len(candidate_table.frontiers_by_distance_m)))),
            (
                "rows",
                str(
                    int(
                        sum(
                            len(frontier)
                            for frontier in candidate_table.frontiers_by_distance_m.values()
                        )
                    )
                ),
            ),
        ],
    )
    return artifact_path


def build_distance_binned_candidate_table(
    *,
    max_workers: int | None = None,
) -> DistanceBinnedCandidateTable:
    """Build the full fixed-grid candidate table used by later user assignment.

    Steps:
    1. Resolve the canonical single-user engine state once.
    2. Enumerate the active feasible table for each configured distance bin.
    3. Keep only full-frame scheduler-facing rows.
    4. Strict-prune dominated rows within each PA family.
    """

    model_inputs = SINGLE_USER_SEARCH_CONFIG
    n_slots_on_space = tuple(range(1, int(model_inputs.n_slots_win) + 1))
    engine_state = _SingleUserEngineState(
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
                build_candidate_frontier_for_distance,
                int(distance_m),
                engine_state=engine_state,
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
    engine_state: _SingleUserEngineState,
) -> pd.DataFrame:
    """Build one strict-pruned scheduler-facing frontier for a fixed distance bin."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=engine_state.model_inputs,
        search_shape=engine_state.search_shape,
        pa_catalog=engine_state.pa_catalog,
    )
    active_table = enumerate_active_candidates(context)
    if active_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    full_frame_table = active_table.loc[
        active_table["n_slots_on"].astype(int) == int(engine_state.model_inputs.n_slots_win)
    ].copy()
    if full_frame_table.empty:
        return pd.DataFrame(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    candidate_table = (
        full_frame_table.assign(
            rate_active_bps=full_frame_table["rate_ach_bps"].astype(float),
            p_dc_active_w=full_frame_table["p_dc_avg_total_w"].astype(float),
        )[BATCH_USER_PARAMETER_SPACE_COLUMNS]
        .sort_values(_CANDIDATE_FRONTIER_SORT_COLUMNS)
        .reset_index(drop=True)
    )
    return prune_candidate_frontier(candidate_table)


__all__ = [
    "build_distance_binned_candidate_table",
    "load_distance_binned_candidate_table",
    "load_or_build_distance_binned_candidate_table",
    "save_distance_binned_candidate_table",
]
