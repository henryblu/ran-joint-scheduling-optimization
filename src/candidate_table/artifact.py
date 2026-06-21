from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .logging import emit_candidate_table_console_log
from .models import DistanceBinnedCandidateTable


DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "distance_binned_candidate_table.json"
)
CANDIDATE_FRONTIER_DTYPE_MAP = {
    "pa_id": "int64",
    "n_prb": "int64",
    "layers": "int64",
    "mcs": "int64",
    "bits_per_slot": "float64",
    "p_dc_active_w": "float64",
    "p_out_total_w": "float64",
}
CANDIDATE_FRONTIER_SORT_COLUMNS = [
    "pa_id",
    "p_dc_active_w",
    "n_prb",
    "bits_per_slot",
    "mcs",
    "layers",
]


def load_or_build_candidate_table(
    path: str | Path | None = None,
    *,
    max_workers: int | None = None,
) -> DistanceBinnedCandidateTable:
    """Load the persisted candidate table when present, otherwise build and save it."""

    artifact_path = DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    if artifact_path.exists():
        return load_candidate_table(artifact_path)

    from .build import build_candidate_table

    candidate_table = build_candidate_table(max_workers=max_workers)
    save_candidate_table(candidate_table, artifact_path)
    return candidate_table


def load_candidate_table(
    path: str | Path | None = None,
) -> DistanceBinnedCandidateTable:
    """Load the persisted candidate table JSON and restore typed frontier tables."""

    artifact_path = DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frontiers_by_distance_m = {}
    for distance_m, rows in payload["frontiers_by_distance_m"].items():
        frontiers_by_distance_m[int(distance_m)] = (
            pd.DataFrame(rows, columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
            .astype(CANDIDATE_FRONTIER_DTYPE_MAP)
            .sort_values(CANDIDATE_FRONTIER_SORT_COLUMNS)
            .reset_index(drop=True)
        )

    return DistanceBinnedCandidateTable(frontiers_by_distance_m=frontiers_by_distance_m)


def save_candidate_table(
    candidate_table: DistanceBinnedCandidateTable,
    path: str | Path | None = None,
) -> Path:
    """Write the distance-binned candidate table JSON artifact to disk."""

    artifact_path = DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH if path is None else Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    frontiers_by_distance_m = {}
    for distance_m, frontier in sorted(candidate_table.frontiers_by_distance_m.items()):
        normalized_frontier = (
            frontier.reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
            .astype(CANDIDATE_FRONTIER_DTYPE_MAP)
            .sort_values(CANDIDATE_FRONTIER_SORT_COLUMNS)
            .reset_index(drop=True)
        )
        frontiers_by_distance_m[str(int(distance_m))] = [
            [
                int(row.pa_id),
                int(row.n_prb),
                int(row.layers),
                int(row.mcs),
                float(row.bits_per_slot),
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


__all__ = [
    "CANDIDATE_FRONTIER_DTYPE_MAP",
    "CANDIDATE_FRONTIER_SORT_COLUMNS",
    "DEFAULT_DISTANCE_BINNED_CANDIDATE_TABLE_PATH",
    "load_candidate_table",
    "load_or_build_candidate_table",
    "save_candidate_table",
]
