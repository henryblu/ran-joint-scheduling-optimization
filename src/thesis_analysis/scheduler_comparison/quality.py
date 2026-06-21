from __future__ import annotations

"""Coverage and CSV-quality checks for scheduler-comparison HPC tables."""

from pathlib import Path

import pandas as pd

from .quality_tables import check_row, example_values, point_examples
from .result_contracts import result_quality_check_rows
from .row_states import bool_like, certified_skipped_row_mask


EXPECTED_CHUNK_COUNT = 10
EXPECTED_POINT_COUNT = 20_250

IDENTITY_COLUMNS = (
    "scheduler_mode",
    "switch_policy",
    "active_user_count",
    "load_factor",
    "distance_min_m",
    "distance_max_m",
    "distance_model",
    "mean_distance_m",
    "sigma_distance_m",
    "reference_backlog_bits",
    "frame_duration_s",
)

NUMERIC_IDENTITY_COLUMNS = {
    "active_user_count",
    "load_factor",
    "distance_min_m",
    "distance_max_m",
    "mean_distance_m",
    "sigma_distance_m",
    "reference_backlog_bits",
    "frame_duration_s",
}

def build_point_coverage(manifest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    manifest_counts = count_point_rows(manifest, count_column="manifest_duplicate_count")
    result_counts = count_point_rows(results, count_column="result_duplicate_count")
    manifest_identity = point_identity_frame(manifest, prefix="manifest")
    result_identity = point_identity_frame(results, prefix="result")
    result_status = result_status_frame(results)

    coverage = pd.merge(
        manifest_counts,
        result_counts,
        on=("source_chunk_index", "point_id"),
        how="outer",
    )
    coverage = pd.merge(coverage, manifest_identity, on=("source_chunk_index", "point_id"), how="left")
    coverage = pd.merge(coverage, result_identity, on=("source_chunk_index", "point_id"), how="left")
    coverage = pd.merge(coverage, result_status, on=("source_chunk_index", "point_id"), how="left")

    coverage["manifest_duplicate_count"] = coverage["manifest_duplicate_count"].fillna(0).astype(int)
    coverage["result_duplicate_count"] = coverage["result_duplicate_count"].fillna(0).astype(int)
    coverage["in_manifest"] = coverage["manifest_duplicate_count"] > 0
    coverage["in_results"] = coverage["result_duplicate_count"] > 0
    coverage["identity_fields_match"] = coverage.apply(identity_fields_match, axis=1)
    return coverage.sort_values(["source_chunk_index", "point_id"]).reset_index(drop=True)


def build_sanity_checks(
    chunks: tuple[object, ...],
    manifest: pd.DataFrame,
    results: pd.DataFrame,
    point_coverage: pd.DataFrame,
) -> pd.DataFrame:
    manifest_point_ids = set(manifest["point_id"]) if "point_id" in manifest else set()
    result_point_ids = set(results["point_id"]) if "point_id" in results else set()
    observed_point_ids = set(point_coverage["point_id"]) if "point_id" in point_coverage else set()

    rows = [
        check_row(
            "chunk_count",
            len(chunks) == EXPECTED_CHUNK_COUNT,
            EXPECTED_CHUNK_COUNT - len(chunks),
            f"observed_chunks={len(chunks)} expected_chunks={EXPECTED_CHUNK_COUNT}",
        ),
        check_row(
            "manifest_row_count",
            len(manifest) == EXPECTED_POINT_COUNT,
            EXPECTED_POINT_COUNT - len(manifest),
            f"observed_manifest_rows={len(manifest)} expected_rows={EXPECTED_POINT_COUNT}",
        ),
        check_row(
            "results_row_count",
            len(results) == EXPECTED_POINT_COUNT,
            EXPECTED_POINT_COUNT - len(results),
            f"observed_result_rows={len(results)} expected_rows={EXPECTED_POINT_COUNT}",
        ),
        check_row(
            "observed_unique_point_count",
            len(observed_point_ids) == EXPECTED_POINT_COUNT,
            EXPECTED_POINT_COUNT - len(observed_point_ids),
            f"observed_unique_points={len(observed_point_ids)} expected_points={EXPECTED_POINT_COUNT}",
        ),
        check_row(
            "manifest_result_point_sets_match",
            manifest_point_ids == result_point_ids,
            len(manifest_point_ids.symmetric_difference(result_point_ids)),
            example_values(manifest_point_ids.symmetric_difference(result_point_ids)),
        ),
        check_row(
            "manifest_uses_one_distance_model",
            single_distance_model(manifest),
            distance_model_count(manifest),
            distance_model_examples(manifest),
        ),
        check_row(
            "results_uses_one_distance_model",
            single_distance_model(results),
            distance_model_count(results),
            distance_model_examples(results),
        ),
    ]

    rows.extend(coverage_check_rows(point_coverage))
    rows.extend(result_quality_check_rows(results))
    return pd.DataFrame(rows)


def single_distance_model(frame: pd.DataFrame) -> bool:
    return distance_model_count(frame) == 1


def distance_model_count(frame: pd.DataFrame) -> int:
    if "distance_model" not in frame:
        return 0
    return int(frame["distance_model"].dropna().astype(str).nunique())


def distance_model_examples(frame: pd.DataFrame) -> str:
    if "distance_model" not in frame:
        return "missing distance_model column"
    return "; ".join(sorted(frame["distance_model"].dropna().astype(str).unique()))


def coverage_check_rows(point_coverage: pd.DataFrame) -> list[dict[str, object]]:
    missing_results = point_coverage.loc[~point_coverage["in_results"]]
    missing_manifest = point_coverage.loc[~point_coverage["in_manifest"]]
    duplicate_manifest = point_coverage.loc[point_coverage["manifest_duplicate_count"] > 1]
    duplicate_results = point_coverage.loc[point_coverage["result_duplicate_count"] > 1]
    identity_mismatch = point_coverage.loc[~point_coverage["identity_fields_match"]]
    return [
        check_row("manifest_rows_without_results", missing_results.empty, len(missing_results), point_examples(missing_results)),
        check_row("results_rows_without_manifest", missing_manifest.empty, len(missing_manifest), point_examples(missing_manifest)),
        check_row("duplicate_manifest_point_rows", duplicate_manifest.empty, len(duplicate_manifest), point_examples(duplicate_manifest)),
        check_row("duplicate_result_point_rows", duplicate_results.empty, len(duplicate_results), point_examples(duplicate_results)),
        check_row("manifest_result_identity_match", identity_mismatch.empty, len(identity_mismatch), point_examples(identity_mismatch)),
    ]


def count_point_rows(frame: pd.DataFrame, *, count_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_chunk_index", "point_id", count_column])
    return frame.groupby(["source_chunk_index", "point_id"]).size().reset_index(name=count_column)


def point_identity_frame(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    columns = ["source_chunk_index", "point_id", *IDENTITY_COLUMNS]
    if frame.empty:
        return pd.DataFrame(columns=[f"{prefix}_{column}" if column in IDENTITY_COLUMNS else column for column in columns])

    identity = frame.loc[:, columns].drop_duplicates(["source_chunk_index", "point_id"])
    return identity.rename(columns={column: f"{prefix}_{column}" for column in IDENTITY_COLUMNS})


def result_status_frame(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_chunk_index",
        "point_id",
        "status",
        "feasible",
        "infeasible_reason",
        "skip_reason",
        "source_point_id",
        "source_bound",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    return results.loc[:, columns].drop_duplicates(["source_chunk_index", "point_id"])


def identity_fields_match(row: pd.Series) -> bool:
    if not bool(row["in_manifest"]) or not bool(row["in_results"]):
        return False

    for column in IDENTITY_COLUMNS:
        manifest_value = row[f"manifest_{column}"]
        result_value = row[f"result_{column}"]
        if not identity_values_match(column, manifest_value, result_value):
            return False
    return True


def identity_values_match(column: str, manifest_value: object, result_value: object) -> bool:
    if column not in NUMERIC_IDENTITY_COLUMNS:
        return str(manifest_value) == str(result_value)

    left = pd.to_numeric(pd.Series([manifest_value]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([result_value]), errors="coerce").iloc[0]
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return abs(float(left) - float(right)) <= 1e-9


__all__ = [
    "EXPECTED_CHUNK_COUNT",
    "EXPECTED_POINT_COUNT",
    "build_point_coverage",
    "build_sanity_checks",
    "bool_like",
    "certified_skipped_row_mask",
]
