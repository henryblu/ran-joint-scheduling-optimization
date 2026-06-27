from __future__ import annotations

"""CSV-first preprocessing for extracted scheduler-comparison HPC chunks."""

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .artifacts import resolve_scheduler_comparison_input_root
from .breakpoints import build_breakpoint_summary, build_infeasibility_reason_summary
from .quality import (
    IDENTITY_COLUMNS,
    build_point_coverage,
    build_sanity_checks,
)
from .reporting import build_markdown_summary
from .summaries import (
    build_load_chain_summary,
    build_policy_summary,
    build_scheduler_summary,
)


CHUNK_NAME_PATTERN = re.compile(r"^chunk_(?P<index>\d{2})_of_(?P<count>\d+)$")
MANIFEST_FILENAME = "scheduler_comparison_manifest.csv"
RESULTS_FILENAME = "scheduler_comparison_results.csv"
SOURCE_METADATA_COLUMNS = (
    "source_chunk_index",
    "source_chunk_name",
    "source_inner_chunk_dir",
    "source_csv_path",
    "row_index_in_source_file",
)
REQUIRED_MANIFEST_COLUMNS = (*SOURCE_METADATA_COLUMNS, "point_id", *IDENTITY_COLUMNS)
REQUIRED_RESULT_COLUMNS = (
    *SOURCE_METADATA_COLUMNS,
    "point_id",
    *IDENTITY_COLUMNS,
    "status",
    "feasible",
    "infeasible_reason",
    "skip_reason",
    "source_point_id",
    "source_bound",
    "total_demand_bits",
    "requested_rate_sum_bps",
    "single_user_elapsed_s",
    "joint_elapsed_s",
    "total_elapsed_s",
    "frame_energy_j",
    "average_frame_dc_power_w",
    "delivered_rate_sum_bps",
)


@dataclass(frozen=True)
class HpcChunkCsvs:
    """The two flat campaign CSVs for one extracted chunk."""

    chunk_index: int
    chunk_name: str
    outer_extraction_dir: Path
    inner_chunk_dir: Path
    manifest_path: Path
    results_path: Path


def preprocess_scheduler_comparison_hpc_results(
    *,
    output_root: Path,
    input_root: Path | None = None,
    artifact_zip: Path | None = None,
    extraction_root: Path | None = None,
    force_extract: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build CSV-first analysis artifacts from the scheduler-comparison sweep.

    Resolve the canonical ZIP or an explicit extracted root, discover chunk
    CSVs, combine manifest/result tables, write local derived outputs, and
    return the analysis tables needed by thesis figure builders.
    """

    resolved_input_root = resolve_scheduler_comparison_input_root(
        input_root=input_root,
        artifact_zip=artifact_zip,
        extraction_root=extraction_root,
        force_extract=force_extract,
    )
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    chunks = discover_chunk_csvs(resolved_input_root)
    chunk_inventory = build_chunk_inventory(chunks)
    combined_manifest = load_combined_csvs(chunks, csv_kind="manifest")
    combined_results = load_combined_csvs(chunks, csv_kind="results")
    validate_required_columns(combined_manifest, REQUIRED_MANIFEST_COLUMNS, table_name="combined_manifest")
    validate_required_columns(combined_results, REQUIRED_RESULT_COLUMNS, table_name="combined_results")
    point_coverage = build_point_coverage(combined_manifest, combined_results)
    sanity_checks = build_sanity_checks(chunks, combined_manifest, combined_results, point_coverage)
    breakpoint_summary = build_breakpoint_summary(combined_results)
    infeasibility_reason_summary = build_infeasibility_reason_summary(combined_results)
    scheduler_summary = build_scheduler_summary(combined_results)
    policy_summary = build_policy_summary(combined_results)
    load_chain_summary = build_load_chain_summary(combined_results)

    write_analysis_outputs(
        output_root=resolved_output_root,
        input_root=resolved_input_root,
        chunk_inventory=chunk_inventory,
        combined_manifest=combined_manifest,
        combined_results=combined_results,
        point_coverage=point_coverage,
        sanity_checks=sanity_checks,
        breakpoint_summary=breakpoint_summary,
        infeasibility_reason_summary=infeasibility_reason_summary,
        scheduler_summary=scheduler_summary,
        policy_summary=policy_summary,
        load_chain_summary=load_chain_summary,
    )

    return {
        "chunk_inventory": chunk_inventory,
        "combined_manifest": combined_manifest,
        "combined_results": combined_results,
        "point_coverage": point_coverage,
        "first_stage_sanity_checks": sanity_checks,
        "breakpoint_summary": breakpoint_summary,
        "infeasibility_reason_summary": infeasibility_reason_summary,
        "scheduler_summary": scheduler_summary,
        "policy_summary": policy_summary,
        "load_chain_summary": load_chain_summary,
    }


def discover_chunk_csvs(input_root: Path) -> tuple[HpcChunkCsvs, ...]:
    """Return extracted chunk CSV locations keyed by inner chunk folder name."""

    chunk_csvs = []
    for candidate in sorted(Path(input_root).rglob("*")):
        if not candidate.is_dir():
            continue

        match = CHUNK_NAME_PATTERN.match(candidate.name)
        if match is None:
            continue

        manifest_path = candidate / MANIFEST_FILENAME
        results_path = candidate / RESULTS_FILENAME
        if not manifest_path.exists() and not results_path.exists():
            continue

        chunk_csvs.append(
            HpcChunkCsvs(
                chunk_index=int(match.group("index")),
                chunk_name=candidate.name,
                outer_extraction_dir=candidate.parent,
                inner_chunk_dir=candidate,
                manifest_path=manifest_path,
                results_path=results_path,
            )
        )

    return tuple(sorted(chunk_csvs, key=lambda chunk: chunk.chunk_index))


def build_chunk_inventory(chunks: tuple[HpcChunkCsvs, ...]) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        manifest_exists = chunk.manifest_path.exists()
        results_exists = chunk.results_path.exists()
        rows.append(
            {
                "chunk_index": int(chunk.chunk_index),
                "chunk_name": str(chunk.chunk_name),
                "outer_extraction_dir": str(chunk.outer_extraction_dir),
                "inner_chunk_dir": str(chunk.inner_chunk_dir),
                "manifest_path": str(chunk.manifest_path),
                "results_path": str(chunk.results_path),
                "manifest_exists": bool(manifest_exists),
                "results_exists": bool(results_exists),
                "manifest_size_bytes": file_size(chunk.manifest_path),
                "results_size_bytes": file_size(chunk.results_path),
                "manifest_row_count": csv_row_count(chunk.manifest_path) if manifest_exists else 0,
                "results_row_count": csv_row_count(chunk.results_path) if results_exists else 0,
            }
        )

    return pd.DataFrame(rows)


def load_combined_csvs(chunks: tuple[HpcChunkCsvs, ...], *, csv_kind: str) -> pd.DataFrame:
    frames = []
    for chunk in chunks:
        csv_path = chunk.manifest_path if csv_kind == "manifest" else chunk.results_path
        frame = pd.read_csv(csv_path)
        frame.insert(0, "row_index_in_source_file", range(len(frame)))
        frame.insert(0, "source_csv_path", str(csv_path))
        frame.insert(0, "source_inner_chunk_dir", str(chunk.inner_chunk_dir))
        frame.insert(0, "source_chunk_name", str(chunk.chunk_name))
        frame.insert(0, "source_chunk_index", int(chunk.chunk_index))
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def validate_required_columns(frame: pd.DataFrame, required_columns: tuple[str, ...], *, table_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def write_analysis_outputs(
    *,
    output_root: Path,
    input_root: Path,
    chunk_inventory: pd.DataFrame,
    combined_manifest: pd.DataFrame,
    combined_results: pd.DataFrame,
    point_coverage: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    breakpoint_summary: pd.DataFrame,
    infeasibility_reason_summary: pd.DataFrame,
    scheduler_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    load_chain_summary: pd.DataFrame,
) -> None:
    chunk_inventory.to_csv(output_root / "chunk_inventory.csv", index=False)
    combined_manifest.to_csv(output_root / "combined_manifest.csv", index=False)
    combined_results.to_csv(output_root / "combined_results.csv", index=False)
    point_coverage.to_csv(output_root / "point_coverage.csv", index=False)
    breakpoint_summary.to_csv(output_root / "breakpoint_summary.csv", index=False)
    infeasibility_reason_summary.to_csv(output_root / "infeasibility_reason_summary.csv", index=False)
    scheduler_summary.to_csv(output_root / "scheduler_summary.csv", index=False)
    policy_summary.to_csv(output_root / "policy_summary.csv", index=False)
    load_chain_summary.to_csv(output_root / "load_chain_summary.csv", index=False)
    sanity_checks.to_csv(output_root / "first_stage_sanity_checks.csv", index=False)
    (output_root / "first_stage_summary.md").write_text(
        build_markdown_summary(
            input_root=input_root,
            output_root=output_root,
            chunk_inventory=chunk_inventory,
            combined_manifest=combined_manifest,
            combined_results=combined_results,
            point_coverage=point_coverage,
            sanity_checks=sanity_checks,
            breakpoint_summary=breakpoint_summary,
            infeasibility_reason_summary=infeasibility_reason_summary,
            scheduler_summary=scheduler_summary,
            policy_summary=policy_summary,
        ),
        encoding="utf-8",
    )


def file_size(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return int(Path(path).stat().st_size)


def csv_row_count(path: Path) -> int:
    return int(len(pd.read_csv(path)))


__all__ = [
    "HpcChunkCsvs",
    "preprocess_scheduler_comparison_hpc_results",
    "discover_chunk_csvs",
]
