from __future__ import annotations

"""CSV-first preprocessing for extracted scheduler-comparison HPC chunks."""

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .artifacts import resolve_scheduler_comparison_input_root
from .breakpoints import (
    build_breakpoint_summary,
    build_infeasibility_reason_summary,
)
from .quality import (
    EXPECTED_POINT_COUNT,
    build_point_coverage,
    build_sanity_checks,
)


CHUNK_NAME_PATTERN = re.compile(r"^chunk_(?P<index>\d{2})_of_(?P<count>\d+)$")
MANIFEST_FILENAME = "scheduler_comparison_manifest.csv"
RESULTS_FILENAME = "scheduler_comparison_results.csv"


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
    point_coverage = build_point_coverage(combined_manifest, combined_results)
    sanity_checks = build_sanity_checks(chunks, combined_manifest, combined_results, point_coverage)
    breakpoint_summary = build_breakpoint_summary(combined_results)
    infeasibility_reason_summary = build_infeasibility_reason_summary(combined_results)

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
    )

    return {
        "chunk_inventory": chunk_inventory,
        "combined_manifest": combined_manifest,
        "combined_results": combined_results,
        "point_coverage": point_coverage,
        "first_stage_sanity_checks": sanity_checks,
        "breakpoint_summary": breakpoint_summary,
        "infeasibility_reason_summary": infeasibility_reason_summary,
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
) -> None:
    chunk_inventory.to_csv(output_root / "chunk_inventory.csv", index=False)
    combined_manifest.to_csv(output_root / "combined_manifest.csv", index=False)
    combined_results.to_csv(output_root / "combined_results.csv", index=False)
    point_coverage.to_csv(output_root / "point_coverage.csv", index=False)
    breakpoint_summary.to_csv(output_root / "breakpoint_summary.csv", index=False)
    infeasibility_reason_summary.to_csv(output_root / "infeasibility_reason_summary.csv", index=False)
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
        ),
        encoding="utf-8",
    )


def build_markdown_summary(
    *,
    input_root: Path,
    output_root: Path,
    chunk_inventory: pd.DataFrame,
    combined_manifest: pd.DataFrame,
    combined_results: pd.DataFrame,
    point_coverage: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    breakpoint_summary: pd.DataFrame,
    infeasibility_reason_summary: pd.DataFrame,
) -> str:
    failed_checks = sanity_checks.loc[~sanity_checks["passed"]]
    unexpected_breakpoints = breakpoint_summary.loc[breakpoint_summary["unexpected_breakpoint_flag"]]
    first_breaks = breakpoint_summary.loc[breakpoint_summary["breakpoint_category"] != "all_solved"]
    status_counts = combined_results.groupby(["scheduler_mode", "switch_policy", "status"]).size().reset_index(name="count")
    breakpoint_counts = breakpoint_summary.groupby(["scheduler_mode", "switch_policy", "breakpoint_category"]).size().reset_index(name="count")

    lines = [
        "# Scheduler Comparison HPC First-Stage Summary",
        "",
        f"Input root: `{input_root}`",
        f"Output root: `{output_root}`",
        "",
        "## Coverage",
        "",
        f"- Chunks discovered: {len(chunk_inventory)}",
        f"- Manifest rows: {len(combined_manifest)}",
        f"- Result rows: {len(combined_results)}",
        f"- Expected grid points: {EXPECTED_POINT_COUNT}",
        f"- Coverage rows: {len(point_coverage)}",
        f"- Failed sanity checks: {len(failed_checks)}",
        "",
        "## Breakpoints",
        "",
        f"- Load chains: {len(breakpoint_summary)}",
        f"- Chains with a first non-solved row: {len(first_breaks)}",
        f"- Chains with solved rows after an earlier non-solved row: {len(unexpected_breakpoints)}",
        "",
        "### Breakpoint Categories",
        "",
        markdown_table(breakpoint_counts.head(40)),
        "",
        "### Earliest Breakpoints",
        "",
        markdown_table(earliest_breakpoint_table(first_breaks)),
        "",
        "### Unexpected Nonmonotone Breakpoints",
        "",
        markdown_table(unexpected_breakpoint_table(unexpected_breakpoints)),
        "",
        "## Status Distribution",
        "",
        markdown_table(status_counts),
        "",
        "## Most Common Infeasibility Reasons",
        "",
        markdown_table(infeasibility_reason_summary.head(30)),
        "",
        "## Failed Sanity Checks",
        "",
        markdown_table(failed_checks),
        "",
        "## Recommended JSON Inspection Targets",
        "",
        "- First non-solved row in each load chain from `breakpoint_summary.csv`.",
        "- Rows where `unexpected_breakpoint_flag` is true.",
        "- Certified skipped rows whose source metadata fails sanity checks.",
        "- Failed or other-status rows before ordinary infeasibility boundaries.",
        "",
    ]
    return "\n".join(lines)


def earliest_breakpoint_table(first_breaks: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scheduler_mode",
        "switch_policy",
        "active_user_count",
        "distance_model",
        "mean_distance_m",
        "sigma_distance_m",
        "first_unsolved_load_factor",
        "first_unsolved_status",
        "first_unsolved_reason",
        "breakpoint_category",
    ]
    return first_breaks.sort_values(
        ["first_unsolved_load_factor", "scheduler_mode", "switch_policy", "active_user_count"],
        na_position="last",
    ).loc[:, columns].head(30)


def unexpected_breakpoint_table(unexpected_breakpoints: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scheduler_mode",
        "switch_policy",
        "active_user_count",
        "distance_model",
        "mean_distance_m",
        "sigma_distance_m",
        "first_unsolved_load_factor",
        "first_unsolved_status",
        "last_solved_load_factor",
        "first_unsolved_point_id",
    ]
    return unexpected_breakpoints.loc[:, columns].head(30)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"

    display = frame.fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(escape_markdown_cell(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def file_size(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return int(Path(path).stat().st_size)


def csv_row_count(path: Path) -> int:
    return int(len(pd.read_csv(path)))


__all__ = [
    "preprocess_scheduler_comparison_hpc_results",
    "discover_chunk_csvs",
]
