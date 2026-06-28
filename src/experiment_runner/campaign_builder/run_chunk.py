from __future__ import annotations

"""Execute one deterministic campaign chunk by delegating points to the runner."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from experiment_runner.result_recording import (
    build_campaign_manifest_row,
    build_campaign_result_row,
    build_certified_skip_result_row,
    print_campaign_chunk_result,
    write_campaign_tables,
)
from experiment_runner.runner import run_experiment_case

from .chunking import select_chunk
from .config_mapping import build_experiment_run_config_for_point
from .points import CampaignPoint
from .pruning import CampaignSkipState


@dataclass(frozen=True)
class CampaignChunkRunResult:
    """Summary of one campaign chunk execution."""

    output_root: Path
    manifest_path: Path
    results_path: Path
    selected_count: int
    solved_count: int
    skipped_count: int


def run_campaign_chunk(
    points: Sequence[CampaignPoint],
    *,
    output_root: Path,
    chunk_index: int,
    chunk_count: int,
    cores: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    argv: Sequence[str] | None = None,
) -> CampaignChunkRunResult:
    """Run the selected campaign chunk and record manifest/result tables.

    Steps:
    1. Select a deterministic chunk without splitting exact-scenario load chains.
    2. Emit manifest rows for every selected point.
    3. Solve or certify-skip points in order, using the official single-case runner.
    4. Write stable CSV tables through the experiment-runner recording boundary.
    """

    selected_points = _apply_limit(
        select_chunk(points, chunk_index=int(chunk_index), chunk_count=int(chunk_count)),
        limit=limit,
    )
    chunk_output_root = _chunk_output_root(Path(output_root), int(chunk_index), int(chunk_count))
    run_argv = tuple(str(arg) for arg in (argv or ()))
    manifest_rows = [
        build_campaign_manifest_row(
            point,
            output_dir=_point_output_dir(chunk_output_root, point),
            argv=run_argv,
        )
        for point in selected_points
    ]

    result_rows = []
    skip_state = CampaignSkipState()
    solved_count = 0
    skipped_count = 0
    if not bool(dry_run):
        for point in selected_points:
            skip_decision = skip_state.decide(point)
            if skip_decision.should_skip:
                result_rows.append(
                    build_certified_skip_result_row(
                        point,
                        source_point_id=skip_decision.source_point_id,
                        source_bound=skip_decision.source_bound,
                        skip_reason=skip_decision.skip_reason,
                    )
                )
                skipped_count += 1
                continue

            config = build_experiment_run_config_for_point(point, cores=int(cores))
            result = run_experiment_case(config)
            result_rows.append(build_campaign_result_row(point, result))
            skip_state.record_result(point, feasible=bool(result.schedule_result.feasible))
            solved_count += 1

    manifest_path, results_path = write_campaign_tables(
        output_root=chunk_output_root,
        manifest_rows=manifest_rows,
        result_rows=result_rows,
    )
    print_campaign_chunk_result(
        output_root=chunk_output_root,
        chunk_index=int(chunk_index),
        chunk_count=int(chunk_count),
        selected_count=len(selected_points),
        solved_count=solved_count,
        skipped_count=skipped_count,
    )
    return CampaignChunkRunResult(
        output_root=chunk_output_root,
        manifest_path=manifest_path,
        results_path=results_path,
        selected_count=len(selected_points),
        solved_count=solved_count,
        skipped_count=skipped_count,
    )


def _apply_limit(points: tuple[CampaignPoint, ...], *, limit: int | None) -> tuple[CampaignPoint, ...]:
    if limit is None:
        return points
    if int(limit) < 0:
        raise ValueError("limit must be non-negative.")
    return points[: int(limit)]


def _chunk_output_root(output_root: Path, chunk_index: int, chunk_count: int) -> Path:
    return Path(output_root) / f"chunk_{int(chunk_index):02d}_of_{int(chunk_count):02d}"


def _point_output_dir(chunk_output_root: Path, point: CampaignPoint) -> Path:
    return Path(chunk_output_root) / str(point.point_id)


__all__ = ["CampaignChunkRunResult", "run_campaign_chunk"]
