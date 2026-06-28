from __future__ import annotations

"""Public console and table recording for experiment runs."""

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from .campaign_builder.points import CampaignPoint, requested_rate_sum_bps, total_point_demand_bits
from .models import ExperimentRunConfig, ExperimentRunResult


MANIFEST_COLUMNS = (
    "point_id",
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
    "total_demand_bits",
    "requested_rate_sum_bps",
    "output_dir",
    "main_argv",
)

CAMPAIGN_RESULT_COLUMNS = (
    "point_id",
    "scheduler_mode",
    "switch_policy",
    "active_user_count",
    "load_factor",
    "distance_model",
    "mean_distance_m",
    "sigma_distance_m",
    "total_demand_bits",
    "requested_rate_sum_bps",
    "status",
    "feasible",
    "infeasible_reason",
    "skip_reason",
    "source_point_id",
    "source_bound",
    "active_slots",
    "allocations",
    "frame_energy_j",
    "average_frame_dc_power_w",
    "delivered_rate_sum_bps",
    "candidate_table_elapsed_s",
    "user_generation_elapsed_s",
    "candidate_lookup_elapsed_s",
    "scheduler_elapsed_s",
    "total_elapsed_s",
)


def print_experiment_result(config: ExperimentRunConfig, result: ExperimentRunResult) -> None:
    """Print the compact summary for a completed finite-frame experiment."""

    schedule_result = result.schedule_result
    solver_details = dict(schedule_result.solver_details)
    power_summary = schedule_result.power_summary
    active_slots = sum(1 for slot in schedule_result.slot_schedules if slot.active)
    allocation_count = sum(len(slot.allocations) for slot in schedule_result.slot_schedules)
    print(
        "EXPERIMENT_RUN",
        f"status={result.status}",
        f"scheduler={schedule_result.scheduler_mode.value}",
        f"algorithm={solver_details.get('algorithm', 'unknown')}",
        f"policy={config.switch_policy.value}",
        f"users={config.user_generation_config.active_user_count}",
        f"load={config.user_generation_config.load_factor:g}",
        f"distance_m={config.user_generation_config.distance_max_m:g}",
    )
    print(
        "EXPERIMENT_RESULT",
        f"feasible={schedule_result.feasible}",
        f"infeasible_reason={schedule_result.infeasible_reason}",
        f"active_slots={active_slots}",
        f"allocations={allocation_count}",
        f"avg_dc_w={power_summary.average_frame_dc_power_w:.9g}",
        f"frame_energy_j={power_summary.frame_energy_j:.9g}",
    )
    print(
        "EXPERIMENT_TIMINGS",
        f"candidate_table_s={result.candidate_table_elapsed_s:.3f}",
        f"user_generation_s={result.user_generation_elapsed_s:.3f}",
        f"candidate_lookup_s={result.candidate_lookup_elapsed_s:.3f}",
        f"scheduler_s={result.scheduler_elapsed_s:.3f}",
        f"total_s={result.total_elapsed_s:.3f}",
    )


def print_campaign_chunk_result(
    *,
    output_root: Path,
    chunk_index: int,
    chunk_count: int,
    selected_count: int,
    solved_count: int,
    skipped_count: int,
) -> None:
    """Print the compact summary for a campaign chunk run."""

    print(
        "CAMPAIGN_CHUNK",
        f"chunk_index={int(chunk_index)}",
        f"chunk_count={int(chunk_count)}",
        f"selected={int(selected_count)}",
        f"solved={int(solved_count)}",
        f"skipped={int(skipped_count)}",
        f"output_root={Path(output_root)}",
    )


def build_campaign_manifest_row(
    point: CampaignPoint,
    *,
    output_dir: Path,
    argv: Sequence[str],
) -> dict[str, object]:
    """Build the reproducibility manifest row for one campaign point."""

    return {
        "point_id": str(point.point_id),
        "scheduler_mode": str(point.scheduler_mode),
        "switch_policy": str(point.switch_policy),
        "active_user_count": int(point.active_user_count),
        "load_factor": float(point.load_factor),
        "distance_min_m": float(point.distance_min_m),
        "distance_max_m": float(point.distance_max_m),
        "distance_model": str(point.distance_model),
        "mean_distance_m": float(point.mean_distance_m),
        "sigma_distance_m": float(point.sigma_distance_m),
        "reference_backlog_bits": int(point.reference_backlog_bits),
        "frame_duration_s": float(point.frame_duration_s),
        "total_demand_bits": int(total_point_demand_bits(point)),
        "requested_rate_sum_bps": float(requested_rate_sum_bps(point)),
        "output_dir": str(Path(output_dir)),
        "main_argv": " ".join(str(arg) for arg in argv),
    }


def build_campaign_result_row(
    point: CampaignPoint,
    result: ExperimentRunResult,
) -> dict[str, object]:
    """Build the canonical result row for one solved campaign point."""

    schedule_result = result.schedule_result
    power_summary = schedule_result.power_summary
    active_slots = sum(1 for slot in schedule_result.slot_schedules if slot.active)
    allocation_count = sum(len(slot.allocations) for slot in schedule_result.slot_schedules)
    return {
        **_campaign_point_result_axes(point),
        "status": str(result.status),
        "feasible": bool(schedule_result.feasible),
        "infeasible_reason": schedule_result.infeasible_reason,
        "skip_reason": "",
        "source_point_id": "",
        "source_bound": "",
        "active_slots": int(active_slots),
        "allocations": int(allocation_count),
        "frame_energy_j": float(power_summary.frame_energy_j),
        "average_frame_dc_power_w": float(power_summary.average_frame_dc_power_w),
        "delivered_rate_sum_bps": float(
            sum(summary.delivered_rate_bps for summary in schedule_result.user_summaries)
        ),
        "candidate_table_elapsed_s": float(result.candidate_table_elapsed_s),
        "user_generation_elapsed_s": float(result.user_generation_elapsed_s),
        "candidate_lookup_elapsed_s": float(result.candidate_lookup_elapsed_s),
        "scheduler_elapsed_s": float(result.scheduler_elapsed_s),
        "total_elapsed_s": float(result.total_elapsed_s),
    }


def build_certified_skip_result_row(
    point: CampaignPoint,
    *,
    source_point_id: str,
    source_bound: float,
    skip_reason: str,
) -> dict[str, object]:
    """Build a result row for a point skipped by a trusted campaign pruning rule."""

    return {
        **_campaign_point_result_axes(point),
        "status": "skipped",
        "feasible": False,
        "infeasible_reason": "certified_skip",
        "skip_reason": str(skip_reason),
        "source_point_id": str(source_point_id),
        "source_bound": float(source_bound),
        "active_slots": "",
        "allocations": "",
        "frame_energy_j": "",
        "average_frame_dc_power_w": "",
        "delivered_rate_sum_bps": "",
        "candidate_table_elapsed_s": 0.0,
        "user_generation_elapsed_s": 0.0,
        "candidate_lookup_elapsed_s": 0.0,
        "scheduler_elapsed_s": 0.0,
        "total_elapsed_s": 0.0,
    }


def write_campaign_tables(
    *,
    output_root: Path,
    manifest_rows: Iterable[dict[str, object]],
    result_rows: Iterable[dict[str, object]],
) -> tuple[Path, Path]:
    """Write campaign manifest and result CSVs with stable schemas."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    results_path = output_root / "results.csv"
    _write_csv(manifest_path, MANIFEST_COLUMNS, manifest_rows)
    _write_csv(results_path, CAMPAIGN_RESULT_COLUMNS, result_rows)
    return manifest_path, results_path


def _campaign_point_result_axes(point: CampaignPoint) -> dict[str, object]:
    return {
        "point_id": str(point.point_id),
        "scheduler_mode": str(point.scheduler_mode),
        "switch_policy": str(point.switch_policy),
        "active_user_count": int(point.active_user_count),
        "load_factor": float(point.load_factor),
        "distance_model": str(point.distance_model),
        "mean_distance_m": float(point.mean_distance_m),
        "sigma_distance_m": float(point.sigma_distance_m),
        "total_demand_bits": int(total_point_demand_bits(point)),
        "requested_rate_sum_bps": float(requested_rate_sum_bps(point)),
    }


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "CAMPAIGN_RESULT_COLUMNS",
    "MANIFEST_COLUMNS",
    "build_campaign_manifest_row",
    "build_campaign_result_row",
    "build_certified_skip_result_row",
    "print_campaign_chunk_result",
    "print_experiment_result",
    "write_campaign_tables",
]
