from __future__ import annotations

"""Manifest-row construction for scheduler-comparison campaign outputs."""

from pathlib import Path

from .points import SchedulerComparisonPoint, requested_rate_sum_bps, total_point_demand_bits


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


def build_manifest_row(
    point: SchedulerComparisonPoint,
    *,
    output_dir: Path,
    argv: list[str],
) -> dict[str, object]:
    """Build the lean manifest row emitted once for each campaign point.

    The manifest records the scenario axes needed to reproduce a point, the
    derived demand target, and the exact output/argument strings used by the
    runner. Per-scheduler solver metrics belong in result rows, not here.
    """

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


__all__ = [
    "MANIFEST_COLUMNS",
    "build_manifest_row",
    "validate_manifest_row_contract",
]
def validate_manifest_row_contract(row: dict[str, object]) -> None:
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in row]
    if missing_columns:
        raise ValueError(f"manifest row is missing columns: {missing_columns}")


