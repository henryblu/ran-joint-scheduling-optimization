"""Scheduler-comparison campaign grid, chunking, and manifest contracts."""

from .chunking import (
    exact_scenario_key,
    group_points_by_exact_scenario,
    order_scheduler_comparison_points,
    scheduler_comparison_run_order_key,
    select_chunk,
)
from .manifest import MANIFEST_COLUMNS, build_manifest_row
from .points import (
    DEFAULT_HPC_CHUNK_COUNT,
    SchedulerComparisonPoint,
    build_scheduler_comparison_hpc_points,
    build_scheduler_comparison_point_id,
    requested_rate_sum_bps,
    total_point_demand_bits,
)

__all__ = [
    "DEFAULT_HPC_CHUNK_COUNT",
    "MANIFEST_COLUMNS",
    "SchedulerComparisonPoint",
    "build_manifest_row",
    "build_scheduler_comparison_hpc_points",
    "build_scheduler_comparison_point_id",
    "exact_scenario_key",
    "group_points_by_exact_scenario",
    "order_scheduler_comparison_points",
    "requested_rate_sum_bps",
    "scheduler_comparison_run_order_key",
    "select_chunk",
    "total_point_demand_bits",
]
