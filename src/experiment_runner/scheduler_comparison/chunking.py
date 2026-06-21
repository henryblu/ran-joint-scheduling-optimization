from __future__ import annotations

"""Stable ordering and exact-scenario chunking for scheduler-comparison campaigns."""

from collections.abc import Iterable

from .points import SchedulerComparisonPoint


def order_scheduler_comparison_points(
    points: Iterable[SchedulerComparisonPoint],
) -> tuple[SchedulerComparisonPoint, ...]:
    """Return points ordered so exact-scenario load chains stay contiguous."""

    return tuple(sorted(tuple(points), key=scheduler_comparison_run_order_key))


def select_chunk(
    points: Iterable[SchedulerComparisonPoint],
    *,
    chunk_index: int,
    chunk_count: int,
) -> tuple[SchedulerComparisonPoint, ...]:
    """Select one deterministic chunk without splitting exact-scenario load chains."""

    if int(chunk_count) <= 0:
        raise ValueError("chunk_count must be positive.")
    if int(chunk_index) < 0 or int(chunk_index) >= int(chunk_count):
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < chunk_count.")

    selected = []
    grouped_points = group_points_by_exact_scenario(order_scheduler_comparison_points(points))
    for group_index, group in enumerate(grouped_points):
        if int(group_index) % int(chunk_count) != int(chunk_index):
            continue
        selected.extend(group)
    return tuple(selected)


def group_points_by_exact_scenario(
    points: Iterable[SchedulerComparisonPoint],
) -> tuple[tuple[SchedulerComparisonPoint, ...], ...]:
    """Group already-ordered points by scheduler, PA policy, users, and distance population."""

    groups: list[list[SchedulerComparisonPoint]] = []
    current_key: tuple[object, ...] | None = None
    for point in points:
        point_key = exact_scenario_key(point)
        if point_key != current_key:
            groups.append([])
            current_key = point_key
        groups[-1].append(point)
    return tuple(tuple(group) for group in groups)


def scheduler_comparison_run_order_key(point: SchedulerComparisonPoint) -> tuple[object, ...]:
    return (
        str(point.scheduler_mode),
        str(point.switch_policy),
        str(point.distance_model),
        float(point.mean_distance_m),
        float(point.sigma_distance_m),
        int(point.active_user_count),
        float(point.load_factor),
    )



def exact_scenario_key(point: SchedulerComparisonPoint) -> tuple[object, ...]:
    return scheduler_comparison_run_order_key(point)[:-1]

