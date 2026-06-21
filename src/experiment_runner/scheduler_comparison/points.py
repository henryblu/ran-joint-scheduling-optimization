from __future__ import annotations

"""Canonical point grid for the scheduler-comparison thesis campaign."""

from dataclasses import dataclass
from collections.abc import Iterable
from itertools import product


TDMA_SCHEDULER = "tdma"
OFDMA_ROUND_ROBIN_SCHEDULER = "ofdma_round_robin"
OFDMA_MILP_SINGLE_SNAPSHOT_SCHEDULER = "ofdma_milp_single_snapshot"

BASELINE_8W_ONLY_POLICY = "baseline_8w_only"
HARD_OFF_POLICY = "hard_off"
DUAL_SWITCHABLE_POLICY = "dual_switchable"

TRUNCATED_NORMAL_DISTANCE_MODEL = "truncated_normal_mean_sweep"

DEFAULT_DISTANCE_MIN_M = 25.0
DEFAULT_DISTANCE_MAX_M = 500.0
DEFAULT_REFERENCE_BACKLOG_BITS = 100_000
DEFAULT_FRAME_DURATION_S = 0.010
DEFAULT_SIGMA_DISTANCE_M = 100.0
DEFAULT_HPC_CHUNK_COUNT = 32

HPC_SCHEDULER_MODES = (
    TDMA_SCHEDULER,
    OFDMA_ROUND_ROBIN_SCHEDULER,
    OFDMA_MILP_SINGLE_SNAPSHOT_SCHEDULER,
)
HPC_ACTIVE_USER_COUNTS = tuple(range(4, 34, 2))
HPC_LOAD_FACTORS = tuple(round(0.2 * index, 1) for index in range(1, 16))
HPC_MEAN_DISTANCE_VALUES_M = (
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    350.0,
    400.0,
    450.0,
    500.0,
)
HPC_SWITCH_POLICIES = (
    BASELINE_8W_ONLY_POLICY,
    HARD_OFF_POLICY,
    DUAL_SWITCHABLE_POLICY,
)


@dataclass(frozen=True)
class SchedulerComparisonPoint:
    """One finite-buffer scenario point in the scheduler-comparison campaign."""

    point_id: str
    scheduler_mode: str
    active_user_count: int
    load_factor: float
    distance_min_m: float
    distance_max_m: float
    distance_model: str
    mean_distance_m: float
    sigma_distance_m: float
    reference_backlog_bits: int
    frame_duration_s: float
    switch_policy: str


def build_scheduler_comparison_hpc_points() -> tuple[SchedulerComparisonPoint, ...]:
    """Return the historical full scheduler-comparison grid used by the canonical ZIP."""

    return build_scheduler_comparison_points(
        scheduler_modes=HPC_SCHEDULER_MODES,
        active_user_counts=HPC_ACTIVE_USER_COUNTS,
        load_factors=HPC_LOAD_FACTORS,
        mean_distance_values_m=HPC_MEAN_DISTANCE_VALUES_M,
        switch_policies=HPC_SWITCH_POLICIES,
    )


def build_scheduler_comparison_points(
    *,
    scheduler_modes: Iterable[str],
    active_user_counts: Iterable[int],
    load_factors: Iterable[float],
    mean_distance_values_m: Iterable[float],
    switch_policies: Iterable[str],
    sigma_distance_m: float = DEFAULT_SIGMA_DISTANCE_M,
) -> tuple[SchedulerComparisonPoint, ...]:
    """Build a matched Cartesian grid over scheduler and distance-population axes."""

    points = []
    for scheduler_mode, active_user_count, load_factor, mean_distance_m, switch_policy in product(
        scheduler_modes,
        active_user_counts,
        load_factors,
        mean_distance_values_m,
        switch_policies,
    ):
        points.append(
            build_scheduler_comparison_point(
                scheduler_mode=str(scheduler_mode),
                active_user_count=int(active_user_count),
                load_factor=float(load_factor),
                distance_min_m=DEFAULT_DISTANCE_MIN_M,
                distance_max_m=DEFAULT_DISTANCE_MAX_M,
                distance_model=TRUNCATED_NORMAL_DISTANCE_MODEL,
                mean_distance_m=float(mean_distance_m),
                sigma_distance_m=float(sigma_distance_m),
                reference_backlog_bits=DEFAULT_REFERENCE_BACKLOG_BITS,
                frame_duration_s=DEFAULT_FRAME_DURATION_S,
                switch_policy=str(switch_policy),
            )
        )
    return tuple(points)


def build_scheduler_comparison_point(
    *,
    scheduler_mode: str,
    active_user_count: int,
    load_factor: float,
    distance_min_m: float,
    distance_max_m: float,
    distance_model: str,
    mean_distance_m: float,
    sigma_distance_m: float,
    reference_backlog_bits: int,
    frame_duration_s: float,
    switch_policy: str,
) -> SchedulerComparisonPoint:
    """Build one normalized scheduler-comparison point and its stable point ID."""

    return SchedulerComparisonPoint(
        point_id=build_scheduler_comparison_point_id(
            scheduler_mode=scheduler_mode,
            active_user_count=int(active_user_count),
            load_factor=float(load_factor),
            mean_distance_m=float(mean_distance_m),
            sigma_distance_m=float(sigma_distance_m),
            switch_policy=switch_policy,
        ),
        scheduler_mode=str(scheduler_mode),
        active_user_count=int(active_user_count),
        load_factor=float(load_factor),
        distance_min_m=float(distance_min_m),
        distance_max_m=float(distance_max_m),
        distance_model=str(distance_model),
        mean_distance_m=float(mean_distance_m),
        sigma_distance_m=float(sigma_distance_m),
        reference_backlog_bits=int(reference_backlog_bits),
        frame_duration_s=float(frame_duration_s),
        switch_policy=str(switch_policy),
    )


def build_scheduler_comparison_point_id(
    *,
    scheduler_mode: str,
    active_user_count: int,
    load_factor: float,
    mean_distance_m: float,
    sigma_distance_m: float,
    switch_policy: str,
) -> str:
    """Return the filesystem-safe point identifier used inside chunk outputs."""

    load_token = str(float(load_factor)).replace(".", "p")
    return (
        f"{scheduler_mode}_u{int(active_user_count):02d}"
        f"_load{load_token}"
        f"_mean{int(mean_distance_m):03d}"
        f"_sigma{int(sigma_distance_m):03d}"
        f"_{switch_policy}"
    )


def total_point_demand_bits(point: SchedulerComparisonPoint) -> int:
    return int(
        round(
            float(point.load_factor)
            * float(point.active_user_count)
            * float(point.reference_backlog_bits)
        )
    )


def requested_rate_sum_bps(point: SchedulerComparisonPoint) -> float:
    return float(total_point_demand_bits(point)) / float(point.frame_duration_s)


__all__ = [
    "DEFAULT_HPC_CHUNK_COUNT",
    "SchedulerComparisonPoint",
    "build_scheduler_comparison_hpc_points",
    "build_scheduler_comparison_point",
    "build_scheduler_comparison_point_id",
    "build_scheduler_comparison_points",
    "requested_rate_sum_bps",
    "total_point_demand_bits",
]
