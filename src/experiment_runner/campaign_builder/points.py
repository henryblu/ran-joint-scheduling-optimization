from __future__ import annotations

"""Canonical point grid for cleaned finite-frame experiment campaigns."""

from dataclasses import dataclass
from collections.abc import Iterable
from itertools import product

from models import PASwitchPolicy, SchedulerMode
from user_generation.models import TRUNCATED_NORMAL_DISTANCE_MODEL


DEFAULT_DISTANCE_MIN_M = 25.0
DEFAULT_DISTANCE_MAX_M = 500.0
DEFAULT_REFERENCE_BACKLOG_BITS = 100_000
DEFAULT_FRAME_DURATION_S = 0.010
DEFAULT_SIGMA_DISTANCE_M = 100.0
DEFAULT_CAMPAIGN_CHUNK_COUNT = 32

DEFAULT_SCHEDULER_MODES = (
    SchedulerMode.ROUND_ROBIN.value,
    SchedulerMode.K_MILP.value,
)
DEFAULT_ACTIVE_USER_COUNTS = tuple(range(4, 34, 2))
DEFAULT_LOAD_FACTORS = tuple(round(0.2 * index, 1) for index in range(1, 16))
DEFAULT_MEAN_DISTANCE_VALUES_M = (
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
DEFAULT_SWITCH_POLICIES = (
    PASwitchPolicy.BASELINE_8W_ONLY.value,
    PASwitchPolicy.HARD_OFF.value,
    PASwitchPolicy.DUAL_SWITCHABLE.value,
)


@dataclass(frozen=True)
class CampaignPoint:
    """One finite-frame scenario point in an experiment campaign."""

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


def build_default_campaign_points() -> tuple[CampaignPoint, ...]:
    """Return the default cleaned campaign grid for finite-frame scheduler runs."""

    return build_campaign_points(
        scheduler_modes=DEFAULT_SCHEDULER_MODES,
        active_user_counts=DEFAULT_ACTIVE_USER_COUNTS,
        load_factors=DEFAULT_LOAD_FACTORS,
        mean_distance_values_m=DEFAULT_MEAN_DISTANCE_VALUES_M,
        switch_policies=DEFAULT_SWITCH_POLICIES,
    )


def build_campaign_points(
    *,
    scheduler_modes: Iterable[str],
    active_user_counts: Iterable[int],
    load_factors: Iterable[float],
    mean_distance_values_m: Iterable[float],
    switch_policies: Iterable[str],
    sigma_distance_m: float = DEFAULT_SIGMA_DISTANCE_M,
) -> tuple[CampaignPoint, ...]:
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
            build_campaign_point(
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


def build_campaign_point(
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
) -> CampaignPoint:
    """Build one normalized campaign point and its stable point ID."""

    resolved_scheduler_mode = SchedulerMode(str(scheduler_mode)).value
    resolved_switch_policy = PASwitchPolicy(str(switch_policy)).value
    return CampaignPoint(
        point_id=build_campaign_point_id(
            scheduler_mode=resolved_scheduler_mode,
            active_user_count=int(active_user_count),
            load_factor=float(load_factor),
            mean_distance_m=float(mean_distance_m),
            sigma_distance_m=float(sigma_distance_m),
            switch_policy=resolved_switch_policy,
        ),
        scheduler_mode=resolved_scheduler_mode,
        active_user_count=int(active_user_count),
        load_factor=float(load_factor),
        distance_min_m=float(distance_min_m),
        distance_max_m=float(distance_max_m),
        distance_model=str(distance_model),
        mean_distance_m=float(mean_distance_m),
        sigma_distance_m=float(sigma_distance_m),
        reference_backlog_bits=int(reference_backlog_bits),
        frame_duration_s=float(frame_duration_s),
        switch_policy=resolved_switch_policy,
    )


def build_campaign_point_id(
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


def total_point_demand_bits(point: CampaignPoint) -> int:
    return int(
        round(
            float(point.load_factor)
            * float(point.active_user_count)
            * float(point.reference_backlog_bits)
        )
    )


def requested_rate_sum_bps(point: CampaignPoint) -> float:
    return float(total_point_demand_bits(point)) / float(point.frame_duration_s)


__all__ = [
    "CampaignPoint",
    "DEFAULT_CAMPAIGN_CHUNK_COUNT",
    "build_campaign_point",
    "build_campaign_point_id",
    "build_campaign_points",
    "build_default_campaign_points",
    "requested_rate_sum_bps",
    "total_point_demand_bits",
]
