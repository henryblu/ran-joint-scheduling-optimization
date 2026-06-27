from __future__ import annotations

"""Build deterministic finite-frame user populations for scheduler runs."""

import math

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from configs.user import USER_REQUIREMENT_COLUMNS

from .models import (
    GeneratedUserDemand,
    LEGACY_DISTANCE_MODEL,
    TRUNCATED_NORMAL_DISTANCE_MODEL,
    UserGenerationConfig,
)


DISTANCE_LAYOUTS = ("area_uniform", "edge_heavy", "all_edge")
DISTANCE_MODELS = (LEGACY_DISTANCE_MODEL, TRUNCATED_NORMAL_DISTANCE_MODEL)


def build_user_generation_snapshot(config: UserGenerationConfig) -> tuple[GeneratedUserDemand, ...]:
    """Build one deterministic finite-frame user generation snapshot.

    Steps:
    1. Resolve deterministic user distances for the configured population model.
    2. Split the aggregate finite-frame demand across active users.
    3. Return stable generated demand records for the scheduler handoff.
    """

    distances_m = _build_distances_m(config)
    demand_bits_by_user = _split_total_demand_bits(config)
    return tuple(
        GeneratedUserDemand(
            user_id=user_id,
            distance_m=float(distances_m[user_id - 1]),
            demand_bits=int(demand_bits_by_user[user_id - 1]),
        )
        for user_id in range(1, int(config.active_user_count) + 1)
    )


def build_scheduler_user_table(config: UserGenerationConfig) -> pd.DataFrame:
    """Return the lean scheduler-facing user table for one generated frame."""

    demands = build_user_generation_snapshot(config)
    rows = [
        {
            "user_id": int(demand.user_id),
            "distance_m": float(demand.distance_m),
            "required_rate_bps": float(demand.demand_bits) / float(config.frame_duration_s),
        }
        for demand in demands
    ]
    return pd.DataFrame(rows, columns=USER_REQUIREMENT_COLUMNS)


def build_user_demand_table(config: UserGenerationConfig) -> pd.DataFrame:
    """Return the report-facing generated-user table including demand bits."""

    demands = build_user_generation_snapshot(config)
    rows = [
        {
            "user_id": int(demand.user_id),
            "distance_m": float(demand.distance_m),
            "demand_bits": int(demand.demand_bits),
            "required_rate_bps": float(demand.demand_bits) / float(config.frame_duration_s),
        }
        for demand in demands
    ]
    return pd.DataFrame(rows, columns=["user_id", "distance_m", "demand_bits", "required_rate_bps"])


def build_truncated_normal_distances_m(config: UserGenerationConfig) -> tuple[float, ...]:
    """Return midpoint quantiles from the configured truncated normal population."""

    user_count = int(config.active_user_count)
    if user_count <= 0:
        raise ValueError("active_user_count must be positive.")

    distance_min_m = float(config.distance_min_m)
    distance_max_m = float(config.distance_max_m)
    mean_distance_m = float(config.mean_distance_m)
    sigma_distance_m = float(config.sigma_distance_m)
    if distance_min_m >= distance_max_m:
        raise ValueError("distance_min_m must be below distance_max_m.")
    if sigma_distance_m <= 0.0:
        raise ValueError("sigma_distance_m must be positive.")

    lower = (distance_min_m - mean_distance_m) / sigma_distance_m
    upper = (distance_max_m - mean_distance_m) / sigma_distance_m
    quantiles = (np.arange(1, user_count + 1, dtype=float) - 0.5) / float(user_count)
    distances = truncnorm.ppf(
        quantiles,
        lower,
        upper,
        loc=mean_distance_m,
        scale=sigma_distance_m,
    )
    return tuple(float(distance) for distance in distances)


def build_distance_population_summary(config: UserGenerationConfig) -> dict[str, object]:
    """Return one inspectable summary row for the generated distance population."""

    distances = _build_distances_m(config)
    return {
        "distance_model": str(config.distance_model),
        "distance_min_m": float(config.distance_min_m),
        "distance_max_m": float(config.distance_max_m),
        "mean_distance_m": float(config.mean_distance_m),
        "sigma_distance_m": float(config.sigma_distance_m),
        "active_user_count": int(config.active_user_count),
        "min_generated_distance_m": float(min(distances)),
        "max_generated_distance_m": float(max(distances)),
        "empirical_mean_distance_m": float(np.mean(distances)),
        "empirical_median_distance_m": float(np.median(distances)),
        "empirical_std_distance_m": float(np.std(distances, ddof=0)),
        "user_distance_m_list": [float(distance) for distance in distances],
        "user_distance_generation_rule": str(config.user_distance_generation_rule),
    }


def _build_distances_m(config: UserGenerationConfig) -> tuple[float, ...]:
    distance_model = str(config.distance_model)
    if distance_model == TRUNCATED_NORMAL_DISTANCE_MODEL:
        return build_truncated_normal_distances_m(config)
    if distance_model != LEGACY_DISTANCE_MODEL:
        raise ValueError(f"Unsupported distance model: {distance_model}")

    layout = str(config.distance_layout)
    if layout == "area_uniform":
        return _area_uniform_distances_m(config)
    if layout == "edge_heavy":
        return _edge_heavy_distances_m(config)
    if layout == "all_edge":
        return tuple(float(config.distance_max_m) for _ in range(int(config.active_user_count)))

    raise ValueError(f"Unsupported distance layout: {layout}")


def _area_uniform_distances_m(config: UserGenerationConfig) -> tuple[float, ...]:
    user_count = _positive_user_count(config)
    d_min_sq = float(config.distance_min_m) ** 2
    d_max_sq = float(config.distance_max_m) ** 2
    return tuple(
        math.sqrt(d_min_sq + ((user_id - 0.5) / float(user_count)) * (d_max_sq - d_min_sq))
        for user_id in range(1, user_count + 1)
    )


def _edge_heavy_distances_m(config: UserGenerationConfig) -> tuple[float, ...]:
    user_count = _positive_user_count(config)
    d_min_sq = float(config.distance_min_m) ** 2
    d_max_sq = float(config.distance_max_m) ** 2
    return tuple(
        math.sqrt(d_min_sq + (((user_id - 0.5) / float(user_count)) ** 0.5) * (d_max_sq - d_min_sq))
        for user_id in range(1, user_count + 1)
    )


def _split_total_demand_bits(config: UserGenerationConfig) -> tuple[int, ...]:
    user_count = _positive_user_count(config)
    total_demand_bits = int(
        round(float(config.load_factor) * float(user_count) * float(config.reference_backlog_bits))
    )
    base_demand_bits = total_demand_bits // user_count
    remainder_bits = total_demand_bits % user_count
    return tuple(
        int(base_demand_bits + (1 if user_index < remainder_bits else 0))
        for user_index in range(user_count)
    )


def _positive_user_count(config: UserGenerationConfig) -> int:
    user_count = int(config.active_user_count)
    if user_count <= 0:
        raise ValueError("active_user_count must be positive.")
    return user_count


build_finite_buffer_demand_snapshot = build_user_generation_snapshot
build_scheduler_snapshot_table = build_scheduler_user_table
build_snapshot_demand_table = build_user_demand_table


__all__ = [
    "DISTANCE_LAYOUTS",
    "DISTANCE_MODELS",
    "build_distance_population_summary",
    "build_finite_buffer_demand_snapshot",
    "build_scheduler_snapshot_table",
    "build_scheduler_user_table",
    "build_snapshot_demand_table",
    "build_truncated_normal_distances_m",
    "build_user_demand_table",
    "build_user_generation_snapshot",
]
