from __future__ import annotations

import math

from configs.user import USER_REQUIREMENT_COLUMNS
from user_generation import (
    TRUNCATED_NORMAL_DISTANCE_MODEL,
    UserGenerationConfig,
    build_distance_population_summary,
    build_scheduler_user_table,
    build_user_demand_table,
    build_user_generation_snapshot,
)


def test_user_generation_builds_scheduler_table_contract():
    config = UserGenerationConfig(
        active_user_count=3,
        load_factor=1.5,
        distance_min_m=25.0,
        distance_max_m=500.0,
        reference_backlog_bits=100_000,
        frame_duration_s=0.010,
        distance_layout="area_uniform",
    )

    scheduler_table = build_scheduler_user_table(config)
    demand_table = build_user_demand_table(config)
    snapshot = build_user_generation_snapshot(config)

    assert list(scheduler_table.columns) == USER_REQUIREMENT_COLUMNS
    assert list(demand_table.columns) == ["user_id", "distance_m", "demand_bits", "required_rate_bps"]
    assert tuple(scheduler_table["user_id"]) == (1, 2, 3)
    assert tuple(demand.demand_bits for demand in snapshot) == (150_000, 150_000, 150_000)
    assert math.isclose(float(scheduler_table["required_rate_bps"].sum()), 45_000_000.0)


def test_truncated_normal_user_distances_are_deterministic_midpoint_quantiles():
    config = UserGenerationConfig(
        active_user_count=4,
        load_factor=1.0,
        distance_min_m=25.0,
        distance_max_m=500.0,
        reference_backlog_bits=100_000,
        frame_duration_s=0.010,
        distance_model=TRUNCATED_NORMAL_DISTANCE_MODEL,
        mean_distance_m=250.0,
        sigma_distance_m=100.0,
    )

    table = build_scheduler_user_table(config)
    summary = build_distance_population_summary(config)

    distances = tuple(float(distance) for distance in table["distance_m"])
    assert distances == tuple(sorted(distances))
    assert min(distances) >= config.distance_min_m
    assert max(distances) <= config.distance_max_m
    assert summary["user_distance_m_list"] == list(distances)
    assert summary["distance_model"] == TRUNCATED_NORMAL_DISTANCE_MODEL
