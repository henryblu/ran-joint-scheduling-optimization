"""Finite-frame user generation for scheduler comparison runs."""

from .finite_frame import (
    DISTANCE_LAYOUTS,
    DISTANCE_MODELS,
    build_distance_population_summary,
    build_scheduler_user_table,
    build_user_demand_table,
    build_user_generation_snapshot,
    build_truncated_normal_distances_m,
)
from .models import (
    DEFAULT_MEAN_DISTANCE_M,
    DEFAULT_SIGMA_DISTANCE_M,
    LEGACY_DISTANCE_MODEL,
    TRUNCATED_NORMAL_DISTANCE_MODEL,
    TRUNCATED_NORMAL_GENERATION_RULE,
    GeneratedUserDemand,
    UserGenerationConfig,
)


__all__ = [
    "DEFAULT_MEAN_DISTANCE_M",
    "DEFAULT_SIGMA_DISTANCE_M",
    "DISTANCE_LAYOUTS",
    "DISTANCE_MODELS",
    "GeneratedUserDemand",
    "LEGACY_DISTANCE_MODEL",
    "TRUNCATED_NORMAL_DISTANCE_MODEL",
    "TRUNCATED_NORMAL_GENERATION_RULE",
    "UserGenerationConfig",
    "build_distance_population_summary",
    "build_scheduler_user_table",
    "build_truncated_normal_distances_m",
    "build_user_demand_table",
    "build_user_generation_snapshot",
]
