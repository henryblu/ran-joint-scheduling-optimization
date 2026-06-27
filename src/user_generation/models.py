from __future__ import annotations

"""User-generation models for one finite scheduler frame."""

from dataclasses import dataclass


LEGACY_DISTANCE_MODEL = "legacy_max_layout"
TRUNCATED_NORMAL_DISTANCE_MODEL = "truncated_normal_mean_sweep"
TRUNCATED_NORMAL_GENERATION_RULE = "midpoint_quantiles_truncated_normal"
DEFAULT_MEAN_DISTANCE_M = 250.0
DEFAULT_SIGMA_DISTANCE_M = 100.0


@dataclass(frozen=True)
class UserGenerationConfig:
    """Resolved inputs for one finite-frame scheduler user population."""

    active_user_count: int
    load_factor: float
    distance_min_m: float
    distance_max_m: float
    reference_backlog_bits: int
    frame_duration_s: float
    distance_layout: str = "area_uniform"
    distance_model: str = LEGACY_DISTANCE_MODEL
    mean_distance_m: float = DEFAULT_MEAN_DISTANCE_M
    sigma_distance_m: float = DEFAULT_SIGMA_DISTANCE_M
    user_distance_generation_rule: str = TRUNCATED_NORMAL_GENERATION_RULE


@dataclass(frozen=True)
class GeneratedUserDemand:
    """One generated scheduler-facing user demand record."""

    user_id: int
    distance_m: float
    demand_bits: int


FiniteBufferDemandSnapshotConfig = UserGenerationConfig
FiniteBufferDemand = GeneratedUserDemand


__all__ = [
    "DEFAULT_MEAN_DISTANCE_M",
    "DEFAULT_SIGMA_DISTANCE_M",
    "FiniteBufferDemand",
    "FiniteBufferDemandSnapshotConfig",
    "GeneratedUserDemand",
    "LEGACY_DISTANCE_MODEL",
    "TRUNCATED_NORMAL_DISTANCE_MODEL",
    "TRUNCATED_NORMAL_GENERATION_RULE",
    "UserGenerationConfig",
]
