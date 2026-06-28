from __future__ import annotations

"""Map campaign points onto the official single-case experiment config."""

from models import PASwitchPolicy, SchedulerMode
from user_generation import UserGenerationConfig

from experiment_runner.models import ExperimentRunConfig

from .points import CampaignPoint


def build_experiment_run_config_for_point(
    point: CampaignPoint,
    *,
    cores: int = 1,
) -> ExperimentRunConfig:
    """Return the official experiment-runner config for one campaign point."""

    user_generation_config = UserGenerationConfig(
        active_user_count=int(point.active_user_count),
        load_factor=float(point.load_factor),
        distance_min_m=float(point.distance_min_m),
        distance_max_m=float(point.distance_max_m),
        reference_backlog_bits=int(point.reference_backlog_bits),
        frame_duration_s=float(point.frame_duration_s),
        distance_model=str(point.distance_model),
        mean_distance_m=float(point.mean_distance_m),
        sigma_distance_m=float(point.sigma_distance_m),
    )
    return ExperimentRunConfig(
        user_generation_config=user_generation_config,
        scheduler_mode=SchedulerMode(str(point.scheduler_mode)),
        switch_policy=PASwitchPolicy(str(point.switch_policy)),
        cores=int(cores),
    )


__all__ = ["build_experiment_run_config_for_point"]
