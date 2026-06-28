from __future__ import annotations

"""CLI for one finite-frame scheduler experiment run."""

import argparse

from configs.snapshot_run import DEFAULT_USER_GENERATION_CONFIG, LOG_LEVEL_CHOICES
from models import PASwitchPolicy, SchedulerMode
from user_generation import UserGenerationConfig

from .models import ExperimentRunConfig, ExperimentRunResult
from .result_recording import print_experiment_result
from .runner import run_experiment_case


DEFAULT_SMOKE_ACTIVE_USER_COUNT = 15
DEFAULT_SMOKE_LOAD_FACTOR = 0.4
DEFAULT_SMOKE_DISTANCE_M = 250.0


def run_from_cli(argv: list[str] | None = None) -> ExperimentRunResult:
    """Parse CLI inputs, run one scheduler experiment case, and print a summary."""

    args = parse_args(argv)
    config = build_experiment_run_config(
        scheduler_mode=SchedulerMode(args.scheduler_mode),
        switch_policy=PASwitchPolicy(args.switch_policy),
        active_user_count=int(args.active_user_count),
        load_factor=float(args.load_factor),
        distance_m=float(args.distance_m),
        reference_backlog_bits=int(args.reference_backlog_bits),
        frame_duration_s=float(args.frame_duration_s),
        cores=int(args.cores),
    )
    result = run_experiment_case(config)
    print_experiment_result(config, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the official single-case experiment CLI surface."""

    parser = argparse.ArgumentParser(
        description="Run one scheduler experiment case through user generation, candidate lookup, and scheduling."
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=[mode.value for mode in SchedulerMode],
        default=SchedulerMode.K_MILP.value,
        help="Scheduler backend to run.",
    )
    parser.add_argument(
        "--switch-policy",
        choices=[policy.value for policy in PASwitchPolicy],
        default=PASwitchPolicy.DUAL_SWITCHABLE.value,
        help="PA switching scenario passed to the scheduler.",
    )
    parser.add_argument(
        "--active-user-count",
        type=int,
        default=DEFAULT_SMOKE_ACTIVE_USER_COUNT,
        help="Number of generated users.",
    )
    parser.add_argument(
        "--load-factor",
        type=float,
        default=DEFAULT_SMOKE_LOAD_FACTOR,
        help="Finite-frame demand multiplier.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=DEFAULT_SMOKE_DISTANCE_M,
        help="Distance assigned to every generated user in the default smoke case.",
    )
    parser.add_argument(
        "--reference-backlog-bits",
        type=int,
        default=DEFAULT_USER_GENERATION_CONFIG.reference_backlog_bits,
        help="Per-user reference backlog before load-factor scaling.",
    )
    parser.add_argument(
        "--frame-duration-s",
        type=float,
        default=DEFAULT_USER_GENERATION_CONFIG.frame_duration_s,
        help="Scheduling horizon used to convert generated backlog bits to required rate.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Core budget for candidate-table generation when the artifact is absent.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=None,
        help="Accepted for CLI compatibility; experiment runs print a compact summary.",
    )
    return parser.parse_args(argv)


def build_experiment_run_config(
    *,
    scheduler_mode: SchedulerMode = SchedulerMode.K_MILP,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
    active_user_count: int = DEFAULT_SMOKE_ACTIVE_USER_COUNT,
    load_factor: float = DEFAULT_SMOKE_LOAD_FACTOR,
    distance_m: float = DEFAULT_SMOKE_DISTANCE_M,
    reference_backlog_bits: int = DEFAULT_USER_GENERATION_CONFIG.reference_backlog_bits,
    frame_duration_s: float = DEFAULT_USER_GENERATION_CONFIG.frame_duration_s,
    cores: int = 1,
) -> ExperimentRunConfig:
    """Build the single-case experiment config used by the default main smoke."""

    user_generation_config = UserGenerationConfig(
        active_user_count=int(active_user_count),
        load_factor=float(load_factor),
        distance_min_m=float(distance_m),
        distance_max_m=float(distance_m),
        reference_backlog_bits=int(reference_backlog_bits),
        frame_duration_s=float(frame_duration_s),
        distance_layout="all_edge",
    )
    return ExperimentRunConfig(
        user_generation_config=user_generation_config,
        scheduler_mode=_resolve_scheduler_mode(scheduler_mode),
        switch_policy=_resolve_switch_policy(switch_policy),
        cores=int(cores),
    )


def _resolve_scheduler_mode(scheduler_mode: SchedulerMode | str) -> SchedulerMode:
    if isinstance(scheduler_mode, SchedulerMode):
        return scheduler_mode
    return SchedulerMode(str(scheduler_mode))


def _resolve_switch_policy(switch_policy: PASwitchPolicy | str) -> PASwitchPolicy:
    if isinstance(switch_policy, PASwitchPolicy):
        return switch_policy
    return PASwitchPolicy(str(switch_policy))


__all__ = [
    "build_experiment_run_config",
    "parse_args",
    "run_from_cli",
]
