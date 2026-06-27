from __future__ import annotations

"""CLI for one finite-frame scheduler smoke run."""

import argparse

from configs.snapshot_run import DEFAULT_USER_GENERATION_CONFIG, LOG_LEVEL_CHOICES
from models import PASwitchPolicy, SchedulerMode
from user_generation import UserGenerationConfig

from .models import FiniteFrameRunConfig, FiniteFrameRunResult
from .runner import run_finite_frame


DEFAULT_SMOKE_ACTIVE_USER_COUNT = 15
DEFAULT_SMOKE_LOAD_FACTOR = 0.4
DEFAULT_SMOKE_DISTANCE_M = 250.0


def run_from_cli(argv: list[str] | None = None) -> FiniteFrameRunResult:
    """Parse CLI inputs, run one finite-frame scheduler case, and print a summary."""

    args = parse_args(argv)
    config = build_finite_frame_run_config(
        scheduler_mode=SchedulerMode(args.scheduler_mode),
        switch_policy=PASwitchPolicy(args.switch_policy),
        active_user_count=int(args.active_user_count),
        load_factor=float(args.load_factor),
        distance_m=float(args.distance_m),
        reference_backlog_bits=int(args.reference_backlog_bits),
        frame_duration_s=float(args.frame_duration_s),
        cores=int(args.cores),
    )
    result = run_finite_frame(config)
    print_finite_frame_result(config, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the finite-frame smoke CLI surface."""

    parser = argparse.ArgumentParser(
        description="Run one finite-frame scheduler case through user generation, candidate lookup, and scheduling."
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
        help="Accepted for CLI compatibility; finite-frame smoke prints a compact summary.",
    )
    return parser.parse_args(argv)


def build_finite_frame_run_config(
    *,
    scheduler_mode: SchedulerMode = SchedulerMode.K_MILP,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
    active_user_count: int = DEFAULT_SMOKE_ACTIVE_USER_COUNT,
    load_factor: float = DEFAULT_SMOKE_LOAD_FACTOR,
    distance_m: float = DEFAULT_SMOKE_DISTANCE_M,
    reference_backlog_bits: int = DEFAULT_USER_GENERATION_CONFIG.reference_backlog_bits,
    frame_duration_s: float = DEFAULT_USER_GENERATION_CONFIG.frame_duration_s,
    cores: int = 1,
) -> FiniteFrameRunConfig:
    """Build the finite-frame run config used by the default main smoke."""

    user_generation_config = UserGenerationConfig(
        active_user_count=int(active_user_count),
        load_factor=float(load_factor),
        distance_min_m=float(distance_m),
        distance_max_m=float(distance_m),
        reference_backlog_bits=int(reference_backlog_bits),
        frame_duration_s=float(frame_duration_s),
        distance_layout="all_edge",
    )
    return FiniteFrameRunConfig(
        user_generation_config=user_generation_config,
        scheduler_mode=scheduler_mode if isinstance(scheduler_mode, SchedulerMode) else SchedulerMode(str(scheduler_mode)),
        switch_policy=switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy)),
        cores=int(cores),
    )


def print_finite_frame_result(config: FiniteFrameRunConfig, result: FiniteFrameRunResult) -> None:
    """Print the compatibility smoke summary for a completed finite-frame run."""

    schedule_result = result.schedule_result
    solver_details = dict(schedule_result.solver_details)
    power_summary = schedule_result.power_summary
    active_slots = sum(1 for slot in schedule_result.slot_schedules if slot.active)
    allocation_count = sum(len(slot.allocations) for slot in schedule_result.slot_schedules)
    print(
        "FINITE_FRAME_RUN",
        f"status={result.status}",
        f"scheduler={schedule_result.scheduler_mode.value}",
        f"algorithm={solver_details.get('algorithm', 'unknown')}",
        f"policy={config.switch_policy.value}",
        f"users={config.user_generation_config.active_user_count}",
        f"load={config.user_generation_config.load_factor:g}",
        f"distance_m={config.user_generation_config.distance_max_m:g}",
    )
    print(
        "FINITE_FRAME_RESULT",
        f"feasible={schedule_result.feasible}",
        f"infeasible_reason={schedule_result.infeasible_reason}",
        f"active_slots={active_slots}",
        f"allocations={allocation_count}",
        f"avg_dc_w={power_summary.average_frame_dc_power_w:.9g}",
        f"frame_energy_j={power_summary.frame_energy_j:.9g}",
    )
    print(
        "FINITE_FRAME_TIMINGS",
        f"candidate_table_s={result.candidate_table_elapsed_s:.3f}",
        f"user_generation_s={result.user_generation_elapsed_s:.3f}",
        f"candidate_lookup_s={result.candidate_lookup_elapsed_s:.3f}",
        f"scheduler_s={result.scheduler_elapsed_s:.3f}",
        f"total_s={result.total_elapsed_s:.3f}",
    )


__all__ = [
    "build_finite_frame_run_config",
    "parse_args",
    "print_finite_frame_result",
    "run_from_cli",
]
