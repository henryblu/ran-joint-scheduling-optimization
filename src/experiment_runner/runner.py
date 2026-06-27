from __future__ import annotations

"""Run one finite-frame scheduler experiment from generated users to schedule result."""

from time import perf_counter

from candidate_table import build_batch_user_parameter_space, load_or_build_candidate_table
from schedulers import run_scheduler
from user_generation import build_scheduler_user_table

from .models import ExperimentRunConfig, ExperimentRunResult


def run_experiment_case(config: ExperimentRunConfig) -> ExperimentRunResult:
    """Run the official single-case experiment workflow.

    Steps:
    1. Load or rebuild the stored candidate table artifact.
    2. Generate one scheduler-facing finite-frame user table.
    3. Convert generated users into per-user candidate spaces.
    4. Run the selected scheduler backend and return a compact result.
    """

    total_started_at = perf_counter()

    candidate_table_started_at = perf_counter()
    load_or_build_candidate_table(max_workers=int(config.cores))
    candidate_table_elapsed_s = float(perf_counter() - candidate_table_started_at)

    user_generation_started_at = perf_counter()
    scheduler_user_table = build_scheduler_user_table(config.user_generation_config)
    user_generation_elapsed_s = float(perf_counter() - user_generation_started_at)

    candidate_lookup_started_at = perf_counter()
    batch_space = build_batch_user_parameter_space(scheduler_user_table)
    candidate_lookup_elapsed_s = float(perf_counter() - candidate_lookup_started_at)

    scheduler_started_at = perf_counter()
    schedule_result = run_scheduler(
        batch_space,
        scheduler_mode=config.scheduler_mode,
        switch_policy=config.switch_policy,
    )
    scheduler_elapsed_s = float(perf_counter() - scheduler_started_at)

    return ExperimentRunResult(
        status="solved" if schedule_result.feasible else "infeasible",
        scheduler_user_table=scheduler_user_table,
        schedule_result=schedule_result,
        candidate_table_elapsed_s=candidate_table_elapsed_s,
        user_generation_elapsed_s=user_generation_elapsed_s,
        candidate_lookup_elapsed_s=candidate_lookup_elapsed_s,
        scheduler_elapsed_s=scheduler_elapsed_s,
        total_elapsed_s=float(perf_counter() - total_started_at),
    )


__all__ = ["run_experiment_case"]
