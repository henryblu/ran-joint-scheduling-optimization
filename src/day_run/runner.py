from __future__ import annotations

"""Run the full synthetic day workflow from demand generation to export.

The runner owns only top-level orchestration. Lower layers still own:
- generating valid scheduler-ready demand rows,
- validating and preparing single-user spaces,
- preparing and solving the joint TDMA problem.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from time import perf_counter

import pandas as pd

from configs import USER_REQUIREMENT_COLUMNS
from configs.day_run import (
    ACTIVE_BIN_SCOPE_ENV_VAR,
    PROGRESS_LOG_INTERVAL,
    THREAD_LIMIT_ENV_VARS,
    WORKER_LOG_LEVEL_ENV_VAR,
)
from day_cycle_simulation.generation import build_scheduler_day_user_table
from models import PASwitchPolicy
from models.day_run import BinRunResult, DayRunConfig
from multi_user_tdma_scheduler.api import run_multi_user_tdma_scheduler
from single_user_lookup.api import build_batch_user_parameter_space
from run_reporting import (
    log_bin_result,
    log_run_progress,
    log_run_setup,
    log_run_summary,
    configure_run_logging,
)

from .export import write_day_run_result


def run_day(config: DayRunConfig) -> list[BinRunResult]:
    """Run the full day workflow from daily demand generation to JSON export.

    Steps:
    1. Build the day-wide scheduler-facing user table once.
    2. Project that table into one user table per simulation bin.
    3. Solve every bin and report sparse progress to the console.
    4. Write the final authoritative export and summary line.
    """

    scheduler_day_user_table = build_scheduler_day_user_table(
        load_curve_csv=config.load_curve_csv,
        config=config.session_generation_config,
    )
    # Keep the runner-side projection simple: the day-cycle layer already owns
    # session generation, and the batch layer owns user-table validation.
    user_tables_by_bin = {
        bin_index: (
            scheduler_day_user_table.loc[
                scheduler_day_user_table["bin_index"] == int(bin_index),
                USER_REQUIREMENT_COLUMNS,
            ].reset_index(drop=True)
        )
        for bin_index in range(int(config.session_generation_config.day_bin_count))
    }

    run_started_at = perf_counter()
    log_run_setup(config)
    bin_results = run_day_bins(
        config=config,
        user_tables_by_bin=user_tables_by_bin,
        run_started_at=run_started_at,
    )
    write_day_run_result(
        config=config,
        user_tables_by_bin=user_tables_by_bin,
        bin_results=bin_results,
    )
    log_run_summary(
        bin_results=bin_results,
        elapsed_s=float(perf_counter() - run_started_at),
    )
    return bin_results


def run_day_bins(
    *,
    config: DayRunConfig,
    user_tables_by_bin: dict[int, pd.DataFrame],
    run_started_at: float | None = None,
) -> list[BinRunResult]:
    """Run the day's independent bins and emit sparse run-level progress updates.

    Use one pool-based execution path for both small and large runs so the
    orchestration code stays linear and easy to follow.
    """

    total_bins = int(len(user_tables_by_bin))
    bin_workers = min(int(config.cores), total_bins)
    # Prevent BLAS/OpenMP libraries from multiplying threads inside each worker.
    threads_per_worker = max(1, int(config.cores) // max(1, int(bin_workers)))
    for env_var in THREAD_LIMIT_ENV_VARS:
        os.environ[env_var] = str(threads_per_worker)

    started_at = perf_counter() if run_started_at is None else float(run_started_at)

    # Keep progress accounting local to this function; it is only used to emit
    # the sparse parent-owned run summary lines.
    completed_bins = 0
    solved_bins = 0
    infeasible_bins = 0
    empty_bins = 0
    with ProcessPoolExecutor(max_workers=bin_workers) as executor:
        futures = [
            executor.submit(
                run_bin,
                bin_index,
                user_table,
                config.window_n_frames,
                config.switch_policy,
            )
            for bin_index, user_table in user_tables_by_bin.items()
        ]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_bins += 1
            solved_bins += int(result.status == "solved")
            infeasible_bins += int(result.status == "infeasible")
            empty_bins += int(result.status == "empty")
            log_bin_result(result)
            if completed_bins == total_bins or completed_bins % PROGRESS_LOG_INTERVAL == 0:
                log_run_progress(
                    total_bins=total_bins,
                    completed_bins=completed_bins,
                    solved_bins=solved_bins,
                    infeasible_bins=infeasible_bins,
                    empty_bins=empty_bins,
                    elapsed_s=float(perf_counter() - started_at),
                )

    return sorted(results, key=lambda result: int(result.bin_index))


def run_bin(
    bin_index: int,
    user_table: pd.DataFrame,
    window_n_frames: int | None,
    switch_policy: PASwitchPolicy,
) -> BinRunResult:
    """Run one bin: build the user spaces, solve the joint scheduler, and return the lean result.

    Steps:
    1. Reuse the scheduler-facing user table built earlier in the day run.
    2. Let the single-user batch layer own all user-table validation and space building.
    3. Run the joint TDMA scheduler on that trusted batch artifact.
    4. Collapse the result to the small fields the day-run layer actually exports.
    """

    with _worker_logging_scope(bin_index):
        user_count = int(len(user_table))
        total_started_at = perf_counter()
        if user_table.empty:
            return BinRunResult(
                bin_index=int(bin_index),
                status="empty",
                user_count=0,
                total_elapsed_s=float(perf_counter() - total_started_at),
            )

        # The batch layer owns all table normalization and candidate-space
        # preparation. The runner just measures and passes the artifact onward.
        single_user_started_at = perf_counter()
        batch_space = build_batch_user_parameter_space(user_table)
        single_user_elapsed_s = float(perf_counter() - single_user_started_at)
        joint_started_at = perf_counter()
        try:
            scheduler_result = run_multi_user_tdma_scheduler(
                batch_space,
                window_n_frames=window_n_frames,
                switch_policy=switch_policy,
            )
        except RuntimeError:
            return BinRunResult(
                bin_index=int(bin_index),
                status="infeasible",
                user_count=user_count,
                single_user_elapsed_s=single_user_elapsed_s,
                joint_elapsed_s=float(perf_counter() - joint_started_at),
                total_elapsed_s=float(perf_counter() - total_started_at),
            )

        return BinRunResult(
            bin_index=int(bin_index),
            status="solved",
            user_count=user_count,
            single_user_elapsed_s=single_user_elapsed_s,
            joint_elapsed_s=float(perf_counter() - joint_started_at),
            total_elapsed_s=float(perf_counter() - total_started_at),
            best_schedule=scheduler_result.best_schedule,
        )


@contextmanager
def _worker_logging_scope(bin_index: int):
    """Attach the parent log level and active bin label to one worker execution."""

    if not logging.getLogger().handlers:
        inherited_level = os.environ.get(WORKER_LOG_LEVEL_ENV_VAR)
        if inherited_level is not None:
            # Rebuild the same formatter in spawned workers without duplicating
            # the logging policy in the scheduler or batch layers.
            configure_run_logging(inherited_level)

    previous_scope = os.environ.get(ACTIVE_BIN_SCOPE_ENV_VAR)
    os.environ[ACTIVE_BIN_SCOPE_ENV_VAR] = f"B{int(bin_index):03d}"
    try:
        yield
    finally:
        if previous_scope is None:
            os.environ.pop(ACTIVE_BIN_SCOPE_ENV_VAR, None)
        else:
            os.environ[ACTIVE_BIN_SCOPE_ENV_VAR] = previous_scope


__all__ = [
    "run_bin",
    "run_day",
    "run_day_bins",
]
