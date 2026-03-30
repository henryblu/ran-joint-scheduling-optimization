"""Thin orchestration entry point for one synthetic day TDMA simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from day_cycle_simulation.generation import build_scheduler_day_user_table
from day_cycle_simulation.models import (
    DEFAULT_SYNTHETIC_SESSION_GENERATION_CONFIG,
    SyntheticSessionGenerationConfig,
)
from configs import USER_REQUIREMENT_COLUMNS
from models import PASwitchPolicy
from multi_user_tdma_scheduler.api import run_multi_user_tdma_scheduler
from multi_user_tdma_scheduler.models import MultiUserTdmaSchedulerResult
from single_user_parameter_space.api import build_batch_user_parameter_space


@dataclass(frozen=True)
class ExperimentConfig:
    """High-level choices for one full-day simulation run."""

    load_curve_csv: Path
    session_generation_config: SyntheticSessionGenerationConfig
    window_n_frames: int | None
    switch_policy: PASwitchPolicy
    output_dir: Path


@dataclass(frozen=True)
class DaySimulationState:
    """Day-level demand artifacts prepared once and reused across all bins."""

    scheduler_day_user_table: pd.DataFrame
    simulation_bin_indices: tuple[int, ...]


@dataclass
class BinRunResult:
    """Per-bin orchestration summary kept lean after the scheduler handoff."""

    bin_index: int
    status: str
    user_count: int
    scheduler_result: MultiUserTdmaSchedulerResult | None = None
    error_message: str = ""


def main():
    """Run one full synthetic day from load curve to per-bin TDMA schedules."""

    experiment = _build_experiment()
    day_state = _build_day_state(experiment)

    bin_results = []
    for bin_index in day_state.simulation_bin_indices:
        bin_results.append(
            _run_bin(
                bin_index=int(bin_index),
                experiment=experiment,
                day_state=day_state,
            )
        )

    _finalize_run(
        experiment=experiment,
        day_state=day_state,
        bin_results=bin_results,
    )
    return bin_results


def _build_experiment() -> ExperimentConfig:
    """Choose one concrete full-day experiment using the current default inputs."""

    return ExperimentConfig(
        load_curve_csv=REPO_ROOT / "data" / "total_network_load_curve.csv",
        session_generation_config=DEFAULT_SYNTHETIC_SESSION_GENERATION_CONFIG,
        window_n_frames=None,
        switch_policy=PASwitchPolicy.STANDBY,
        output_dir=REPO_ROOT / "outputs" / "default_day_run",
    )


def _build_day_state(experiment: ExperimentConfig) -> DaySimulationState:
    """Prepare the lean day-wide scheduler request table once for the run."""

    scheduler_day_user_table = build_scheduler_day_user_table(
        load_curve_csv=experiment.load_curve_csv,
        config=experiment.session_generation_config,
    )
    return DaySimulationState(
        scheduler_day_user_table=scheduler_day_user_table,
        simulation_bin_indices=tuple(range(int(experiment.session_generation_config.day_bin_count))),
    )


def _run_bin(
    *,
    bin_index: int,
    experiment: ExperimentConfig,
    day_state: DaySimulationState,
) -> BinRunResult:
    """Run one 15-minute simulation bin from scheduler-ready users to joint schedule."""

    user_table = (
        day_state.scheduler_day_user_table.loc[
            day_state.scheduler_day_user_table["bin_index"] == int(bin_index),
            USER_REQUIREMENT_COLUMNS,
        ]
        .reset_index(drop=True)
    )
    if user_table.empty:
        return BinRunResult(
            bin_index=bin_index,
            status="empty",
            user_count=0,
        )

    batch_space = build_batch_user_parameter_space(user_table)

    try:
        scheduler_result = run_multi_user_tdma_scheduler(
            batch_space,
            window_n_frames=experiment.window_n_frames,
            switch_policy=experiment.switch_policy,
        )
    except RuntimeError as error:
        return BinRunResult(
            bin_index=bin_index,
            status="infeasible",
            user_count=int(len(user_table)),
            error_message=str(error),
        )

    return BinRunResult(
        bin_index=bin_index,
        status="solved",
        user_count=int(len(user_table)),
        scheduler_result=scheduler_result,
    )


def _finalize_run(
    *,
    experiment: ExperimentConfig,
    day_state: DaySimulationState,
    bin_results: list[BinRunResult],
) -> None:
    """Save the simplest useful raw artifacts for the completed run."""

    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    day_state.scheduler_day_user_table.to_csv(
        experiment.output_dir / "scheduler_day_user_table.csv",
        index=False,
    )
    _build_bin_result_table(bin_results).to_csv(
        experiment.output_dir / "bin_run_summary.csv",
        index=False,
    )


def _build_bin_result_table(bin_results: list[BinRunResult]) -> pd.DataFrame:
    """Build one compact status table for the per-bin orchestration results."""

    rows = []
    for result in bin_results:
        best_schedule = None if result.scheduler_result is None else result.scheduler_result.best_schedule
        rows.append(
            {
                "bin_index": int(result.bin_index),
                "status": result.status,
                "user_count": int(result.user_count),
                "scheduled_user_count": 0 if best_schedule is None else int(len(best_schedule["rows"])),
                "slot_total": None if best_schedule is None else int(best_schedule["slot_total"]),
                "unused_slots": None if best_schedule is None else int(best_schedule["unused_slots"]),
                "total_rate_bps": None if best_schedule is None else float(best_schedule["total_rate_bps"]),
                "schedule_p_dc_total_avg_frame_w": None
                if best_schedule is None
                else float(best_schedule["schedule_p_dc_total_avg_frame_w"]),
                "error_message": result.error_message,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
