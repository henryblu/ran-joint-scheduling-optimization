from __future__ import annotations

"""Shared models used by the day-run orchestration layer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from day_cycle_simulation.models import SyntheticSessionGenerationConfig

from .pa import PASwitchPolicy
from .scheduler import SchedulerMode


@dataclass(frozen=True)
class DayRunConfig:
    """Resolved inputs for one full-day multi-user scheduler run."""

    load_curve_csv: Path
    session_generation_config: SyntheticSessionGenerationConfig
    switch_policy: PASwitchPolicy
    cores: int
    output_dir: Path
    log_level: str | None
    scheduler_mode: SchedulerMode = SchedulerMode.TDMA


@dataclass
class BinRunResult:
    """Lean per-bin result kept by the day-run layer after scheduler execution."""

    bin_index: int
    status: str
    user_count: int
    single_user_elapsed_s: float = 0.0
    joint_elapsed_s: float = 0.0
    total_elapsed_s: float = 0.0
    best_schedule: dict[str, Any] | None = None


ExperimentConfig = DayRunConfig


__all__ = [
    "BinRunResult",
    "DayRunConfig",
    "ExperimentConfig",
]
