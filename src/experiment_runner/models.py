from __future__ import annotations

"""Models for one finite-frame scheduler experiment run."""

from dataclasses import dataclass

import pandas as pd

from models import MultiUserScheduleResult, PASwitchPolicy, SchedulerMode
from user_generation import UserGenerationConfig


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Resolved inputs for one generated finite-frame scheduler experiment."""

    user_generation_config: UserGenerationConfig
    scheduler_mode: SchedulerMode
    switch_policy: PASwitchPolicy
    cores: int = 1


@dataclass(frozen=True)
class ExperimentRunResult:
    """Result and timings for one finite-frame experiment handoff."""

    status: str
    scheduler_user_table: pd.DataFrame
    schedule_result: MultiUserScheduleResult
    candidate_table_elapsed_s: float
    user_generation_elapsed_s: float
    candidate_lookup_elapsed_s: float
    scheduler_elapsed_s: float
    total_elapsed_s: float


__all__ = [
    "ExperimentRunConfig",
    "ExperimentRunResult",
]
