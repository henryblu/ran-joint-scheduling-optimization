from __future__ import annotations

"""Models for one finite-frame scheduler run."""

from dataclasses import dataclass

import pandas as pd

from models import MultiUserScheduleResult, PASwitchPolicy, SchedulerMode
from user_generation import UserGenerationConfig


@dataclass(frozen=True)
class FiniteFrameRunConfig:
    """Resolved inputs for one generated finite-frame scheduler run."""

    user_generation_config: UserGenerationConfig
    scheduler_mode: SchedulerMode
    switch_policy: PASwitchPolicy
    cores: int = 1


@dataclass(frozen=True)
class FiniteFrameRunResult:
    """Result and timings for the finite-frame workflow handoff."""

    status: str
    scheduler_user_table: pd.DataFrame
    schedule_result: MultiUserScheduleResult
    candidate_table_elapsed_s: float
    user_generation_elapsed_s: float
    candidate_lookup_elapsed_s: float
    scheduler_elapsed_s: float
    total_elapsed_s: float


__all__ = [
    "FiniteFrameRunConfig",
    "FiniteFrameRunResult",
]
