from __future__ import annotations

"""Shared public dispatcher for final multi-user scheduler backends."""

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy, SchedulerMode

from .k_milp.api import run_k_milp_scheduler
from .round_robin.api import run_round_robin_scheduler

MultiUserSchedulerResult = MultiUserScheduleResult


def run_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    scheduler_mode: SchedulerMode = SchedulerMode.K_MILP,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserSchedulerResult:
    """Run one final scheduler backend from a trusted batch artifact.

    Steps:
    1. Resolve the source-facing scheduler family.
    2. Dispatch the trusted batch artifact to the matching backend.
    3. Return the backend's shared public scheduler result directly.
    """

    resolved_mode = (
        scheduler_mode
        if isinstance(scheduler_mode, SchedulerMode)
        else SchedulerMode(str(scheduler_mode))
    )
    if resolved_mode == SchedulerMode.ROUND_ROBIN:
        return run_round_robin_scheduler(
            batch_space,
            switch_policy=switch_policy,
        )
    if resolved_mode == SchedulerMode.K_MILP:
        return run_k_milp_scheduler(
            batch_space,
            switch_policy=switch_policy,
        )

    raise ValueError(f"Unsupported scheduler mode: {resolved_mode}")


__all__ = [
    "MultiUserSchedulerResult",
    "run_scheduler",
]