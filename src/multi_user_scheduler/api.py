from __future__ import annotations

"""Shared public dispatcher for multi-user scheduler backends."""

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy, SchedulerMode
from multi_user_ofdma_scheduler.api import run_multi_user_ofdma_scheduler
from multi_user_tdma_scheduler.api import run_multi_user_tdma_scheduler

MultiUserSchedulerResult = MultiUserScheduleResult


def run_multi_user_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    scheduler_mode: SchedulerMode = SchedulerMode.TDMA,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserSchedulerResult:
    """Run the selected multi-user scheduler backend from one trusted batch artifact.

    Steps:
    1. Resolve the requested scheduler mode onto the shared public enum.
    2. Dispatch to the selected backend package.
    3. Return the backend's shared public scheduler result directly.
    """

    resolved_mode = (
        scheduler_mode
        if isinstance(scheduler_mode, SchedulerMode)
        else SchedulerMode(str(scheduler_mode))
    )
    if resolved_mode == SchedulerMode.TDMA:
        return run_multi_user_tdma_scheduler(
            batch_space,
            switch_policy=switch_policy,
        )
    if resolved_mode == SchedulerMode.OFDMA:
        return run_multi_user_ofdma_scheduler(
            batch_space,
            switch_policy=switch_policy,
        )

    raise ValueError(f"Unsupported scheduler mode: {resolved_mode}")


__all__ = [
    "MultiUserSchedulerResult",
    "run_multi_user_scheduler",
]
