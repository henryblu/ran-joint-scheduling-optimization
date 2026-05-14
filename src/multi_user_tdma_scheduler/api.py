from __future__ import annotations

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy

from .joint_search import run_joint_schedule_search
from .models import PreparedJointScheduleProblem
from .tdma_space import prepare_joint_schedule_problem as _prepare_joint_schedule_problem


def run_multi_user_tdma_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Prepare the trusted TDMA problem and run the exact joint scheduler."""

    problem = prepare_joint_schedule_problem(batch_space)
    return run_joint_schedule_search(
        problem,
        switch_policy=switch_policy,
    )


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare the exact TDMA problem for notebooks, tests, and staged inspection."""

    return _prepare_joint_schedule_problem(batch_space)


__all__ = [
    "run_multi_user_tdma_scheduler",
    "prepare_joint_schedule_problem",
    "run_joint_schedule_search",
]
