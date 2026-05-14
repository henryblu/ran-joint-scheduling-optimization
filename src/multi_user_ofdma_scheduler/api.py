from __future__ import annotations

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy

from .greedy_search import run_pa_aware_greedy_ofdma_schedule
from .ofdma_space import prepare_joint_ofdma_problem


def run_multi_user_ofdma_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Prepare the trusted OFDMA problem and run the greedy slot scheduler."""

    problem = prepare_joint_ofdma_problem(batch_space)
    return run_pa_aware_greedy_ofdma_schedule(problem, switch_policy=switch_policy)


__all__ = [
    "run_multi_user_ofdma_scheduler",
]
