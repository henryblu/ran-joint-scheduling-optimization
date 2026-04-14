from models import BatchUserParameterSpace, PASwitchPolicy

from .joint_search import run_joint_schedule_search as _run_joint_schedule_search
from .models import MultiUserTdmaSchedulerResult, PreparedJointScheduleProblem
from .tdma_space import prepare_joint_schedule_problem as _prepare_joint_schedule_problem


def run_multi_user_tdma_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserTdmaSchedulerResult:
    """Run the one-step TDMA scheduler from one trusted batch artifact.

    Production callers should usually use this function, or the shared
    ``multi_user_scheduler.run_multi_user_scheduler`` dispatcher, instead of
    calling the TDMA preparation and search stages separately.
    """

    problem = prepare_joint_schedule_problem(batch_space)
    return run_joint_schedule_search(
        problem,
        switch_policy=switch_policy,
    )


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointScheduleProblem:
    """Prepare the exact TDMA problem for notebooks, tests, and staged inspection.

    Steps:
    1. Read the trusted per-user parameter spaces and user requirements from the batch artifact.
    2. Select the full-frame active operating points the TDMA scheduler owns.
    3. Check whether those rows fit inside one shared frame.
    4. Quantize and exact-prune the per-user TDMA spaces passed to the joint search.

    This is an advanced TDMA-specific entry point. Production orchestration
    should usually call ``run_multi_user_tdma_scheduler`` or the shared
    scheduler dispatcher instead.
    """

    return _prepare_joint_schedule_problem(batch_space)


def run_joint_schedule_search(
    problem,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserTdmaSchedulerResult:
    """Run the exact TDMA search on one prepared TDMA problem.

    This is an advanced TDMA-specific entry point intended for notebooks,
    tests, and debugging of an already prepared TDMA search space.
    """

    result = _run_joint_schedule_search(
        problem,
        switch_policy=switch_policy,
    )
    if result.best_schedule is None:
        raise RuntimeError("No feasible joint TDMA schedule was found for the prepared user spaces.")
    return result


def run_multi_user_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserTdmaSchedulerResult:
    """Expose the shared backend runner name for the TDMA scheduler package."""

    return run_multi_user_tdma_scheduler(
        batch_space,
        switch_policy=switch_policy,
    )

__all__ = [
    "run_multi_user_tdma_scheduler",
    "prepare_joint_schedule_problem",
    "run_joint_schedule_search",
]
