from models import BatchUserParameterSpace, PASwitchPolicy

from .joint_search import run_joint_schedule_search as _run_joint_schedule_search
from .models import MultiUserTdmaSchedulerResult, PreparedJointScheduleProblem
from .tdma_space import prepare_joint_schedule_problem as _prepare_joint_schedule_problem


def run_multi_user_tdma_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    window_n_frames=None,
    switch_policy: PASwitchPolicy = PASwitchPolicy.STANDBY,
) -> MultiUserTdmaSchedulerResult:
    """Prepare and run the exact TDMA scheduler from one trusted batch artifact."""

    problem = prepare_joint_schedule_problem(
        batch_space,
        window_n_frames=window_n_frames,
    )
    return run_joint_schedule_search(
        problem,
        switch_policy=switch_policy,
    )


def prepare_joint_schedule_problem(
    batch_space: BatchUserParameterSpace,
    window_n_frames=None,
    max_window_n_frames=32,
) -> PreparedJointScheduleProblem:
    """Prepare the exact TDMA scheduler problem from a trusted batch parameter-space artifact.

    Steps:
    1. Read the trusted per-user parameter spaces and user requirements from the batch artifact.
    2. Select the full-frame active operating points the TDMA scheduler owns.
    3. Resolve the repeated scheduling window in whole frames.
    4. Quantize and exact-prune the per-user TDMA spaces passed to the joint search.
    """

    return _prepare_joint_schedule_problem(
        batch_space,
        window_n_frames=window_n_frames,
        max_window_n_frames=max_window_n_frames,
    )


def run_joint_schedule_search(
    problem,
    switch_policy: PASwitchPolicy = PASwitchPolicy.STANDBY,
) -> MultiUserTdmaSchedulerResult:
    """Run the exact joint TDMA scheduler on one prepared scheduling problem."""

    result = _run_joint_schedule_search(
        problem,
        switch_policy=switch_policy,
    )
    if result.best_schedule is None:
        raise RuntimeError("No feasible joint TDMA schedule was found for the prepared user spaces.")
    return result
__all__ = [
    "run_multi_user_tdma_scheduler",
    "prepare_joint_schedule_problem",
    "run_joint_schedule_search",
]
