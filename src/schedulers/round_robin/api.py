from __future__ import annotations

"""Public entry point for the OFDMA rolling-quantum round-robin baseline."""

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy

from .logging import log_frame_utilization_summary
from .models import RoundRobinAttemptResult, RoundRobinProblem
from .problem import high_power_pa_ids, low_power_pa_ids, prepare_round_robin_problem
from .result_builder import build_infeasible_round_robin_result, build_round_robin_result
from .round_robin import run_round_robin_attempt


def run_round_robin_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Prepare and run the deterministic rolling-quantum OFDMA baseline."""

    resolved_policy = switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy))
    problem = prepare_round_robin_problem(batch_space, switch_policy=resolved_policy)
    if resolved_policy == PASwitchPolicy.HARD_OFF:
        result = run_hard_off_sequence(problem)
        log_frame_utilization_summary(problem, result)
        return result

    if resolved_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        attempt = run_round_robin_attempt(
            problem,
            allowed_pa_ids=high_power_pa_ids(problem),
            attempt_name="baseline_8w_only",
        )
        result = build_single_attempt_result(problem, attempt)
        log_frame_utilization_summary(problem, result)
        return result

    attempts = tuple(
        run_round_robin_attempt(
            problem,
            allowed_pa_ids=(int(pa_id),),
            attempt_name=f"dual_switchable_pa{int(pa_id)}",
        )
        for pa_id in range(len(problem.pa_catalog))
    )
    result = build_best_attempt_result(problem, attempts)
    log_frame_utilization_summary(problem, result)
    return result


def run_hard_off_sequence(problem: RoundRobinProblem) -> MultiUserScheduleResult:
    """Run low-power first, then high-power fallback only when needed."""

    low_attempt = run_round_robin_attempt(
        problem,
        allowed_pa_ids=low_power_pa_ids(problem),
        attempt_name="hard_off_low_power",
    )
    if low_attempt.success:
        return build_round_robin_result(
            problem,
            attempt=low_attempt,
            hard_off_details={
                "hard_off_primary_family": "low_power",
                "hard_off_primary_feasible": True,
                "hard_off_fallback_used": False,
                "attempted_solve_count": 1,
            },
        )

    high_attempt = run_round_robin_attempt(
        problem,
        allowed_pa_ids=high_power_pa_ids(problem),
        attempt_name="hard_off_high_power_fallback",
    )
    hard_off_details = {
        "hard_off_primary_family": "low_power",
        "hard_off_primary_feasible": False,
        "hard_off_fallback_used": bool(high_attempt.success),
        "attempted_solve_count": 2,
    }
    if high_attempt.success:
        return build_round_robin_result(
            problem,
            attempt=high_attempt,
            hard_off_details=hard_off_details,
        )
    return build_infeasible_round_robin_result(
        problem,
        attempts=(low_attempt, high_attempt),
        hard_off_details=hard_off_details,
    )


def build_single_attempt_result(
    problem: RoundRobinProblem,
    attempt: RoundRobinAttemptResult,
) -> MultiUserScheduleResult:
    hard_off_details = {"attempted_solve_count": 1}
    if attempt.success:
        return build_round_robin_result(
            problem,
            attempt=attempt,
            hard_off_details=hard_off_details,
        )
    return build_infeasible_round_robin_result(
        problem,
        attempts=(attempt,),
        hard_off_details=hard_off_details,
    )


def build_best_attempt_result(
    problem: RoundRobinProblem,
    attempts: tuple[RoundRobinAttemptResult, ...],
) -> MultiUserScheduleResult:
    best_attempt = min(attempts, key=attempt_rank)
    hard_off_details = {"attempted_solve_count": int(len(attempts))}
    if best_attempt.success:
        return build_round_robin_result(
            problem,
            attempt=best_attempt,
            hard_off_details=hard_off_details,
        )
    return build_infeasible_round_robin_result(
        problem,
        attempts=attempts,
        hard_off_details=hard_off_details,
    )


def attempt_rank(attempt: RoundRobinAttemptResult) -> tuple[int, float, int, int, str]:
    return (
        0 if attempt.success else 1,
        float("inf") if attempt.frame_energy_j is None else float(attempt.frame_energy_j),
        int(attempt.active_slot_count),
        int(attempt.allocation_count),
        str(attempt.attempt_name),
    )


__all__ = [
    "run_round_robin_scheduler",
]
