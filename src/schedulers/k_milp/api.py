from __future__ import annotations

"""Public entry point for the independent OFDMA MILP oracle backend."""

from models import BatchUserParameterSpace, MultiUserScheduleResult, PASwitchPolicy
from schedulers.feasibility_bounds import (
    InfeasibilityCertificate,
    log_feasibility_certificate,
    row_menu_certificate,
)

from .bounded_pair_policy import build_and_solve_k1_bounded_restricted_pair_schedule
from .logging import log_frame_utilization_summary, log_problem_summary
from .milp_model import build_and_solve_milp_attempt, configured_max_users_per_slot
from .models import MilpAttemptResult, MilpModelSize, MilpVariableIndex, OfdmaMilpProblem
from .pattern_count import build_infeasible_pattern_result
from .problem import high_power_pa_ids, low_power_pa_ids, prepare_ofdma_milp_problem
from .result_builder import build_feasible_milp_result, build_infeasible_milp_result


def run_k_milp_scheduler(
    batch_space: BatchUserParameterSpace,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Prepare and solve the exact OFDMA slot-pattern count oracle."""

    max_users_per_slot = configured_max_users_per_slot()
    resolved_policy = switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy))
    problem = prepare_ofdma_milp_problem(
        batch_space,
        switch_policy=resolved_policy,
        prune_candidate_rows=max_users_per_slot is not None,
    )
    log_problem_summary(problem)
    if max_users_per_slot is not None:
        result = run_slot_indexed_sequence(
            problem,
            switch_policy=resolved_policy,
            max_users_per_slot=int(max_users_per_slot),
        )
        log_frame_utilization_summary(problem, result)
        return result

    if resolved_policy == PASwitchPolicy.HARD_OFF:
        result = run_hard_off_sequence(problem, batch_space=batch_space)
        log_frame_utilization_summary(problem, result)
        return result

    if resolved_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        result = run_single_family_sequence(
            problem,
            batch_space=batch_space,
            allowed_pa_ids=high_power_pa_ids(problem),
            attempt_name="baseline_8w_only",
        )
        log_frame_utilization_summary(problem, result)
        return result

    result = run_dual_switchable_sequence(problem, batch_space=batch_space)
    log_frame_utilization_summary(problem, result)
    return result


def run_single_family_sequence(
    problem: OfdmaMilpProblem,
    *,
    batch_space: BatchUserParameterSpace,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
) -> MultiUserScheduleResult:
    solve = build_and_solve_k1_bounded_restricted_pair_schedule(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        hard_off_details={"attempted_solve_count": 1},
    )
    return solve.result


def run_dual_switchable_sequence(problem: OfdmaMilpProblem, *, batch_space: BatchUserParameterSpace) -> MultiUserScheduleResult:
    """Run mixed-PA K2 first, then fall back to single-family subproblems if needed."""

    solve = build_and_solve_k1_bounded_restricted_pair_schedule(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=tuple(range(len(problem.pa_catalog))),
        attempt_name="dual_switchable",
        hard_off_details={"attempted_solve_count": 1},
    )
    if solve.result.feasible:
        return solve.result

    fallback_results = tuple(
        result
        for result in (
            run_single_family_sequence(
                problem,
                batch_space=batch_space,
                allowed_pa_ids=low_power_pa_ids(problem),
                attempt_name="dual_switchable_low_power_fallback",
            ),
            run_single_family_sequence(
                problem,
                batch_space=batch_space,
                allowed_pa_ids=high_power_pa_ids(problem),
                attempt_name="dual_switchable_high_power_fallback",
            ),
        )
        if result.feasible
    )
    if not fallback_results:
        return solve.result
    return min(fallback_results, key=lambda result: float(result.power_summary.frame_energy_j))


def run_hard_off_sequence(problem: OfdmaMilpProblem, *, batch_space: BatchUserParameterSpace) -> MultiUserScheduleResult:
    """Run low-power first, then high-power fallback only when needed."""

    low_solve = build_and_solve_k1_bounded_restricted_pair_schedule(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=low_power_pa_ids(problem),
        attempt_name="hard_off_low_power",
        hard_off_details={
            "hard_off_primary_family": "low_power",
            "hard_off_primary_feasible": True,
            "hard_off_fallback_used": False,
            "attempted_solve_count": 1,
        },
    )
    if low_solve.result.feasible:
        return low_solve.result

    high_solve = build_and_solve_k1_bounded_restricted_pair_schedule(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=high_power_pa_ids(problem),
        attempt_name="hard_off_high_power_fallback",
        hard_off_details={
            "hard_off_primary_family": "low_power",
            "hard_off_primary_feasible": False,
            "hard_off_fallback_used": True,
            "attempted_solve_count": 2,
        },
    )
    if high_solve.result.feasible:
        return high_solve.result

    return build_infeasible_pattern_result(
        problem,
        attempts=low_solve.attempts + high_solve.attempts,
        hard_off_details={
            "hard_off_primary_family": "low_power",
            "hard_off_primary_feasible": False,
            "hard_off_fallback_used": False,
            "attempted_solve_count": 2,
        },
    )


def run_slot_indexed_sequence(
    problem: OfdmaMilpProblem,
    *,
    switch_policy: PASwitchPolicy,
    max_users_per_slot: int,
) -> MultiUserScheduleResult:
    """Run the capped slot-indexed OFDMA MILP when the config requests a K-user oracle."""

    if switch_policy == PASwitchPolicy.HARD_OFF:
        return run_slot_indexed_hard_off_sequence(
            problem,
            max_users_per_slot=int(max_users_per_slot),
        )

    allowed_pa_ids = tuple(range(len(problem.pa_catalog)))
    attempt_name = "dual_switchable"
    if switch_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        allowed_pa_ids = high_power_pa_ids(problem)
        attempt_name = "baseline_8w_only"

    attempt = build_and_solve_slot_indexed_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        max_users_per_slot=int(max_users_per_slot),
    )
    hard_off_details = {
        "attempted_solve_count": 1,
        "max_users_per_slot": int(max_users_per_slot),
    }
    if attempt.success:
        return build_feasible_milp_result(
            problem,
            attempt=attempt,
            hard_off_details=hard_off_details,
        )
    return build_infeasible_milp_result(
        problem,
        attempts=(attempt,),
        hard_off_details=hard_off_details,
    )


def run_slot_indexed_hard_off_sequence(
    problem: OfdmaMilpProblem,
    *,
    max_users_per_slot: int,
) -> MultiUserScheduleResult:
    """Run capped slot-indexed low-power first, then high-power fallback."""

    low_attempt = build_and_solve_slot_indexed_attempt(
        problem,
        allowed_pa_ids=low_power_pa_ids(problem),
        attempt_name="hard_off_low_power",
        max_users_per_slot=int(max_users_per_slot),
    )
    if low_attempt.success:
        return build_feasible_milp_result(
            problem,
            attempt=low_attempt,
            hard_off_details={
                "hard_off_primary_family": "low_power",
                "hard_off_primary_feasible": True,
                "hard_off_fallback_used": False,
                "attempted_solve_count": 1,
                "max_users_per_slot": int(max_users_per_slot),
            },
        )

    high_attempt = build_and_solve_slot_indexed_attempt(
        problem,
        allowed_pa_ids=high_power_pa_ids(problem),
        attempt_name="hard_off_high_power_fallback",
        max_users_per_slot=int(max_users_per_slot),
    )
    hard_off_details = {
        "hard_off_primary_family": "low_power",
        "hard_off_primary_feasible": False,
        "hard_off_fallback_used": bool(high_attempt.success),
        "attempted_solve_count": 2,
        "max_users_per_slot": int(max_users_per_slot),
    }
    if high_attempt.success:
        return build_feasible_milp_result(
            problem,
            attempt=high_attempt,
            hard_off_details=hard_off_details,
        )
    return build_infeasible_milp_result(
        problem,
        attempts=(low_attempt, high_attempt),
        hard_off_details=hard_off_details,
    )


def build_and_solve_slot_indexed_attempt(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    max_users_per_slot: int,
) -> MilpAttemptResult:
    certificate = slot_indexed_attempt_certificate(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        max_users_per_slot=int(max_users_per_slot),
    )
    if certificate is not None:
        log_feasibility_certificate(
            certificate,
            scheduler_mode="ofdma_milp_single_snapshot",
            policy=problem.switch_policy.value,
            attempt_name=attempt_name,
        )
        return build_bound_infeasible_milp_attempt(
            allowed_pa_ids=allowed_pa_ids,
            attempt_name=attempt_name,
            certificate=certificate,
        )
    return build_and_solve_milp_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        max_users_per_slot=int(max_users_per_slot),
    )


def slot_indexed_attempt_certificate(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    max_users_per_slot: int,
) -> InfeasibilityCertificate | None:
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    bits_by_user = {
        int(user_id): tuple(
            float(row.bits_per_slot)
            for row in problem.candidate_rows_by_user.get(int(user_id), ())
            if int(row.pa_id) in allowed_pa_id_set
        )
        for user_id in sorted(problem.required_rate_by_user)
    }
    return row_menu_certificate(
        demand_bits_by_user=problem.demand_bits_by_user,
        bits_by_user=bits_by_user,
        frame_n_slots=int(problem.frame_n_slots),
        max_users_per_slot=int(max_users_per_slot),
    )


def build_bound_infeasible_milp_attempt(
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    certificate: InfeasibilityCertificate,
) -> MilpAttemptResult:
    return MilpAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=False,
        solver_status=2,
        solver_message=str(certificate.reason),
        objective_pwl_j=None,
        objective_bound=None,
        mip_gap=None,
        solution=(),
        model_size=MilpModelSize(
            variable_count=0,
            binary_variable_count=0,
            continuous_variable_count=0,
            constraint_count=0,
            nonzero_count=0,
        ),
        segments_by_pa={},
        variables=MilpVariableIndex(x={}, z={}, delta={}, beta={}, theta={}, w={}, v={}),
        build_elapsed_s=0.0,
        solve_elapsed_s=0.0,
        diagnostics={},
    )


__all__ = [
    "run_k_milp_scheduler",
]
