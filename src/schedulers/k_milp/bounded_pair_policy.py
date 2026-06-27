from __future__ import annotations

"""K1-bounded K2 policy for TDMA-contained OFDMA slot patterns."""

from dataclasses import dataclass
import logging
from time import perf_counter

from configs.scheduler import K_MILP_SOLVER_CONFIG
from models import BatchUserParameterSpace, MultiUserScheduleResult
from schedulers.feasibility_bounds import (
    InfeasibilityCertificate,
    log_feasibility_certificate,
    row_menu_certificate,
)
from run_reporting import build_console_message, current_run_scope

from .contained_patterns import build_one_ue_baseline_rows_by_user, build_tdma_contained_slot_patterns
from .k1_tdma_highs import K1TdmaHighsAttempt, build_k1_tdma_highs_result, solve_k1_tdma_highs_attempt
from .logging import active_snapshot_index_from_scope, log_admission_summary, log_restricted_pattern_summary
from .models import MilpModelSize, OfdmaMilpProblem, OfdmaSlotPattern, PatternCountAttemptResult
from .pattern_count import (
    build_infeasible_pattern_result,
    build_pattern_result,
    collapse_duplicate_slot_patterns,
    solve_pattern_count_attempt,
)


K2_CUTOFF_ABS_EPS_J = 1e-9
K2_CUTOFF_REL_EPS = 1e-6
TOL = 1e-9
LOGGER = logging.getLogger("snapshot_run")
ADMISSION_DUAL_PATTERN_STRATEGY = "admission"
SPLIT_TEMPLATE_DUAL_PATTERN_STRATEGY = "split_template"
DEFAULT_DUAL_PATTERN_STRATEGY = ADMISSION_DUAL_PATTERN_STRATEGY


@dataclass(frozen=True)
class BoundedPatternSolve:
    result: MultiUserScheduleResult
    attempts: tuple[object, ...]


def build_and_solve_k1_bounded_restricted_pair_schedule(
    problem: OfdmaMilpProblem,
    *,
    batch_space: BatchUserParameterSpace,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    hard_off_details: dict[str, object],
) -> BoundedPatternSolve:
    certificate = restricted_pair_attempt_certificate(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=allowed_pa_ids,
    )
    if certificate is not None:
        return build_bound_infeasible_schedule(
            problem,
            allowed_pa_ids=allowed_pa_ids,
            attempt_name=attempt_name,
            certificate=certificate,
            hard_off_details=hard_off_details,
        )

    k1_attempt = solve_k1_tdma_highs_attempt(
        problem,
        batch_space,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=f"{attempt_name}_k1",
    )
    dual_pattern_strategy = configured_dual_pattern_strategy()
    k2_patterns, build_elapsed_s = build_restricted_pair_pattern_sets(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=allowed_pa_ids,
        dual_pattern_strategy=dual_pattern_strategy,
    )

    if k1_attempt.success:
        return solve_k2_with_k1_cutoff(
            problem,
            allowed_pa_ids=allowed_pa_ids,
            attempt_name=attempt_name,
            patterns=k2_patterns,
            build_elapsed_s=float(build_elapsed_s),
            k1_attempt=k1_attempt,
            hard_off_details=hard_off_details,
        )

    if not k2_patterns:
        empty_attempt = build_empty_pattern_attempt(
            allowed_pa_ids=allowed_pa_ids,
            attempt_name=f"{attempt_name}_k2_without_k1_cutoff",
            build_elapsed_s=float(build_elapsed_s),
        )
        fallback = solve_split_template_fallback_without_k1_cutoff(
            problem,
            batch_space=batch_space,
            allowed_pa_ids=allowed_pa_ids,
            attempt_name=attempt_name,
            hard_off_details=hard_off_details,
            prior_attempts=(k1_attempt, empty_attempt),
            primary_dual_pattern_strategy=dual_pattern_strategy,
        )
        if fallback is not None:
            return fallback
        return BoundedPatternSolve(
            result=build_infeasible_pattern_result(problem, attempts=(empty_attempt,), hard_off_details=hard_off_details),
            attempts=(empty_attempt,),
        )

    k2_attempt = solve_pattern_count_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=f"{attempt_name}_k2_without_k1_cutoff",
        patterns=k2_patterns,
        build_elapsed_s=float(build_elapsed_s),
        mip_rel_gap=K_MILP_SOLVER_CONFIG.k2_accept_rel_gap,
        time_limit_s=K_MILP_SOLVER_CONFIG.k2_cutoff_time_limit_s,
    )
    if k2_attempt.success:
        return BoundedPatternSolve(
            result=build_pattern_result(problem, attempt=k2_attempt, hard_off_details=hard_off_details),
            attempts=(k1_attempt, k2_attempt),
        )
    fallback = solve_split_template_fallback_without_k1_cutoff(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        hard_off_details=hard_off_details,
        prior_attempts=(k1_attempt, k2_attempt),
        primary_dual_pattern_strategy=dual_pattern_strategy,
    )
    if fallback is not None:
        return fallback
    return BoundedPatternSolve(
        result=build_infeasible_pattern_result(problem, attempts=(k1_attempt, k2_attempt), hard_off_details=hard_off_details),
        attempts=(k1_attempt, k2_attempt),
    )


def build_bound_infeasible_schedule(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    certificate: InfeasibilityCertificate,
    hard_off_details: dict[str, object],
) -> BoundedPatternSolve:
    log_feasibility_certificate(
        certificate,
        scheduler_mode="ofdma_milp_single_snapshot",
        policy=problem.switch_policy.value,
        attempt_name=attempt_name,
    )
    attempt = build_bound_infeasible_pattern_attempt(
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        certificate=certificate,
    )
    return BoundedPatternSolve(
        result=build_infeasible_pattern_result(problem, attempts=(attempt,), hard_off_details=hard_off_details),
        attempts=(attempt,),
    )


def build_restricted_pair_pattern_sets(
    problem: OfdmaMilpProblem,
    *,
    batch_space: BatchUserParameterSpace,
    allowed_pa_ids: tuple[int, ...],
    dual_pattern_strategy: str | None = None,
) -> tuple[tuple[OfdmaSlotPattern, ...], float]:
    build_started_at = perf_counter()
    raw_patterns, stats = build_tdma_contained_slot_patterns(
        problem,
        batch_space,
        allowed_pa_ids=allowed_pa_ids,
        max_dual_rows_per_user=K_MILP_SOLVER_CONFIG.dual_admitted_rows_per_user,
        dual_pattern_strategy=configured_dual_pattern_strategy() if dual_pattern_strategy is None else str(dual_pattern_strategy),
    )
    one_ue_raw_patterns = tuple(pattern for pattern in raw_patterns if len(pattern.rows) == 1)
    one_ue_pattern_count = len(one_ue_raw_patterns)
    valid_dual_ue_pattern_count = sum(1 for pattern in raw_patterns if len(pattern.rows) == 2)
    k2_patterns = collapse_duplicate_slot_patterns(problem, raw_patterns)
    retained_dual_ue_pattern_count = sum(1 for pattern in k2_patterns if len(pattern.rows) == 2)
    build_elapsed_s = float(perf_counter() - build_started_at)
    log_admission_summary(
        max_rows_per_user=int(stats.max_dual_rows_per_user),
        raw_rows_by_user=stats.dual_raw_rows_by_user,
        admitted_rows_by_user=stats.dual_admitted_rows_by_user,
    )
    log_restricted_pattern_summary(
        one_ue_pattern_count=int(one_ue_pattern_count),
        raw_dual_ue_pair_bound=int(stats.raw_dual_ue_pair_bound),
        valid_dual_ue_pattern_count=int(valid_dual_ue_pattern_count),
        retained_dual_ue_pattern_count=int(retained_dual_ue_pattern_count),
    )
    return k2_patterns, build_elapsed_s


def configured_dual_pattern_strategy() -> str:
    return str(K_MILP_SOLVER_CONFIG.dual_pattern_strategy or DEFAULT_DUAL_PATTERN_STRATEGY)


def solve_split_template_fallback_without_k1_cutoff(
    problem: OfdmaMilpProblem,
    *,
    batch_space: BatchUserParameterSpace,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    hard_off_details: dict[str, object],
    prior_attempts: tuple[object, ...],
    primary_dual_pattern_strategy: str,
) -> BoundedPatternSolve | None:
    if str(primary_dual_pattern_strategy) != ADMISSION_DUAL_PATTERN_STRATEGY:
        return None

    fallback_patterns, fallback_build_elapsed_s = build_restricted_pair_pattern_sets(
        problem,
        batch_space=batch_space,
        allowed_pa_ids=allowed_pa_ids,
        dual_pattern_strategy=SPLIT_TEMPLATE_DUAL_PATTERN_STRATEGY,
    )
    if not fallback_patterns:
        return None

    fallback_attempt = solve_pattern_count_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=f"{attempt_name}_k2_split_template_fallback",
        patterns=fallback_patterns,
        build_elapsed_s=float(fallback_build_elapsed_s),
        mip_rel_gap=K_MILP_SOLVER_CONFIG.k2_accept_rel_gap,
        time_limit_s=K_MILP_SOLVER_CONFIG.k2_cutoff_time_limit_s,
    )
    attempts = (*prior_attempts, fallback_attempt)
    if fallback_attempt.success:
        return BoundedPatternSolve(
            result=build_pattern_result(problem, attempt=fallback_attempt, hard_off_details=hard_off_details),
            attempts=attempts,
        )
    return BoundedPatternSolve(
        result=build_infeasible_pattern_result(problem, attempts=attempts, hard_off_details=hard_off_details),
        attempts=attempts,
    )


def solve_k2_with_k1_cutoff(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    patterns: tuple[OfdmaSlotPattern, ...],
    build_elapsed_s: float,
    k1_attempt: K1TdmaHighsAttempt,
    hard_off_details: dict[str, object],
) -> BoundedPatternSolve:
    k2_energy_cutoff_j = build_k2_energy_cutoff(k1_attempt.objective_j)
    k2_attempt = solve_pattern_count_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=f"{attempt_name}_k2",
        patterns=patterns,
        build_elapsed_s=float(build_elapsed_s),
        k2_energy_cutoff_j=float(k2_energy_cutoff_j),
        mip_rel_gap=K_MILP_SOLVER_CONFIG.k2_accept_rel_gap,
        time_limit_s=K_MILP_SOLVER_CONFIG.k2_cutoff_time_limit_s,
    )
    accepted = should_accept_k2_attempt(
        k2_attempt,
        k2_energy_cutoff_j=float(k2_energy_cutoff_j),
        mip_rel_gap_target=K_MILP_SOLVER_CONFIG.k2_accept_rel_gap,
    )
    log_k2_cutoff_decision(
        attempt_name=attempt_name,
        accepted=accepted,
        k1_objective_j=float(k1_attempt.objective_j),
        k2_energy_cutoff_j=float(k2_energy_cutoff_j),
        k2_objective_j=k2_attempt.objective_j,
        k2_mip_gap=k2_attempt.mip_gap,
    )
    if accepted:
        return BoundedPatternSolve(
            result=build_pattern_result(problem, attempt=k2_attempt, hard_off_details=hard_off_details),
            attempts=(k1_attempt, k2_attempt),
        )
    return BoundedPatternSolve(
        result=build_k1_tdma_highs_result(problem, attempt=k1_attempt, hard_off_details=hard_off_details),
        attempts=(k1_attempt, k2_attempt),
    )


def log_k2_cutoff_decision(
    *,
    attempt_name: str,
    accepted: bool,
    k1_objective_j: float,
    k2_energy_cutoff_j: float,
    k2_objective_j: float | None,
    k2_mip_gap: float | None,
) -> None:
    LOGGER.info(
        build_console_message(
            level_tag="INFO",
            scope=current_scope(),
            stage="pattern",
            event="k2_cutoff_decision",
            fields=[
                ("attempt", str(attempt_name)),
                ("accepted", str(bool(accepted))),
                ("k1_objective_j", f"{float(k1_objective_j):.12g}"),
                ("k2_cutoff_j", f"{float(k2_energy_cutoff_j):.12g}"),
                ("k2_objective_j", "None" if k2_objective_j is None else f"{float(k2_objective_j):.12g}"),
                ("k2_gap", "None" if k2_mip_gap is None else f"{float(k2_mip_gap):.12g}"),
            ],
        )
    )


def current_scope() -> str:
    return current_run_scope()


def build_k2_energy_cutoff(k1_objective_j: float) -> float:
    return float(k1_objective_j) + k2_energy_cutoff_tolerance_j(float(k1_objective_j))


def k2_energy_cutoff_tolerance_j(k1_objective_j: float) -> float:
    return max(float(K2_CUTOFF_ABS_EPS_J), float(K2_CUTOFF_REL_EPS) * abs(float(k1_objective_j)))


def should_accept_k2_attempt(
    attempt,
    *,
    k2_energy_cutoff_j: float,
    mip_rel_gap_target: float,
) -> bool:
    if not attempt.success:
        return False
    if attempt.objective_j is None:
        return False
    if float(attempt.objective_j) > float(k2_energy_cutoff_j) + TOL:
        return False
    if attempt.mip_gap is None:
        return False
    return float(attempt.mip_gap) <= float(mip_rel_gap_target) + TOL


def restricted_pair_attempt_certificate(
    problem: OfdmaMilpProblem,
    *,
    batch_space: BatchUserParameterSpace,
    allowed_pa_ids: tuple[int, ...],
) -> InfeasibilityCertificate | None:
    baseline_rows_by_user = build_one_ue_baseline_rows_by_user(
        batch_space,
        allowed_pa_ids=allowed_pa_ids,
    )
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    bits_by_user = {}
    for user_id in sorted(problem.required_rate_by_user):
        row_bits = [
            float(row.bits_per_slot)
            for row in baseline_rows_by_user.get(int(user_id), ())
        ]
        row_bits.extend(
            float(row.bits_per_slot)
            for row in problem.candidate_rows_by_user.get(int(user_id), ())
            if int(row.pa_id) in allowed_pa_id_set
        )
        bits_by_user[int(user_id)] = tuple(row_bits)
    return row_menu_certificate(
        demand_bits_by_user=problem.demand_bits_by_user,
        bits_by_user=bits_by_user,
        frame_n_slots=int(problem.frame_n_slots),
        max_users_per_slot=2,
    )


def build_empty_pattern_attempt(
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    build_elapsed_s: float,
) -> PatternCountAttemptResult:
    return PatternCountAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=False,
        solver_status=2,
        solver_message="No feasible slot patterns were generated.",
        objective_j=None,
        objective_bound=None,
        mip_gap=None,
        solution=(),
        patterns=(),
        build_elapsed_s=float(build_elapsed_s),
        solve_elapsed_s=0.0,
        model_size=MilpModelSize(
            variable_count=0,
            binary_variable_count=0,
            continuous_variable_count=0,
            constraint_count=0,
            nonzero_count=0,
        ),
    )


def build_bound_infeasible_pattern_attempt(
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    certificate: InfeasibilityCertificate,
) -> PatternCountAttemptResult:
    return PatternCountAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=False,
        solver_status=2,
        solver_message=str(certificate.reason),
        objective_j=None,
        objective_bound=None,
        mip_gap=None,
        solution=(),
        patterns=(),
        build_elapsed_s=0.0,
        solve_elapsed_s=0.0,
        model_size=MilpModelSize(
            variable_count=0,
            binary_variable_count=0,
            continuous_variable_count=0,
            constraint_count=0,
            nonzero_count=0,
        ),
    )
