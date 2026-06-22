from __future__ import annotations

"""Exact OFDMA oracle over interchangeable one-slot allocation patterns."""

import logging
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from configs.scheduler import K_MILP_SOLVER_CONFIG
from configs.pa import pa_slot_dc_power
from models import MultiUserScheduleResult, SchedulerMode, SchedulerPowerSummary, SlotAllocation, SlotSchedule, UserScheduleSummary
from run_reporting import build_console_message, current_run_scope

from .diagnostics import get_optional_float
from .logging import active_snapshot_index_from_scope
from .models import MilpCandidateRow, MilpModelSize, OfdmaMilpProblem, OfdmaSlotPattern, PatternCountAttemptResult
from .problem import candidate_rows_by_user_pa


TOL = 1e-9
LOGGER = logging.getLogger("snapshot_run")


def build_and_solve_pattern_count_attempt(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
) -> PatternCountAttemptResult:
    """Build and solve the exact slot-pattern count MILP for one PA policy attempt."""

    build_started_at = perf_counter()
    patterns = build_pruned_slot_patterns(problem, allowed_pa_ids=allowed_pa_ids)
    build_elapsed_s = float(perf_counter() - build_started_at)
    return solve_pattern_count_attempt(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        attempt_name=attempt_name,
        patterns=patterns,
        build_elapsed_s=build_elapsed_s,
    )


def solve_pattern_count_attempt(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    patterns: tuple[OfdmaSlotPattern, ...],
    build_elapsed_s: float,
    k2_energy_cutoff_j: float | None = None,
    mip_rel_gap: float | None = None,
    time_limit_s: float | None = None,
) -> PatternCountAttemptResult:
    """Solve one pattern-count MILP attempt for an already-built pattern library."""

    constraint = build_pattern_constraints(
        problem,
        patterns,
        k2_energy_cutoff_j=k2_energy_cutoff_j,
    )
    model_size = MilpModelSize(
        variable_count=int(len(patterns)),
        binary_variable_count=0,
        continuous_variable_count=0,
        constraint_count=int(constraint.A.shape[0]),
        nonzero_count=int(constraint.A.nnz),
    )
    log_pattern_model_summary(attempt_name=attempt_name, patterns=patterns, model_size=model_size, build_elapsed_s=build_elapsed_s)

    solve_started_at = perf_counter()
    result = milp(
        c=np.asarray([pattern.slot_energy_j for pattern in patterns], dtype=float),
        integrality=np.ones(len(patterns), dtype=int),
        bounds=Bounds(np.zeros(len(patterns), dtype=float), np.full(len(patterns), float(problem.frame_n_slots))),
        constraints=constraint,
        options=build_pattern_milp_options(mip_rel_gap=mip_rel_gap, time_limit_s=time_limit_s),
    )
    solve_elapsed_s = float(perf_counter() - solve_started_at)
    log_pattern_solve_summary(
        attempt_name=attempt_name,
        status=int(result.status),
        success=bool(result.success),
        objective_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
        solve_elapsed_s=solve_elapsed_s,
    )
    return PatternCountAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=bool(result.success),
        solver_status=int(result.status),
        solver_message=str(result.message),
        objective_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
        solution=result.x,
        patterns=patterns,
        build_elapsed_s=build_elapsed_s,
        solve_elapsed_s=solve_elapsed_s,
        model_size=model_size,
    )


def build_pruned_slot_patterns(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> tuple[OfdmaSlotPattern, ...]:
    """Enumerate feasible single-slot OFDMA patterns and collapse exact duplicates."""

    patterns = build_slot_patterns(problem, allowed_pa_ids=allowed_pa_ids)
    return collapse_duplicate_slot_patterns(problem, patterns)


def build_slot_patterns(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> tuple[OfdmaSlotPattern, ...]:
    user_ids = tuple(sorted(problem.candidate_rows_by_user))
    patterns = []
    pattern_id = 0
    for pa_id in allowed_pa_ids:
        choices_by_user = [
            (None,)
            + tuple(row for row in problem.candidate_rows_by_user[int(user_id)] if int(row.pa_id) == int(pa_id))
            for user_id in user_ids
        ]
        for selected_rows in product(*choices_by_user):
            rows = tuple(row for row in selected_rows if row is not None)
            if not rows:
                continue
            pattern = build_slot_pattern(problem, pattern_id=pattern_id, pa_id=int(pa_id), rows=rows)
            if pattern is None:
                continue
            patterns.append(pattern)
            pattern_id += 1
    return tuple(patterns)


def build_slot_pattern(
    problem: OfdmaMilpProblem,
    *,
    pattern_id: int,
    pa_id: int,
    rows: tuple[MilpCandidateRow, ...],
) -> OfdmaSlotPattern | None:
    used_prbs = int(sum(int(row.n_prb) for row in rows))
    aggregate_p_out_w = float(sum(float(row.p_out_total_w) for row in rows))
    pa = problem.pa_catalog[int(pa_id)]
    if used_prbs > int(problem.prb_max):
        return None
    if aggregate_p_out_w > float(problem.n_tx_chains) * float(pa.p_max_w) + TOL:
        return None

    dc_power_w = pa_slot_dc_power(
        pa,
        p_out_total_w=float(aggregate_p_out_w),
        n_tx_chains=int(problem.n_tx_chains),
        prb_fraction=float(used_prbs) / float(problem.prb_max),
    )
    return OfdmaSlotPattern(
        pattern_id=int(pattern_id),
        pa_id=int(pa_id),
        rows=tuple(sorted(rows, key=lambda row: int(row.user_id))),
        used_prbs=int(used_prbs),
        aggregate_p_out_w=float(aggregate_p_out_w),
        dc_power_w=float(dc_power_w),
        slot_energy_j=float(dc_power_w) * float(problem.t_slot_s),
        delivered_bits_by_user={
            int(row.user_id): float(row.bits_per_slot)
            for row in rows
        },
    )


def collapse_duplicate_slot_patterns(
    problem: OfdmaMilpProblem,
    patterns: tuple[OfdmaSlotPattern, ...],
) -> tuple[OfdmaSlotPattern, ...]:
    best_by_effective_bits = {}
    for pattern in patterns:
        key = tuple(
            (int(user_id), float(bits))
            for user_id, bits in sorted(effective_pattern_bits(problem, pattern).items())
        )
        incumbent = best_by_effective_bits.get(key)
        if incumbent is not None and pattern_rank(incumbent) <= pattern_rank(pattern):
            continue
        best_by_effective_bits[key] = pattern
    kept = tuple(sorted(best_by_effective_bits.values(), key=pattern_rank))
    return tuple(
        OfdmaSlotPattern(
            pattern_id=index,
            pa_id=pattern.pa_id,
            rows=pattern.rows,
            used_prbs=pattern.used_prbs,
            aggregate_p_out_w=pattern.aggregate_p_out_w,
            dc_power_w=pattern.dc_power_w,
            slot_energy_j=pattern.slot_energy_j,
            delivered_bits_by_user=pattern.delivered_bits_by_user,
        )
        for index, pattern in enumerate(kept)
    )


def effective_pattern_bits(problem: OfdmaMilpProblem, pattern: OfdmaSlotPattern) -> dict[int, float]:
    return {
        int(user_id): min(
            float(pattern.delivered_bits_by_user.get(int(user_id), 0.0)),
            float(problem.demand_bits_by_user[int(user_id)]),
        )
        for user_id in problem.required_rate_by_user
    }


def pattern_rank(pattern: OfdmaSlotPattern) -> tuple[float, int, float, tuple]:
    return (
        float(pattern.slot_energy_j),
        int(pattern.used_prbs),
        float(pattern.aggregate_p_out_w),
        tuple((int(row.user_id), int(row.local_row_id)) for row in pattern.rows),
    )


def build_pattern_constraints(
    problem: OfdmaMilpProblem,
    patterns: tuple[OfdmaSlotPattern, ...],
    *,
    k2_energy_cutoff_j: float | None = None,
) -> LinearConstraint:
    rows = [0 for _ in patterns]
    cols = [int(index) for index in range(len(patterns))]
    values = [1.0 for _ in patterns]
    lower_bounds = [-np.inf]
    upper_bounds = [float(problem.frame_n_slots)]
    row_id = 1
    for user_id in sorted(problem.required_rate_by_user):
        demand_bits = float(problem.demand_bits_by_user[int(user_id)])
        for pattern_index, pattern in enumerate(patterns):
            coefficient = float(pattern.delivered_bits_by_user.get(int(user_id), 0.0)) / demand_bits
            if coefficient <= TOL:
                continue
            rows.append(int(row_id))
            cols.append(int(pattern_index))
            values.append(float(coefficient))
        lower_bounds.append(1.0)
        upper_bounds.append(np.inf)
        row_id += 1

    if k2_energy_cutoff_j is not None:
        for pattern_index, pattern in enumerate(patterns):
            rows.append(int(row_id))
            cols.append(int(pattern_index))
            values.append(float(pattern.slot_energy_j))
        lower_bounds.append(-np.inf)
        upper_bounds.append(float(k2_energy_cutoff_j))

    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower_bounds), len(patterns))).tocsr()
    return LinearConstraint(matrix, np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float))


def build_pattern_result(
    problem: OfdmaMilpProblem,
    *,
    attempt: PatternCountAttemptResult,
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    selected_patterns = decode_selected_patterns(attempt)
    slot_schedules = build_pattern_slot_schedules(problem, selected_patterns=selected_patterns)
    delivered_bits_by_user = {
        int(user_id): sum(
            float(allocation.bits_per_slot)
            for slot in slot_schedules
            for allocation in slot.allocations
            if int(allocation.user_id) == int(user_id)
        )
        for user_id in problem.required_rate_by_user
    }
    frame_energy_j = float(problem.t_slot_s) * float(sum(slot.dc_power_w for slot in slot_schedules))
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.K_MILP,
        feasible=True,
        infeasible_reason=None,
        power_summary=SchedulerPowerSummary(
            frame_energy_j=float(frame_energy_j),
            average_frame_dc_power_w=float(frame_energy_j) / max(float(frame_duration_s), TOL),
            active_energy_j=float(frame_energy_j),
            inactive_energy_j=0.0,
            average_frame_rf_output_w=float(sum(slot.aggregate_p_out_w for slot in slot_schedules)) / max(int(problem.frame_n_slots), 1),
        ),
        user_summaries=build_pattern_user_summaries(problem, delivered_bits_by_user=delivered_bits_by_user),
        slot_schedules=slot_schedules,
        solver_details=build_pattern_solver_details(problem, attempt=attempt, hard_off_details=hard_off_details),
    )


def build_infeasible_pattern_result(
    problem: OfdmaMilpProblem,
    *,
    attempts: tuple[object, ...],
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.K_MILP,
        feasible=False,
        infeasible_reason=infeasible_reason_from_pattern_attempts(attempts),
        power_summary=SchedulerPowerSummary(
            frame_energy_j=0.0,
            average_frame_dc_power_w=0.0,
            active_energy_j=0.0,
            inactive_energy_j=0.0,
            average_frame_rf_output_w=0.0,
        ),
        user_summaries=build_pattern_user_summaries(problem, delivered_bits_by_user={int(user_id): 0.0 for user_id in problem.required_rate_by_user}),
        slot_schedules=tuple(
            SlotSchedule(slot_index=slot_id, active=False, pa_id=None, used_prbs=0, aggregate_p_out_w=0.0, dc_power_w=0.0, allocations=())
            for slot_id in range(int(problem.frame_n_slots))
        ),
        solver_details={
            "algorithm": "ofdma_slot_pattern_count_milp",
            "frame_duration_s": float(frame_duration_s),
            "attempted_solve_count": int(len(attempts)),
            "attempts": [
                {"attempt_name": attempt.attempt_name, "solver_status": attempt.solver_status, "success": attempt.success, "message": attempt.solver_message}
                for attempt in attempts
            ],
            **hard_off_details,
        },
    )


def infeasible_reason_from_pattern_attempts(attempts: tuple[object, ...]) -> str:
    bound_messages = tuple(
        str(getattr(attempt, "solver_message"))
        for attempt in attempts
        if bound_certificate_message(getattr(attempt, "solver_message", None))
    )
    if bound_messages and len(bound_messages) == len(attempts):
        return str(bound_messages[-1])
    return "No feasible OFDMA slot-pattern count schedule was found for the prepared user spaces."


def bound_certificate_message(message: object) -> bool:
    if message is None:
        return False
    return str(message).startswith("Bound-certified infeasible:")


def decode_selected_patterns(attempt: PatternCountAttemptResult) -> tuple[OfdmaSlotPattern, ...]:
    selected = []
    for pattern_index, value in enumerate(attempt.solution):
        count = int(round(float(value)))
        selected.extend([attempt.patterns[int(pattern_index)]] * count)
    return tuple(sorted(selected, key=lambda pattern: (float(pattern.slot_energy_j), int(pattern.pa_id), tuple(int(row.user_id) for row in pattern.rows))))


def build_pattern_slot_schedules(
    problem: OfdmaMilpProblem,
    *,
    selected_patterns: tuple[OfdmaSlotPattern, ...],
) -> tuple[SlotSchedule, ...]:
    slot_schedules = [
        SlotSchedule(
            slot_index=slot_id,
            active=True,
            pa_id=int(pattern.pa_id),
            used_prbs=int(pattern.used_prbs),
            aggregate_p_out_w=float(pattern.aggregate_p_out_w),
            dc_power_w=float(pattern.dc_power_w),
            allocations=tuple(slot_allocation_from_row(row) for row in pattern.rows),
        )
        for slot_id, pattern in enumerate(selected_patterns)
    ]
    for slot_id in range(len(slot_schedules), int(problem.frame_n_slots)):
        slot_schedules.append(SlotSchedule(slot_index=slot_id, active=False, pa_id=None, used_prbs=0, aggregate_p_out_w=0.0, dc_power_w=0.0, allocations=()))
    return tuple(slot_schedules)


def slot_allocation_from_row(row: MilpCandidateRow) -> SlotAllocation:
    return SlotAllocation(
        user_id=int(row.user_id),
        pa_id=int(row.pa_id),
        n_prb=int(row.n_prb),
        layers=int(row.layers),
        mcs=int(row.mcs),
        bits_per_slot=float(row.bits_per_slot),
        p_out_total_w=float(row.p_out_total_w),
        p_dc_active_w=float(row.p_dc_active_w),
    )


def build_pattern_user_summaries(
    problem: OfdmaMilpProblem,
    *,
    delivered_bits_by_user: dict[int, float],
) -> tuple[UserScheduleSummary, ...]:
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    return tuple(
        UserScheduleSummary(
            user_id=int(user_id),
            required_bits=float(problem.demand_bits_by_user[int(user_id)]),
            delivered_bits=float(delivered_bits_by_user[int(user_id)]),
            required_rate_bps=float(problem.required_rate_by_user[int(user_id)]),
            delivered_rate_bps=float(delivered_bits_by_user[int(user_id)]) / max(float(frame_duration_s), TOL),
            satisfied=float(delivered_bits_by_user[int(user_id)]) + TOL >= float(problem.demand_bits_by_user[int(user_id)]),
        )
        for user_id in sorted(problem.required_rate_by_user)
    )


def build_pattern_solver_details(
    problem: OfdmaMilpProblem,
    *,
    attempt: PatternCountAttemptResult,
    hard_off_details: dict[str, object],
) -> dict[str, object]:
    selected_patterns = decode_selected_patterns(attempt)
    return {
        "algorithm": "ofdma_slot_pattern_count_milp",
        "scheduler_mode": SchedulerMode.K_MILP.value,
        "is_oracle": True,
        "is_single_snapshot_mode": True,
        "solver_backend": "scipy.optimize.milp/highs",
        "solver_status": int(attempt.solver_status),
        "solver_message": str(attempt.solver_message),
        "success": bool(attempt.success),
        "objective_j": attempt.objective_j,
        "objective_bound": attempt.objective_bound,
        "mip_gap": attempt.mip_gap,
        "build_elapsed_s": float(attempt.build_elapsed_s),
        "solve_elapsed_s": float(attempt.solve_elapsed_s),
        "variable_count": int(attempt.model_size.variable_count),
        "integer_variable_count": int(attempt.model_size.variable_count),
        "constraint_count": int(attempt.model_size.constraint_count),
        "nonzero_count": int(attempt.model_size.nonzero_count),
        "slot_pattern_count": int(len(attempt.patterns)),
        "active_slot_count": int(len(selected_patterns)),
        "allocation_count": int(sum(len(pattern.rows) for pattern in selected_patterns)),
        "candidate_rows_by_user_pa": candidate_rows_by_user_pa(problem),
        "pa_policy": str(problem.switch_policy.value),
        "selected_snapshot_index": active_snapshot_index_from_scope(),
        **hard_off_details,
    }


def build_pattern_milp_options(
    *,
    mip_rel_gap: float | None = None,
    time_limit_s: float | None = None,
) -> dict[str, bool | int | float]:
    options: dict[str, bool | int | float] = {
        "disp": logging.getLogger("day_run").isEnabledFor(logging.DEBUG),
    }
    resolved_time_limit_s = time_limit_s
    if resolved_time_limit_s is None:
        resolved_time_limit_s = K_MILP_SOLVER_CONFIG.time_limit_s
    if resolved_time_limit_s is not None:
        options["time_limit"] = float(resolved_time_limit_s)
    if K_MILP_SOLVER_CONFIG.node_limit is not None:
        options["node_limit"] = int(K_MILP_SOLVER_CONFIG.node_limit)
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)
        return options
    if K_MILP_SOLVER_CONFIG.rel_gap is not None:
        options["mip_rel_gap"] = float(K_MILP_SOLVER_CONFIG.rel_gap)
    return options


def log_pattern_model_summary(
    *,
    attempt_name: str,
    patterns: tuple[OfdmaSlotPattern, ...],
    model_size: MilpModelSize,
    build_elapsed_s: float,
) -> None:
    LOGGER.info(
        build_console_message(
            level_tag="INFO",
            scope=current_scope(),
            stage="pattern",
            event="built",
            fields=[
                ("attempt", str(attempt_name)),
                ("patterns", str(int(len(patterns)))),
                ("vars", str(int(model_size.variable_count))),
                ("constraints", str(int(model_size.constraint_count))),
                ("nnz", str(int(model_size.nonzero_count))),
                ("elapsed_s", f"{float(build_elapsed_s):.3f}"),
            ],
        )
    )


def log_pattern_solve_summary(
    *,
    attempt_name: str,
    status: int,
    success: bool,
    objective_j: float | None,
    objective_bound: float | None,
    mip_gap: float | None,
    solve_elapsed_s: float,
) -> None:
    LOGGER.info(
        build_console_message(
            level_tag="INFO",
            scope=current_scope(),
            stage="pattern",
            event="solved",
            fields=[
                ("attempt", str(attempt_name)),
                ("status", str(int(status))),
                ("success", str(bool(success))),
                ("objective_j", "None" if objective_j is None else f"{float(objective_j):.12g}"),
                ("bound", "None" if objective_bound is None else f"{float(objective_bound):.12g}"),
                ("gap", "None" if mip_gap is None else f"{float(mip_gap):.12g}"),
                ("elapsed_s", f"{float(solve_elapsed_s):.3f}"),
            ],
        )
    )


def current_scope() -> str:
    return current_run_scope()


__all__ = [
    "build_and_solve_pattern_count_attempt",
    "build_infeasible_pattern_result",
    "build_pattern_result",
    "build_pruned_slot_patterns",
    "build_slot_patterns",
    "collapse_duplicate_slot_patterns",
    "solve_pattern_count_attempt",
]
