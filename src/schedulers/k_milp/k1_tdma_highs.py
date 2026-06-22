from __future__ import annotations

"""Compressed K=1 TDMA-plan HiGHS solver for the OFDMA MILP baseline."""

from dataclasses import dataclass
import logging
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from models import BatchUserParameterSpace, MultiUserScheduleResult, SchedulerMode, SchedulerPowerSummary, SlotAllocation, SlotSchedule
from run_reporting import build_console_message, current_run_scope
from schedulers.k_milp.tdma_plan_frontier import TdmaUserPlan, build_user_tdma_plan_frontier
from schedulers.k_milp.tdma_space import prepare_joint_schedule_problem

from .logging import active_snapshot_index_from_scope
from .models import MilpModelSize, OfdmaMilpProblem
from .pattern_count import build_pattern_milp_options, build_pattern_user_summaries
from .problem import candidate_rows_by_user_pa


TOL = 1e-9
LOGGER = logging.getLogger("snapshot_run")


@dataclass(frozen=True)
class K1TdmaHighsAttempt:
    """One compressed K1 TDMA-plan HiGHS attempt."""

    attempt_name: str
    allowed_pa_ids: tuple[int, ...]
    success: bool
    solver_status: int
    solver_message: str
    objective_j: float | None
    objective_bound: float | None
    mip_gap: float | None
    selected_plans: tuple[TdmaUserPlan, ...]
    plan_frontiers_by_user: dict[int, tuple[TdmaUserPlan, ...]]
    build_elapsed_s: float
    solve_elapsed_s: float
    model_size: MilpModelSize


def solve_k1_tdma_highs_attempt(
    problem: OfdmaMilpProblem,
    batch_space: BatchUserParameterSpace,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
) -> K1TdmaHighsAttempt:
    build_started_at = perf_counter()
    plan_frontiers_by_user = build_k1_plan_frontiers(
        problem,
        batch_space,
        allowed_pa_ids=allowed_pa_ids,
    )
    build_elapsed_s = float(perf_counter() - build_started_at)
    if not plan_frontiers_by_user or any(not plans for plans in plan_frontiers_by_user.values()):
        return build_infeasible_k1_attempt(
            attempt_name=attempt_name,
            allowed_pa_ids=allowed_pa_ids,
            plan_frontiers_by_user=plan_frontiers_by_user,
            build_elapsed_s=build_elapsed_s,
            message="No feasible K1 TDMA plan frontier was generated.",
        )

    c, constraint, plan_index = build_k1_highs_model(problem, plan_frontiers_by_user)
    model_size = MilpModelSize(
        variable_count=int(len(plan_index)),
        binary_variable_count=int(len(plan_index)),
        continuous_variable_count=0,
        constraint_count=int(constraint.A.shape[0]),
        nonzero_count=int(constraint.A.nnz),
    )
    log_k1_tdma_highs_model_summary(
        attempt_name=attempt_name,
        plan_count=len(plan_index),
        model_size=model_size,
        build_elapsed_s=build_elapsed_s,
    )
    solve_started_at = perf_counter()
    result = milp(
        c=c,
        integrality=np.ones(len(plan_index), dtype=int),
        bounds=Bounds(np.zeros(len(plan_index), dtype=float), np.ones(len(plan_index), dtype=float)),
        constraints=constraint,
        options=build_pattern_milp_options(),
    )
    solve_elapsed_s = float(perf_counter() - solve_started_at)
    selected_plans = decode_selected_k1_plans(result.x, plan_index) if result.success else ()
    log_k1_tdma_highs_solve_summary(
        attempt_name=attempt_name,
        status=int(result.status),
        success=bool(result.success),
        objective_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
        solve_elapsed_s=solve_elapsed_s,
    )
    return K1TdmaHighsAttempt(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=bool(result.success),
        solver_status=int(result.status),
        solver_message=str(result.message),
        objective_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
        selected_plans=selected_plans,
        plan_frontiers_by_user=plan_frontiers_by_user,
        build_elapsed_s=build_elapsed_s,
        solve_elapsed_s=solve_elapsed_s,
        model_size=model_size,
    )


def build_k1_plan_frontiers(
    problem: OfdmaMilpProblem,
    batch_space: BatchUserParameterSpace,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> dict[int, tuple[TdmaUserPlan, ...]]:
    tdma_problem = prepare_joint_schedule_problem(batch_space)
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    plan_frontiers_by_user = {}
    for user_row in problem.user_requirements.sort_values("user_id").itertuples(index=False):
        user_id = int(user_row.user_id)
        candidate_table = tdma_problem.user_candidate_spaces[int(user_id)]
        filtered_table = (
            candidate_table.loc[candidate_table["pa_id"].astype(int).isin(allowed_pa_id_set)]
            .copy()
            .reset_index(drop=True)
        )
        plan_frontiers_by_user[int(user_id)] = build_user_tdma_plan_frontier(
            filtered_table,
            required_bits=float(problem.demand_bits_by_user[int(user_id)]),
            frame_n_slots=int(problem.frame_n_slots),
        )
    return plan_frontiers_by_user


def build_k1_highs_model(
    problem: OfdmaMilpProblem,
    plan_frontiers_by_user: dict[int, tuple[TdmaUserPlan, ...]],
) -> tuple[np.ndarray, LinearConstraint, tuple[tuple[int, TdmaUserPlan], ...]]:
    plan_index = tuple(
        (int(user_id), plan)
        for user_id, plans in sorted(plan_frontiers_by_user.items())
        for plan in plans
    )
    c = np.asarray(
        [plan_energy_j(problem, plan) for _user_id, plan in plan_index],
        dtype=float,
    )
    rows = []
    cols = []
    values = []
    lower_bounds = []
    upper_bounds = []
    row_id = 0
    for user_id in sorted(plan_frontiers_by_user):
        for variable_index, (plan_user_id, _plan) in enumerate(plan_index):
            if int(plan_user_id) != int(user_id):
                continue
            rows.append(int(row_id))
            cols.append(int(variable_index))
            values.append(1.0)
        lower_bounds.append(1.0)
        upper_bounds.append(1.0)
        row_id += 1

    for variable_index, (_user_id, plan) in enumerate(plan_index):
        rows.append(int(row_id))
        cols.append(int(variable_index))
        values.append(float(plan.n_slots))
    lower_bounds.append(-np.inf)
    upper_bounds.append(float(problem.frame_n_slots))

    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower_bounds), len(plan_index))).tocsr()
    return c, LinearConstraint(matrix, np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float)), plan_index


def plan_energy_j(problem: OfdmaMilpProblem, plan: TdmaUserPlan) -> float:
    return float(problem.t_slot_s) * float(sum(row.p_dc_active_w for row in plan.slot_rows))


def decode_selected_k1_plans(
    solution,
    plan_index: tuple[tuple[int, TdmaUserPlan], ...],
) -> tuple[TdmaUserPlan, ...]:
    selected = [
        plan
        for variable_index, (_user_id, plan) in enumerate(plan_index)
        if int(round(float(solution[int(variable_index)]))) == 1
    ]
    return tuple(sorted(selected, key=lambda plan: int(plan.user_id)))


def build_k1_tdma_highs_result(
    problem: OfdmaMilpProblem,
    *,
    attempt: K1TdmaHighsAttempt,
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    slot_schedules = build_k1_slot_schedules(problem, selected_plans=attempt.selected_plans)
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
        solver_details=build_k1_tdma_highs_solver_details(problem, attempt=attempt, hard_off_details=hard_off_details),
    )


def build_k1_slot_schedules(
    problem: OfdmaMilpProblem,
    *,
    selected_plans: tuple[TdmaUserPlan, ...],
) -> tuple[SlotSchedule, ...]:
    slot_schedules = []
    slot_index = 0
    for plan in sorted(selected_plans, key=lambda selected_plan: int(selected_plan.user_id)):
        for row in plan.slot_rows:
            slot_schedules.append(
                SlotSchedule(
                    slot_index=int(slot_index),
                    active=True,
                    pa_id=int(row.pa_id),
                    used_prbs=int(row.n_prb),
                    aggregate_p_out_w=float(row.p_out_total_w),
                    dc_power_w=float(row.p_dc_active_w),
                    allocations=(slot_allocation_from_tdma_row(row),),
                )
            )
            slot_index += 1

    while slot_index < int(problem.frame_n_slots):
        slot_schedules.append(
            SlotSchedule(slot_index=slot_index, active=False, pa_id=None, used_prbs=0, aggregate_p_out_w=0.0, dc_power_w=0.0, allocations=())
        )
        slot_index += 1
    return tuple(slot_schedules)


def slot_allocation_from_tdma_row(row) -> SlotAllocation:
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


def build_k1_tdma_highs_solver_details(
    problem: OfdmaMilpProblem,
    *,
    attempt: K1TdmaHighsAttempt,
    hard_off_details: dict[str, object],
) -> dict[str, object]:
    active_slot_count = int(sum(plan.n_slots for plan in attempt.selected_plans))
    return {
        "algorithm": "ofdma_k1_tdma_highs_milp",
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
        "slot_pattern_count": int(sum(len(plans) for plans in attempt.plan_frontiers_by_user.values())),
        "active_slot_count": int(active_slot_count),
        "allocation_count": int(active_slot_count),
        "candidate_rows_by_user_pa": candidate_rows_by_user_pa(problem),
        "pa_policy": str(problem.switch_policy.value),
        "selected_snapshot_index": active_snapshot_index_from_scope(),
        **hard_off_details,
    }


def build_infeasible_k1_attempt(
    *,
    attempt_name: str,
    allowed_pa_ids: tuple[int, ...],
    plan_frontiers_by_user: dict[int, tuple[TdmaUserPlan, ...]],
    build_elapsed_s: float,
    message: str,
) -> K1TdmaHighsAttempt:
    return K1TdmaHighsAttempt(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=False,
        solver_status=2,
        solver_message=str(message),
        objective_j=None,
        objective_bound=None,
        mip_gap=None,
        selected_plans=(),
        plan_frontiers_by_user=plan_frontiers_by_user,
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


def get_optional_float(result, name: str) -> float | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    return float(value)


def log_k1_tdma_highs_model_summary(
    *,
    attempt_name: str,
    plan_count: int,
    model_size: MilpModelSize,
    build_elapsed_s: float,
) -> None:
    LOGGER.info(
        build_console_message(
            level_tag="INFO",
            scope=current_scope(),
            stage="pattern",
            event="k1_tdma_highs_built",
            fields=[
                ("attempt", str(attempt_name)),
                ("plans", str(int(plan_count))),
                ("vars", str(int(model_size.variable_count))),
                ("constraints", str(int(model_size.constraint_count))),
                ("nnz", str(int(model_size.nonzero_count))),
                ("elapsed_s", f"{float(build_elapsed_s):.3f}"),
            ],
        )
    )


def log_k1_tdma_highs_solve_summary(
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
            event="k1_tdma_highs_solved",
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
    "K1TdmaHighsAttempt",
    "build_k1_tdma_highs_result",
    "build_k1_plan_frontiers",
    "plan_energy_j",
    "solve_k1_tdma_highs_attempt",
]
