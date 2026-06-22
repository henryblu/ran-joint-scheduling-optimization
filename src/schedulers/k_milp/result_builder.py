from __future__ import annotations

"""Convert OFDMA MILP oracle variables into the shared scheduler result."""

from configs.pa import pa_slot_dc_power
from models import (
    MultiUserScheduleResult,
    SchedulerMode,
    SchedulerPowerSummary,
    SlotAllocation,
    SlotSchedule,
    UserScheduleSummary,
)

from .logging import active_snapshot_index_from_scope
from .models import MilpAttemptResult, MilpCandidateRow, OfdmaMilpProblem
from .problem import candidate_rows_by_user_pa


TOL = 1e-9


def build_feasible_milp_result(
    problem: OfdmaMilpProblem,
    *,
    attempt: MilpAttemptResult,
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    """Build the shared public scheduler result from one feasible MILP attempt."""

    selected_rows_by_slot = decode_selected_rows_by_slot(problem, attempt=attempt)
    slot_schedules = tuple(
        build_slot_schedule(
            problem,
            slot_id=int(slot_id),
            selected_rows=tuple(selected_rows_by_slot[int(slot_id)]),
        )
        for slot_id in range(int(problem.frame_n_slots))
    )
    delivered_bits_by_user = {
        int(user_id): 0.0
        for user_id in problem.required_rate_by_user
    }
    for slot in slot_schedules:
        for allocation in slot.allocations:
            delivered_bits_by_user[int(allocation.user_id)] += float(allocation.bits_per_slot)

    frame_energy_j = float(problem.t_slot_s) * float(sum(slot.dc_power_w for slot in slot_schedules))
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    abs_error_j = abs(float(frame_energy_j) - float(attempt.objective_pwl_j or 0.0))
    solver_details = build_solver_details(
        problem,
        attempt=attempt,
        hard_off_details=hard_off_details,
        recomputed_frame_energy_j=float(frame_energy_j),
        abs_error_j=float(abs_error_j),
    )
    solver_details["active_slot_count"] = int(sum(slot.active for slot in slot_schedules))
    solver_details["allocation_count"] = int(sum(len(slot.allocations) for slot in slot_schedules))
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.K_MILP,
        feasible=True,
        infeasible_reason=None,
        power_summary=SchedulerPowerSummary(
            frame_energy_j=float(frame_energy_j),
            average_frame_dc_power_w=float(frame_energy_j) / max(float(frame_duration_s), TOL),
            active_energy_j=float(frame_energy_j),
            inactive_energy_j=0.0,
            average_frame_rf_output_w=float(sum(slot.aggregate_p_out_w for slot in slot_schedules))
            / max(int(problem.frame_n_slots), 1),
        ),
        user_summaries=build_user_summaries(problem, delivered_bits_by_user=delivered_bits_by_user),
        slot_schedules=slot_schedules,
        solver_details=solver_details,
    )


def build_infeasible_milp_result(
    problem: OfdmaMilpProblem,
    *,
    attempts: tuple[MilpAttemptResult, ...],
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    """Build the shared infeasible result after all MILP attempts fail."""

    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    slot_schedules = tuple(
        SlotSchedule(
            slot_index=int(slot_id),
            active=False,
            pa_id=None,
            used_prbs=0,
            aggregate_p_out_w=0.0,
            dc_power_w=0.0,
            allocations=(),
        )
        for slot_id in range(int(problem.frame_n_slots))
    )
    delivered_bits_by_user = {int(user_id): 0.0 for user_id in problem.required_rate_by_user}
    solver_details = build_solver_details(
        problem,
        attempt=attempts[-1],
        hard_off_details=hard_off_details,
        recomputed_frame_energy_j=0.0,
        abs_error_j=0.0,
    )
    solver_details["attempts"] = [
        {
            "attempt_name": str(attempt.attempt_name),
            "solver_status": int(attempt.solver_status),
            "success": bool(attempt.success),
            "message": str(attempt.solver_message),
        }
        for attempt in attempts
    ]
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.K_MILP,
        feasible=False,
        infeasible_reason="No feasible slot-indexed OFDMA MILP schedule was found for the prepared user spaces.",
        power_summary=SchedulerPowerSummary(
            frame_energy_j=0.0,
            average_frame_dc_power_w=0.0,
            active_energy_j=0.0,
            inactive_energy_j=0.0,
            average_frame_rf_output_w=0.0,
        ),
        user_summaries=build_user_summaries(problem, delivered_bits_by_user=delivered_bits_by_user),
        slot_schedules=slot_schedules,
        solver_details=solver_details | {"frame_duration_s": float(frame_duration_s)},
    )


def decode_selected_rows_by_slot(
    problem: OfdmaMilpProblem,
    *,
    attempt: MilpAttemptResult,
) -> dict[int, list[MilpCandidateRow]]:
    solution = attempt.solution
    row_by_id = {int(row.global_id): row for row in problem.candidate_rows}
    selected_rows_by_slot = {int(slot_id): [] for slot_id in range(int(problem.frame_n_slots))}
    for row in problem.candidate_rows:
        for slot_id in range(int(problem.frame_n_slots)):
            variable_id = attempt.variables.x[(int(row.global_id), int(slot_id))]
            if float(solution[int(variable_id)]) <= 0.5:
                continue
            selected_rows_by_slot[int(slot_id)].append(row_by_id[int(row.global_id)])
    return selected_rows_by_slot


def build_slot_schedule(
    problem: OfdmaMilpProblem,
    *,
    slot_id: int,
    selected_rows: tuple[MilpCandidateRow, ...],
) -> SlotSchedule:
    if not selected_rows:
        return SlotSchedule(
            slot_index=int(slot_id),
            active=False,
            pa_id=None,
            used_prbs=0,
            aggregate_p_out_w=0.0,
            dc_power_w=0.0,
            allocations=(),
        )

    sorted_rows = tuple(sorted(selected_rows, key=lambda row: (int(row.user_id), int(row.local_row_id))))
    pa_id = int(sorted_rows[0].pa_id)
    used_prbs = int(sum(int(row.n_prb) for row in sorted_rows))
    aggregate_p_out_w = float(sum(float(row.p_out_total_w) for row in sorted_rows))
    dc_power_w = pa_slot_dc_power(
        problem.pa_catalog[int(pa_id)],
        p_out_total_w=float(aggregate_p_out_w),
        n_tx_chains=int(problem.n_tx_chains),
        prb_fraction=float(used_prbs) / float(problem.prb_max),
    )
    return SlotSchedule(
        slot_index=int(slot_id),
        active=True,
        pa_id=int(pa_id),
        used_prbs=int(used_prbs),
        aggregate_p_out_w=float(aggregate_p_out_w),
        dc_power_w=float(dc_power_w),
        allocations=tuple(
            SlotAllocation(
                user_id=int(row.user_id),
                pa_id=int(row.pa_id),
                n_prb=int(row.n_prb),
                layers=int(row.layers),
                mcs=int(row.mcs),
                bits_per_slot=float(row.bits_per_slot),
                p_out_total_w=float(row.p_out_total_w),
                p_dc_active_w=float(row.p_dc_active_w),
            )
            for row in sorted_rows
        ),
    )


def build_user_summaries(
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
            satisfied=float(delivered_bits_by_user[int(user_id)]) + TOL
            >= float(problem.demand_bits_by_user[int(user_id)]),
        )
        for user_id in sorted(problem.required_rate_by_user)
    )


def build_solver_details(
    problem: OfdmaMilpProblem,
    *,
    attempt: MilpAttemptResult,
    hard_off_details: dict[str, object],
    recomputed_frame_energy_j: float,
    abs_error_j: float,
) -> dict[str, object]:
    return {
        "algorithm": "ofdma_slot_indexed_milp",
        "scheduler_mode": SchedulerMode.K_MILP.value,
        "is_oracle": True,
        "is_single_snapshot_mode": True,
        "solver_backend": "scipy.optimize.milp/highs",
        "solver_status": int(attempt.solver_status),
        "solver_message": str(attempt.solver_message),
        "success": bool(attempt.success),
        "objective_pwl_j": attempt.objective_pwl_j,
        "objective_bound": attempt.objective_bound,
        "mip_gap": attempt.mip_gap,
        "recomputed_frame_energy_j": float(recomputed_frame_energy_j),
        "pwl_recompute_abs_error_j": float(abs_error_j),
        "pwl_recompute_rel_error": float(abs_error_j) / max(abs(float(recomputed_frame_energy_j)), TOL),
        "build_elapsed_s": float(attempt.build_elapsed_s),
        "solve_elapsed_s": float(attempt.solve_elapsed_s),
        "variable_count": int(attempt.model_size.variable_count),
        "binary_variable_count": int(attempt.model_size.binary_variable_count),
        "continuous_variable_count": int(attempt.model_size.continuous_variable_count),
        "constraint_count": int(attempt.model_size.constraint_count),
        "nonzero_count": int(attempt.model_size.nonzero_count),
        "candidate_rows_by_user_pa": candidate_rows_by_user_pa(problem),
        "pwl_segments_by_pa": {
            int(pa_id): int(len(segments))
            for pa_id, segments in attempt.segments_by_pa.items()
        },
        "pa_policy": str(problem.switch_policy.value),
        "selected_snapshot_index": active_snapshot_index_from_scope(),
        **hard_off_details,
    }


__all__ = [
    "build_feasible_milp_result",
    "build_infeasible_milp_result",
]
