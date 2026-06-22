from __future__ import annotations

"""Build the shared public scheduler result for the round-robin baseline."""

from models import (
    MultiUserScheduleResult,
    SchedulerMode,
    SchedulerPowerSummary,
    SlotAllocation,
    SlotSchedule,
    UserScheduleSummary,
)

from .models import RoundRobinAttemptResult, RoundRobinCandidateRow, RoundRobinProblem
from .problem import candidate_rows_by_user_pa
from .round_robin import compute_slot_dc_power_w


TOL = 1e-9


def build_round_robin_result(
    problem: RoundRobinProblem,
    *,
    attempt: RoundRobinAttemptResult,
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    """Build a feasible shared scheduler result from one round-robin attempt."""

    slot_schedules = build_slot_schedules(problem, attempt=attempt)
    delivered_bits_by_user = delivered_bits_from_slots(problem, slot_schedules)
    frame_energy_j = float(problem.t_slot_s) * float(sum(slot.dc_power_w for slot in slot_schedules))
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.ROUND_ROBIN,
        feasible=True,
        infeasible_reason=None,
        power_summary=SchedulerPowerSummary(
            frame_energy_j=float(frame_energy_j),
            average_frame_dc_power_w=float(frame_energy_j) / max(float(frame_duration_s), TOL),
            active_energy_j=float(frame_energy_j),
            inactive_energy_j=0.0,
            average_frame_rf_output_w=float(sum(slot.aggregate_p_out_w for slot in slot_schedules)) / max(int(problem.frame_n_slots), 1),
        ),
        user_summaries=build_user_summaries(problem, delivered_bits_by_user=delivered_bits_by_user),
        slot_schedules=slot_schedules,
        solver_details=build_solver_details(
            problem,
            attempt=attempt,
            hard_off_details=hard_off_details,
        ),
    )


def build_infeasible_round_robin_result(
    problem: RoundRobinProblem,
    *,
    attempts: tuple[RoundRobinAttemptResult, ...],
    hard_off_details: dict[str, object],
) -> MultiUserScheduleResult:
    """Build the shared infeasible result after all round-robin attempts fail."""

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
    )
    solver_details["attempts"] = [build_attempt_summary(attempt) for attempt in attempts]
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.ROUND_ROBIN,
        feasible=False,
        infeasible_reason="No feasible OFDMA rolling-quantum round-robin schedule was found for the prepared user spaces.",
        power_summary=SchedulerPowerSummary(
            frame_energy_j=0.0,
            average_frame_dc_power_w=0.0,
            active_energy_j=0.0,
            inactive_energy_j=0.0,
            average_frame_rf_output_w=0.0,
        ),
        user_summaries=build_user_summaries(problem, delivered_bits_by_user=delivered_bits_by_user),
        slot_schedules=slot_schedules,
        solver_details=solver_details,
    )


def build_slot_schedules(
    problem: RoundRobinProblem,
    *,
    attempt: RoundRobinAttemptResult,
) -> tuple[SlotSchedule, ...]:
    return tuple(
        build_slot_schedule(
            problem,
            slot_id=int(slot_id),
            slot_rows=tuple(slot_rows),
        )
        for slot_id, slot_rows in enumerate(attempt.slot_rows_by_slot)
    )


def build_slot_schedule(
    problem: RoundRobinProblem,
    *,
    slot_id: int,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
) -> SlotSchedule:
    if not slot_rows:
        return SlotSchedule(
            slot_index=int(slot_id),
            active=False,
            pa_id=None,
            used_prbs=0,
            aggregate_p_out_w=0.0,
            dc_power_w=0.0,
            allocations=(),
        )

    sorted_rows = tuple(sorted(slot_rows, key=lambda row: (int(row.user_id), int(row.local_row_id))))
    used_prbs = int(sum(int(row.n_prb) for row in sorted_rows))
    aggregate_p_out_w = float(sum(float(row.p_out_total_w) for row in sorted_rows))
    return SlotSchedule(
        slot_index=int(slot_id),
        active=True,
        pa_id=int(sorted_rows[0].pa_id),
        used_prbs=int(used_prbs),
        aggregate_p_out_w=float(aggregate_p_out_w),
        dc_power_w=float(compute_slot_dc_power_w(problem, sorted_rows)),
        allocations=tuple(slot_allocation_from_row(row) for row in sorted_rows),
    )


def slot_allocation_from_row(row: RoundRobinCandidateRow) -> SlotAllocation:
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


def delivered_bits_from_slots(
    problem: RoundRobinProblem,
    slot_schedules: tuple[SlotSchedule, ...],
) -> dict[int, float]:
    return {
        int(user_id): sum(
            float(allocation.bits_per_slot)
            for slot in slot_schedules
            for allocation in slot.allocations
            if int(allocation.user_id) == int(user_id)
        )
        for user_id in problem.required_rate_by_user
    }


def build_user_summaries(
    problem: RoundRobinProblem,
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


def build_solver_details(
    problem: RoundRobinProblem,
    *,
    attempt: RoundRobinAttemptResult,
    hard_off_details: dict[str, object],
) -> dict[str, object]:
    return {
        "algorithm": "ofdma_equal_prb_round_robin",
        "scheduler_mode": SchedulerMode.ROUND_ROBIN.value,
        "pa_policy": str(problem.switch_policy.value),
        "frame_n_slots": int(problem.frame_n_slots),
        "prb_max": int(problem.prb_max),
        "fair_prb_share": int(attempt.fair_prb_share),
        "selected_pa_id": attempt.selected_pa_id,
        "selected_prb_by_user": {
            int(user_id): int(row.n_prb)
            for user_id, row in sorted(attempt.selected_rows_by_user.items())
        },
        "selected_mcs_by_user": {
            int(user_id): int(row.mcs)
            for user_id, row in sorted(attempt.selected_rows_by_user.items())
        },
        "selected_layers_by_user": {
            int(user_id): int(row.layers)
            for user_id, row in sorted(attempt.selected_rows_by_user.items())
        },
        "active_slot_count": int(attempt.active_slot_count),
        "allocation_count": int(attempt.allocation_count),
        "round_robin_cycle_count": int(attempt.round_robin_cycle_count),
        "unsatisfied_user_ids": tuple(int(user_id) for user_id in attempt.unsatisfied_user_ids),
        "failure_reason": attempt.failure_reason,
        "candidate_rows_by_user_pa": candidate_rows_by_user_pa(problem),
        **hard_off_details,
    }


def build_attempt_summary(attempt: RoundRobinAttemptResult) -> dict[str, object]:
    return {
        "attempt_name": str(attempt.attempt_name),
        "allowed_pa_ids": tuple(int(pa_id) for pa_id in attempt.allowed_pa_ids),
        "success": bool(attempt.success),
        "fair_prb_share": int(attempt.fair_prb_share),
        "selected_pa_id": attempt.selected_pa_id,
        "active_slot_count": int(attempt.active_slot_count),
        "allocation_count": int(attempt.allocation_count),
        "frame_energy_j": attempt.frame_energy_j,
        "unsatisfied_user_ids": tuple(int(user_id) for user_id in attempt.unsatisfied_user_ids),
        "failure_reason": attempt.failure_reason,
    }


__all__ = [
    "build_infeasible_round_robin_result",
    "build_round_robin_result",
]
