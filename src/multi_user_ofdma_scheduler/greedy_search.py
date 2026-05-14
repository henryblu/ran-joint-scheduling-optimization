from __future__ import annotations

from models import MultiUserScheduleResult, PASwitchPolicy, SchedulerMode, SchedulerPowerSummary, SlotSchedule, UserScheduleSummary

from .models import PreparedJointOfdmaProblem
from .packer import pack_serial_then_compact
from .plan_builder import (
    MAX_CANDIDATES_PER_USER_GEAR,
    build_candidate_views,
    prune_scenario_candidates,
    validate_problem_for_greedy_search,
)
from .resource_bounds import build_user_area_lower_bounds
from .serial_planner import build_coverage_plans_by_user
from .serial_selection import (
    run_pa_switch_with_slack,
    run_remaining_slack_refinement,
    select_serial_baseline_plan_set,
)


TOL = 1e-12


def run_pa_aware_greedy_ofdma_schedule(
    problem: PreparedJointOfdmaProblem,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Run the top-down QoS-first OFDMA heuristic and return the shared public slot result.

    Steps:
    1. Build policy-pruned candidate views from the trusted one-slot candidate tables.
    2. Build mixed-row serial user plans before any slot packing begins.
    3. Select a serially feasible baseline, then spend slack first on lower-power PA gears.
    4. Spend any remaining slot slack on lower-energy refinements.
    5. Build a serial slot placement first and only compact into OFDMA slots when exact energy improves.
    """

    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    required_rate_by_user, required_bits_by_user, delivered_bits_by_user, user_candidates = build_greedy_search_inputs(
        problem,
        frame_duration_s,
        switch_policy=switch_policy,
    )
    area_lower_bounds = build_user_area_lower_bounds(
        user_candidates,
        max_area=int(problem.frame_n_slots) * int(problem.prb_max),
    )
    infeasible_reason = validate_problem_for_greedy_search(
        problem,
        user_candidates=user_candidates,
        required_bits_by_user=required_bits_by_user,
        area_lower_bounds=area_lower_bounds,
    )
    packed_frame = None
    if infeasible_reason is None:
        packed_frame, infeasible_reason = build_serial_first_packed_frame(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=user_candidates,
        )

    if packed_frame is not None:
        delivered_bits_by_user = {
            int(user_id): float(packed_frame.delivered_bits_by_user.get(int(user_id), 0.0))
            for user_id in sorted(required_bits_by_user)
        }

    return build_schedule_result(
        problem,
        required_rate_by_user=required_rate_by_user,
        required_bits_by_user=required_bits_by_user,
        delivered_bits_by_user=delivered_bits_by_user,
        packed_frame=packed_frame,
        infeasible_reason=infeasible_reason,
        frame_duration_s=frame_duration_s,
    )


def build_serial_first_packed_frame(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user,
    user_candidates,
):
    coverage_plans_by_user, infeasible_reason = build_coverage_plans_by_user(
        problem,
        required_bits_by_user=required_bits_by_user,
        user_candidates=user_candidates,
    )
    if infeasible_reason is not None:
        return None, infeasible_reason

    selected_plans_by_user = select_serial_baseline_plan_set(
        problem,
        required_bits_by_user=required_bits_by_user,
        coverage_plans_by_user=coverage_plans_by_user,
    )
    if selected_plans_by_user is None:
        return None, "The bounded serial-first OFDMA planner could not find a feasible serial baseline."

    selected_plans_by_user = run_pa_switch_with_slack(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=selected_plans_by_user,
    )
    selected_plans_by_user = run_remaining_slack_refinement(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=selected_plans_by_user,
    )
    packed_frame = pack_serial_then_compact(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=selected_plans_by_user,
    )
    if packed_frame is None:
        raise ValueError("The selected OFDMA serial baseline could not be packed into serial slots.")
    return packed_frame, None


def build_greedy_search_inputs(
    problem: PreparedJointOfdmaProblem,
    frame_duration_s: float,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
):
    required_rate_by_user = {
        int(user_row.user_id): float(user_row.required_rate_bps)
        for user_row in problem.user_requirements.sort_values("user_id").itertuples(index=False)
    }
    required_bits_by_user = {
        int(user_id): float(required_rate_bps) * float(frame_duration_s)
        for user_id, required_rate_bps in required_rate_by_user.items()
    }
    delivered_bits_by_user = {
        int(user_id): 0.0
        for user_id in sorted(required_rate_by_user)
    }
    user_candidates = build_policy_pruned_candidate_space(
        problem,
        switch_policy=switch_policy,
        required_bits_by_user=required_bits_by_user,
        raw_user_candidates=build_candidate_views(problem),
    )
    return required_rate_by_user, required_bits_by_user, delivered_bits_by_user, user_candidates


def build_policy_pruned_candidate_space(
    problem: PreparedJointOfdmaProblem,
    *,
    switch_policy: PASwitchPolicy,
    required_bits_by_user: dict[int, float],
    raw_user_candidates,
):
    resolved_policy = switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy))
    if resolved_policy == PASwitchPolicy.DUAL_SWITCHABLE:
        return prune_scenario_candidates(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=raw_user_candidates,
        )
    if resolved_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        return prune_scenario_candidates(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=filter_candidates_by_pa_ids(
                raw_user_candidates,
                allowed_pa_ids=high_power_pa_ids(problem),
            ),
        )
    if resolved_policy == PASwitchPolicy.HARD_OFF:
        low_power_candidates = prune_scenario_candidates(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=filter_candidates_by_pa_ids(
                raw_user_candidates,
                allowed_pa_ids=low_power_pa_ids(problem),
            ),
        )
        if serial_baseline_exists(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=low_power_candidates,
        ):
            return low_power_candidates

        return prune_scenario_candidates(
            problem,
            required_bits_by_user=required_bits_by_user,
            user_candidates=filter_candidates_by_pa_ids(
                raw_user_candidates,
                allowed_pa_ids=high_power_pa_ids(problem),
            ),
        )

    raise ValueError(f"Unsupported OFDMA PA switch policy: {resolved_policy}")


def serial_baseline_exists(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    user_candidates,
) -> bool:
    coverage_plans_by_user, infeasible_reason = build_coverage_plans_by_user(
        problem,
        required_bits_by_user=required_bits_by_user,
        user_candidates=user_candidates,
    )
    if infeasible_reason is not None:
        return False
    return select_serial_baseline_plan_set(
        problem,
        required_bits_by_user=required_bits_by_user,
        coverage_plans_by_user=coverage_plans_by_user,
    ) is not None


def filter_candidates_by_pa_ids(
    user_candidates,
    *,
    allowed_pa_ids: tuple[int, ...],
):
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    return {
        int(user_id): tuple(
            candidate
            for candidate in candidates
            if int(candidate.pa_id) in allowed_pa_id_set
        )
        for user_id, candidates in sorted(user_candidates.items())
    }


def high_power_pa_ids(problem: PreparedJointOfdmaProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="8W PA", use_highest_power=True)


def low_power_pa_ids(problem: PreparedJointOfdmaProblem) -> tuple[int, ...]:
    return pa_ids_by_label_or_power(problem, label="4W PA", use_highest_power=False)


def pa_ids_by_label_or_power(
    problem: PreparedJointOfdmaProblem,
    *,
    label: str,
    use_highest_power: bool,
) -> tuple[int, ...]:
    labeled_pa_ids = tuple(
        int(pa_id)
        for pa_id, pa in enumerate(problem.pa_catalog)
        if str(pa.scenario_label) == str(label)
    )
    if labeled_pa_ids:
        return labeled_pa_ids
    if not problem.pa_catalog:
        return ()

    selected_power = (
        max(float(pa.p_max_w) for pa in problem.pa_catalog)
        if bool(use_highest_power)
        else min(float(pa.p_max_w) for pa in problem.pa_catalog)
    )
    return tuple(
        int(pa_id)
        for pa_id, pa in enumerate(problem.pa_catalog)
        if abs(float(pa.p_max_w) - float(selected_power)) <= TOL
    )


def build_schedule_result(
    problem: PreparedJointOfdmaProblem,
    *,
    required_rate_by_user: dict[int, float],
    required_bits_by_user: dict[int, float],
    delivered_bits_by_user: dict[int, float],
    packed_frame,
    infeasible_reason: str | None,
    frame_duration_s: float,
) -> MultiUserScheduleResult:
    if infeasible_reason is None and packed_frame is not None:
        slot_schedules = packed_frame.slot_schedules
        frame_energy_j = float(packed_frame.frame_energy_j)
        average_frame_dc_power_w = float(packed_frame.average_frame_dc_power_w)
        average_frame_rf_output_w = float(packed_frame.average_frame_rf_output_w)
    else:
        slot_schedules = tuple(
            SlotSchedule(
                slot_index=int(slot_index),
                active=False,
                pa_id=None,
                used_prbs=0,
                aggregate_p_out_w=0.0,
                dc_power_w=0.0,
                allocations=(),
            )
            for slot_index in range(int(problem.frame_n_slots))
        )
        frame_energy_j = 0.0
        average_frame_dc_power_w = 0.0
        average_frame_rf_output_w = 0.0
        delivered_bits_by_user = {
            int(user_id): 0.0
            for user_id in sorted(required_bits_by_user)
        }

    user_summaries = tuple(
        UserScheduleSummary(
            user_id=int(user_id),
            required_bits=float(required_bits_by_user[int(user_id)]),
            delivered_bits=float(delivered_bits_by_user.get(int(user_id), 0.0)),
            required_rate_bps=float(required_rate_by_user[int(user_id)]),
            delivered_rate_bps=float(delivered_bits_by_user.get(int(user_id), 0.0)) / max(float(frame_duration_s), TOL),
            satisfied=float(delivered_bits_by_user.get(int(user_id), 0.0)) + TOL >= float(required_bits_by_user[int(user_id)]),
        )
        for user_id in sorted(required_rate_by_user)
    )
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.OFDMA,
        feasible=infeasible_reason is None,
        infeasible_reason=infeasible_reason,
        power_summary=SchedulerPowerSummary(
            frame_energy_j=float(frame_energy_j),
            average_frame_dc_power_w=float(average_frame_dc_power_w),
            active_energy_j=float(frame_energy_j),
            inactive_energy_j=0.0,
            average_frame_rf_output_w=float(average_frame_rf_output_w),
        ),
        user_summaries=user_summaries,
        slot_schedules=slot_schedules,
        solver_details={"algorithm": "pa_aware_greedy_ofdma"},
    )


__all__ = ["run_pa_aware_greedy_ofdma_schedule"]
