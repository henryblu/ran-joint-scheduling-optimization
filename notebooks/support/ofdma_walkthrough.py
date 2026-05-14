from __future__ import annotations

"""Notebook OFDMA walkthrough support layered on the production scheduler."""

from types import SimpleNamespace
from typing import Any

from multi_user_ofdma_scheduler.ofdma_space import prepare_joint_ofdma_problem
from multi_user_ofdma_scheduler.packer import (
    pack_selected_plans_serially,
    pack_serial_then_compact,
)
from multi_user_ofdma_scheduler.plan_builder import (
    build_candidate_views,
    prune_scenario_candidates,
    validate_problem_for_greedy_search,
)
from multi_user_ofdma_scheduler.resource_bounds import build_user_area_lower_bounds
from multi_user_ofdma_scheduler.serial_planner import build_coverage_plans_by_user
from multi_user_ofdma_scheduler.serial_selection import (
    run_pa_switch_with_slack,
    run_remaining_slack_refinement,
    select_serial_baseline_plan_set,
)


def build_ofdma_walkthrough_artifacts(batch_space: Any) -> SimpleNamespace:
    """Return the real OFDMA scheduler states used by the notebook.

    Steps:
    1. Prepare the trusted slot-level OFDMA problem and locally prune per-user rows.
    2. Build the minimum-slot serial baseline used before power refinement.
    3. Spend available slot slack on lower-power PA replacements.
    4. Spend remaining slack on lower-energy plan replacements.
    5. Pack the final selected plan set through the production serial-then-compact path.

    Packed-slot PA DC power is evaluated once from aggregate slot RF output;
    row-level active DC values are single-row diagnostics.
    """

    problem = prepare_joint_ofdma_problem(batch_space)
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    required_rate_by_user = {
        int(user_row.user_id): float(user_row.required_rate_bps)
        for user_row in problem.user_requirements.sort_values("user_id").itertuples(index=False)
    }
    required_bits_by_user = {
        int(user_id): float(required_rate_bps) * float(frame_duration_s)
        for user_id, required_rate_bps in required_rate_by_user.items()
    }
    raw_user_candidates = build_candidate_views(problem)
    user_candidates = prune_scenario_candidates(
        problem,
        required_bits_by_user=required_bits_by_user,
        user_candidates=raw_user_candidates,
    )
    infeasible_reason = validate_problem_for_greedy_search(
        problem,
        user_candidates=user_candidates,
        required_bits_by_user=required_bits_by_user,
        area_lower_bounds=build_user_area_lower_bounds(
            user_candidates,
            max_area=int(problem.frame_n_slots) * int(problem.prb_max),
        ),
    )
    if infeasible_reason is not None:
        raise ValueError(str(infeasible_reason))

    coverage_plans_by_user, infeasible_reason = build_coverage_plans_by_user(
        problem,
        required_bits_by_user=required_bits_by_user,
        user_candidates=user_candidates,
    )
    if infeasible_reason is not None:
        raise ValueError(str(infeasible_reason))

    baseline_plans = select_serial_baseline_plan_set(
        problem,
        required_bits_by_user=required_bits_by_user,
        coverage_plans_by_user=coverage_plans_by_user,
    )
    if baseline_plans is None:
        raise ValueError("The bounded serial-first OFDMA planner could not find a feasible serial baseline.")

    baseline_frame = _require_serial_frame(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=baseline_plans,
    )

    pa_switched_plans = run_pa_switch_with_slack(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=baseline_plans,
    )
    pa_switched_frame = _require_serial_frame(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=pa_switched_plans,
    )
    slack_refined_plans = run_remaining_slack_refinement(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=pa_switched_plans,
    )
    slack_refined_frame = _require_serial_frame(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=slack_refined_plans,
    )
    final_frame = pack_serial_then_compact(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=slack_refined_plans,
    )
    if final_frame is None:
        raise ValueError("The selected OFDMA plan set could not be packed into the frame.")

    return SimpleNamespace(
        problem=problem,
        frame_duration_s=float(frame_duration_s),
        required_rate_by_user=required_rate_by_user,
        required_bits_by_user=required_bits_by_user,
        raw_user_candidates=raw_user_candidates,
        pruned_user_candidates=user_candidates,
        coverage_plans_by_user=coverage_plans_by_user,
        baseline_plans=baseline_plans,
        pa_switched_plans=pa_switched_plans,
        slack_refined_plans=slack_refined_plans,
        baseline_frame=baseline_frame,
        pa_switched_frame=pa_switched_frame,
        slack_refined_frame=slack_refined_frame,
        final_frame=final_frame,
    )


def _require_serial_frame(
    problem,
    *,
    required_bits_by_user: dict[int, float],
    selected_plans_by_user: dict[int, Any],
):
    packed_frame = pack_selected_plans_serially(
        problem,
        required_bits_by_user=required_bits_by_user,
        selected_plans_by_user=selected_plans_by_user,
    )
    if packed_frame is None:
        raise ValueError("The serial OFDMA frame did not satisfy every user payload.")
    return packed_frame

__all__ = [
    "build_ofdma_walkthrough_artifacts",
]
