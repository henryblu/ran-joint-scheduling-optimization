from __future__ import annotations

from collections import deque

from .models import PreparedJointOfdmaProblem
from .plan_types import _CoveragePlan


TOL = 1e-12


def select_bounded_user_plans(
    problem: PreparedJointOfdmaProblem,
    plans: tuple[_CoveragePlan, ...],
    *,
    max_plans: int,
) -> tuple[_CoveragePlan, ...]:
    deduplicated_plans = deduplicate_user_plans(plans)
    if len(deduplicated_plans) <= int(max_plans):
        return deduplicated_plans

    selected_plans = []
    selected_signatures = set()
    ranked_lists = [
        deque(sorted(deduplicated_plans, key=ranking))
        for ranking in build_user_plan_cap_rankings(problem)
    ]
    while len(selected_plans) < int(max_plans):
        added_this_round = False
        for ranked_plans in ranked_lists:
            if len(selected_plans) >= int(max_plans):
                break
            if add_next_ranked_plan(
                ranked_plans=ranked_plans,
                selected_plans=selected_plans,
                selected_signatures=selected_signatures,
            ):
                added_this_round = True
        if not added_this_round:
            break

    return tuple(sorted(selected_plans, key=lambda plan: plan.rank_key()))


def deduplicate_user_plans(
    plans: tuple[_CoveragePlan, ...],
) -> tuple[_CoveragePlan, ...]:
    deduplicated_plans = []
    seen_signatures = set()
    for plan in sorted(plans, key=lambda candidate_plan: candidate_plan.rank_key()):
        signature = plan.signature()
        if signature in seen_signatures:
            continue
        deduplicated_plans.append(plan)
        seen_signatures.add(signature)
    return tuple(deduplicated_plans)


def build_user_plan_cap_rankings(
    problem: PreparedJointOfdmaProblem,
):
    return (
        lambda plan: plan.rank_key(),
        lambda plan: (
            float(plan.total_exact_serial_energy_j),
            int(plan.n_slots),
            float(plan.overdelivery_bits),
            plan.signature(),
        ),
        lambda plan: (
            float(plan.total_exact_serial_energy_j) / max(float(plan.delivered_bits), TOL),
            int(plan.n_slots),
            plan.signature(),
        ),
        lambda plan: (
            plan_max_pa_power_w(problem, plan),
            float(plan.total_exact_serial_energy_j),
            int(plan.n_slots),
            plan.signature(),
        ),
        lambda plan: (
            int(plan.area_prb_slots),
            float(plan.total_exact_serial_energy_j),
            plan.signature(),
        ),
        lambda plan: (
            float(plan.total_p_out_w),
            int(plan.n_slots),
            plan.signature(),
        ),
    )


def add_next_ranked_plan(
    *,
    ranked_plans: deque[_CoveragePlan],
    selected_plans: list[_CoveragePlan],
    selected_signatures: set[tuple[tuple[int, int], ...]],
) -> bool:
    while ranked_plans and ranked_plans[0].signature() in selected_signatures:
        ranked_plans.popleft()
    if not ranked_plans:
        return False

    plan = ranked_plans.popleft()
    selected_plans.append(plan)
    selected_signatures.add(plan.signature())
    return True


def plan_max_pa_power_w(
    problem: PreparedJointOfdmaProblem,
    plan: _CoveragePlan,
) -> float:
    return max(
        float(problem.pa_catalog[int(pa_id)].p_max_w)
        for pa_id in plan.uses_pa_ids()
    )


__all__ = [
    "plan_max_pa_power_w",
    "select_bounded_user_plans",
]
