from __future__ import annotations

from .models import PreparedJointOfdmaProblem
from .plan_cap import plan_max_pa_power_w
from .plan_types import _CoveragePlan
from .serial_planner import MAX_USER_GREEDY_PLANS


TOL = 1e-12
MAX_REFINEMENT_ITERATIONS_FACTOR = 2


def select_serial_baseline_plan_set(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    coverage_plans_by_user: dict[int, tuple[_CoveragePlan, ...]],
) -> dict[int, _CoveragePlan] | None:
    positive_users = [
        int(user_id)
        for user_id in sorted(required_bits_by_user)
        if float(required_bits_by_user[int(user_id)]) > TOL
    ]
    if not positive_users:
        return {}

    states: dict[int, dict[int, _CoveragePlan]] = {0: {}}
    for user_id in positive_users:
        states = expand_serial_baseline_states(
            problem,
            user_id=int(user_id),
            states=states,
            coverage_plans=coverage_plans_by_user[int(user_id)],
        )
        if not states:
            return None

    return min(states.values(), key=serial_baseline_rank)


def expand_serial_baseline_states(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    states: dict[int, dict[int, _CoveragePlan]],
    coverage_plans: tuple[_CoveragePlan, ...],
) -> dict[int, dict[int, _CoveragePlan]]:
    next_states = {}
    for used_slots, partial_plan_set in sorted(states.items()):
        for plan in coverage_plans[:MAX_USER_GREEDY_PLANS]:
            next_used_slots = int(used_slots) + int(plan.n_slots)
            if int(next_used_slots) > int(problem.frame_n_slots):
                continue
            expanded_plan_set = dict(partial_plan_set)
            expanded_plan_set[int(user_id)] = plan
            current_best = next_states.get(int(next_used_slots))
            if current_best is not None and serial_plan_set_rank(expanded_plan_set) >= serial_plan_set_rank(current_best):
                continue
            next_states[int(next_used_slots)] = expanded_plan_set
    return next_states


def serial_baseline_rank(
    plan_set: dict[int, _CoveragePlan],
) -> tuple[int, float, float, tuple[tuple[int, tuple[tuple[int, int], ...]], ...]]:
    return (
        int(sum(plan.n_slots for plan in plan_set.values())),
        float(sum(plan.total_exact_serial_energy_j for plan in plan_set.values())),
        float(sum(plan.overdelivery_bits for plan in plan_set.values())),
        plan_set_signature(plan_set),
    )


def serial_plan_set_rank(
    plan_set: dict[int, _CoveragePlan],
) -> tuple[float, float, tuple[tuple[int, tuple[tuple[int, int], ...]], ...]]:
    return (
        float(sum(plan.total_exact_serial_energy_j for plan in plan_set.values())),
        float(sum(plan.overdelivery_bits for plan in plan_set.values())),
        plan_set_signature(plan_set),
    )


def run_remaining_slack_refinement(
    problem: PreparedJointOfdmaProblem,
    *,
    coverage_plans_by_user: dict[int, tuple[_CoveragePlan, ...]],
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> dict[int, _CoveragePlan]:
    return run_slack_plan_replacement_pass(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=selected_plans_by_user,
        max_iterations=max(1, MAX_REFINEMENT_ITERATIONS_FACTOR * len(selected_plans_by_user)),
        alternative_filter=lambda current_plan, alternative: True,
    )


def run_pa_switch_with_slack(
    problem: PreparedJointOfdmaProblem,
    *,
    coverage_plans_by_user: dict[int, tuple[_CoveragePlan, ...]],
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> dict[int, _CoveragePlan]:
    return run_slack_plan_replacement_pass(
        problem,
        coverage_plans_by_user=coverage_plans_by_user,
        selected_plans_by_user=selected_plans_by_user,
        max_iterations=max(1, len(selected_plans_by_user)),
        alternative_filter=lambda current_plan, alternative: (
            plan_max_pa_power_w(problem, alternative) + TOL < plan_max_pa_power_w(problem, current_plan)
        ),
    )


def run_slack_plan_replacement_pass(
    problem: PreparedJointOfdmaProblem,
    *,
    coverage_plans_by_user: dict[int, tuple[_CoveragePlan, ...]],
    selected_plans_by_user: dict[int, _CoveragePlan],
    max_iterations: int,
    alternative_filter,
) -> dict[int, _CoveragePlan]:
    accepted_plans = dict(selected_plans_by_user)
    for _ in range(int(max_iterations)):
        replacement = find_best_slack_replacement(
            problem,
            coverage_plans_by_user=coverage_plans_by_user,
            accepted_plans=accepted_plans,
            alternative_filter=alternative_filter,
        )
        if replacement is None:
            return accepted_plans
        user_id, alternative = replacement
        accepted_plans[int(user_id)] = alternative
    return accepted_plans


def find_best_slack_replacement(
    problem: PreparedJointOfdmaProblem,
    *,
    coverage_plans_by_user: dict[int, tuple[_CoveragePlan, ...]],
    accepted_plans: dict[int, _CoveragePlan],
    alternative_filter,
) -> tuple[int, _CoveragePlan] | None:
    used_slots = int(sum(plan.n_slots for plan in accepted_plans.values()))
    replacements = []
    for user_id in sorted(accepted_plans):
        replacements.extend(
            admissible_slack_replacements(
                problem,
                user_id=int(user_id),
                used_slots=int(used_slots),
                current_plan=accepted_plans[int(user_id)],
                alternatives=coverage_plans_by_user[int(user_id)][:MAX_USER_GREEDY_PLANS],
                alternative_filter=alternative_filter,
            )
        )
    if not replacements:
        return None
    _, user_id, alternative = min(replacements, key=lambda replacement: replacement[0])
    return int(user_id), alternative


def admissible_slack_replacements(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    used_slots: int,
    current_plan: _CoveragePlan,
    alternatives: tuple[_CoveragePlan, ...],
    alternative_filter,
) -> list[tuple[tuple, int, _CoveragePlan]]:
    replacements = []
    for alternative in alternatives:
        if not replacement_is_admissible(
            problem,
            used_slots=int(used_slots),
            current_plan=current_plan,
            alternative=alternative,
            alternative_filter=alternative_filter,
        ):
            continue
        energy_reduction = float(current_plan.total_exact_serial_energy_j) - float(alternative.total_exact_serial_energy_j)
        replacements.append(
            (
                (
                    -float(energy_reduction),
                    int(alternative.n_slots) - int(current_plan.n_slots),
                    float(alternative.total_p_out_w),
                    int(user_id),
                    alternative.signature(),
                ),
                int(user_id),
                alternative,
            )
        )
    return replacements


def replacement_is_admissible(
    problem: PreparedJointOfdmaProblem,
    *,
    used_slots: int,
    current_plan: _CoveragePlan,
    alternative: _CoveragePlan,
    alternative_filter,
) -> bool:
    if alternative.signature() == current_plan.signature():
        return False
    if not alternative_filter(current_plan, alternative):
        return False

    next_used_slots = int(used_slots) - int(current_plan.n_slots) + int(alternative.n_slots)
    if int(next_used_slots) > int(problem.frame_n_slots):
        return False

    energy_reduction = float(current_plan.total_exact_serial_energy_j) - float(alternative.total_exact_serial_energy_j)
    return float(energy_reduction) > TOL


def plan_set_signature(
    plan_set: dict[int, _CoveragePlan],
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    return tuple(
        (int(user_id), plan.signature())
        for user_id, plan in sorted(plan_set.items())
    )


__all__ = [
    "run_pa_switch_with_slack",
    "run_remaining_slack_refinement",
    "select_serial_baseline_plan_set",
]
