from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
import math

from .models import PreparedJointOfdmaProblem
from .plan_cap import select_bounded_user_plans
from .plan_builder import exact_single_slot_dc_w, plan_n_slots
from .plan_types import _CandidateView, _CoveragePlan, _UserPlanRowInstance


TOL = 1e-12
MAX_TAIL_CANDIDATES = 24
MAX_USER_GREEDY_PLANS = 48


def build_coverage_plans_by_user(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    user_candidates: dict[int, tuple[_CandidateView, ...]],
) -> tuple[dict[int, tuple[_CoveragePlan, ...]], str | None]:
    coverage_plans_by_user = {}
    for user_id in sorted(required_bits_by_user):
        required_bits = float(required_bits_by_user[int(user_id)])
        if required_bits <= TOL:
            coverage_plans_by_user[int(user_id)] = ()
            continue

        plans = build_user_greedy_plans(
            problem,
            user_id=int(user_id),
            required_bits=float(required_bits),
            candidates=user_candidates.get(int(user_id), ()),
        )
        if not plans:
            return coverage_plans_by_user, (
                f"User {int(user_id)} has no feasible frame coverage plan from the prepared OFDMA slot rows."
            )

        coverage_plans_by_user[int(user_id)] = tuple(sorted(plans, key=lambda plan: plan.rank_key()))
    return coverage_plans_by_user, None


def build_user_greedy_plans(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    required_bits: float,
    candidates: tuple[_CandidateView, ...],
) -> tuple[_CoveragePlan, ...]:
    valid_candidates = tuple(
        candidate
        for candidate in candidates
        if int(candidate.user_id) == int(user_id) and float(candidate.bits_per_slot) > TOL
    )
    if not valid_candidates:
        return ()

    plans = []
    for ordered_candidates in build_user_greedy_candidate_orders(
        problem,
        required_bits=float(required_bits),
        candidates=valid_candidates,
    ):
        plans.extend(
            build_greedy_prefix_tail_plans(
                problem,
                user_id=int(user_id),
                required_bits=float(required_bits),
                ordered_candidates=ordered_candidates,
            )
        )

    plans.extend(
        build_repeated_candidate_plans(
            problem,
            user_id=int(user_id),
            required_bits=float(required_bits),
            candidates=valid_candidates,
        )
    )
    return select_bounded_user_plans(problem, tuple(plans), max_plans=MAX_USER_GREEDY_PLANS)


def build_user_greedy_candidate_orders(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
    candidates: tuple[_CandidateView, ...],
) -> tuple[tuple[_CandidateView, ...], ...]:
    orders = [
        tuple(sorted(candidates, key=lambda candidate: greedy_fill_rank(problem, candidate))),
        tuple(sorted(candidates, key=lambda candidate: high_power_fill_rank(problem, candidate))),
        tuple(sorted(candidates, key=lambda candidate: dc_per_bit_fill_rank(problem, candidate))),
        tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate),
                    -float(candidate.bits_per_slot),
                    exact_single_slot_dc_w(problem, candidate),
                    candidate.rank_key(),
                ),
            )
        ),
    ]
    return deduplicate_candidate_orders(orders)


def greedy_fill_rank(
    problem: PreparedJointOfdmaProblem,
    candidate: _CandidateView,
) -> tuple[float, float, float, int, tuple]:
    return (
        -float(candidate.bits_per_slot),
        exact_single_slot_dc_w(problem, candidate),
        float(candidate.p_out_total_w),
        int(candidate.n_prb),
        candidate.rank_key(),
    )


def high_power_fill_rank(
    problem: PreparedJointOfdmaProblem,
    candidate: _CandidateView,
) -> tuple[float, float, float, tuple]:
    return (
        -float(problem.pa_catalog[int(candidate.pa_id)].p_max_w),
        -float(candidate.bits_per_slot),
        exact_single_slot_dc_w(problem, candidate),
        candidate.rank_key(),
    )


def dc_per_bit_fill_rank(
    problem: PreparedJointOfdmaProblem,
    candidate: _CandidateView,
) -> tuple[float, float, tuple]:
    return (
        exact_single_slot_dc_w(problem, candidate) / max(float(candidate.bits_per_slot), TOL),
        -float(candidate.bits_per_slot),
        candidate.rank_key(),
    )


def deduplicate_candidate_orders(
    orders: list[tuple[_CandidateView, ...]],
) -> tuple[tuple[_CandidateView, ...], ...]:
    deduplicated_orders = []
    seen_signatures = set()
    for order in orders:
        signature = tuple(int(candidate.candidate_id) for candidate in order)
        if signature in seen_signatures:
            continue
        deduplicated_orders.append(order)
        seen_signatures.add(signature)
    return tuple(deduplicated_orders)


def build_greedy_prefix_tail_plans(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    required_bits: float,
    ordered_candidates: tuple[_CandidateView, ...],
) -> tuple[_CoveragePlan, ...]:
    if not ordered_candidates:
        return ()

    prefix_candidate = ordered_candidates[0]
    maximum_prefix_count = min(
        int(problem.frame_n_slots),
        max(0, int(math.floor(float(required_bits) / float(prefix_candidate.bits_per_slot) + TOL))),
    )
    tail_candidates = select_tail_candidates(problem, ordered_candidates)
    plans = []
    for prefix_count in unique_prefix_counts(maximum_prefix_count):
        plan = build_prefix_tail_plan(
            problem,
            user_id=int(user_id),
            required_bits=float(required_bits),
            prefix_candidate=prefix_candidate,
            prefix_count=int(prefix_count),
            tail_candidates=tail_candidates,
        )
        if plan is not None:
            plans.append(plan)
    return tuple(plans)


def unique_prefix_counts(
    maximum_prefix_count: int,
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            count
            for count in (
                int(maximum_prefix_count),
                int(maximum_prefix_count) - 1,
                int(maximum_prefix_count) - 2,
                0,
            )
            if count >= 0
        )
    )


def build_prefix_tail_plan(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    required_bits: float,
    prefix_candidate: _CandidateView,
    prefix_count: int,
    tail_candidates: tuple[_CandidateView, ...],
) -> _CoveragePlan | None:
    prefix_bits = int(prefix_count) * float(prefix_candidate.bits_per_slot)
    if prefix_bits + TOL >= float(required_bits):
        return make_plan_from_candidate_counts(
            problem,
            user_id=int(user_id),
            required_bits=float(required_bits),
            candidate_counts={prefix_candidate: int(prefix_count)},
        )

    tail_counts = find_best_tail_counts(
        problem,
        residual_bits=float(required_bits) - float(prefix_bits),
        max_tail_slots=min(3, int(problem.frame_n_slots) - int(prefix_count)),
        tail_candidates=tail_candidates,
    )
    if tail_counts is None:
        return None

    candidate_counts = {prefix_candidate: int(prefix_count)}
    for candidate, count in tail_counts.items():
        candidate_counts[candidate] = candidate_counts.get(candidate, 0) + int(count)
    return make_plan_from_candidate_counts(
        problem,
        user_id=int(user_id),
        required_bits=float(required_bits),
        candidate_counts=candidate_counts,
    )


def select_tail_candidates(
    problem: PreparedJointOfdmaProblem,
    candidates: tuple[_CandidateView, ...],
) -> tuple[_CandidateView, ...]:
    ranked_candidates = []
    selected_ids = set()
    rankings = [
        lambda candidate: greedy_fill_rank(problem, candidate),
        lambda candidate: (exact_single_slot_dc_w(problem, candidate), candidate.rank_key()),
        lambda candidate: (float(candidate.p_out_total_w), candidate.rank_key()),
        lambda candidate: dc_per_bit_fill_rank(problem, candidate),
        lambda candidate: (int(candidate.n_prb), candidate.rank_key()),
    ]
    for ranking in rankings:
        for candidate in sorted(candidates, key=ranking):
            if int(candidate.candidate_id) in selected_ids:
                continue
            ranked_candidates.append(candidate)
            selected_ids.add(int(candidate.candidate_id))
            break

    for candidate in candidates:
        if int(candidate.candidate_id) in selected_ids:
            continue
        ranked_candidates.append(candidate)
        selected_ids.add(int(candidate.candidate_id))
        if len(ranked_candidates) >= MAX_TAIL_CANDIDATES:
            break
    return tuple(ranked_candidates[:MAX_TAIL_CANDIDATES])


def find_best_tail_counts(
    problem: PreparedJointOfdmaProblem,
    *,
    residual_bits: float,
    max_tail_slots: int,
    tail_candidates: tuple[_CandidateView, ...],
) -> dict[_CandidateView, int] | None:
    if float(residual_bits) <= TOL:
        return {}
    if int(max_tail_slots) <= 0:
        return None

    best_tail = None
    for tail_slots in range(1, int(max_tail_slots) + 1):
        best_tail = best_tail_for_slot_count(
            problem,
            residual_bits=float(residual_bits),
            tail_slots=int(tail_slots),
            tail_candidates=tail_candidates,
            best_tail=best_tail,
        )
        if best_tail is not None:
            return best_tail[1]
    return None


def best_tail_for_slot_count(
    problem: PreparedJointOfdmaProblem,
    *,
    residual_bits: float,
    tail_slots: int,
    tail_candidates: tuple[_CandidateView, ...],
    best_tail,
):
    for candidate_tuple in combinations_with_replacement(tail_candidates, int(tail_slots)):
        delivered_bits = sum(float(candidate.bits_per_slot) for candidate in candidate_tuple)
        if float(delivered_bits) + TOL < float(residual_bits):
            continue
        candidate_counts = count_tail_candidates(candidate_tuple)
        rank = tail_counts_rank(problem, residual_bits=float(residual_bits), candidate_counts=candidate_counts)
        if best_tail is not None and rank >= best_tail[0]:
            continue
        best_tail = (rank, candidate_counts)
    return best_tail


def count_tail_candidates(
    candidate_tuple: tuple[_CandidateView, ...],
) -> dict[_CandidateView, int]:
    candidate_counts = defaultdict(int)
    for candidate in candidate_tuple:
        candidate_counts[candidate] += 1
    return dict(candidate_counts)


def tail_counts_rank(
    problem: PreparedJointOfdmaProblem,
    *,
    residual_bits: float,
    candidate_counts: dict[_CandidateView, int],
) -> tuple[int, float, float, tuple[tuple[int, int], ...]]:
    delivered_bits = sum(int(count) * float(candidate.bits_per_slot) for candidate, count in candidate_counts.items())
    exact_energy_w = sum(
        int(count) * exact_single_slot_dc_w(problem, candidate)
        for candidate, count in candidate_counts.items()
    )
    return (
        int(sum(candidate_counts.values())),
        float(delivered_bits) - float(residual_bits),
        float(exact_energy_w),
        tuple(
            (int(candidate.candidate_id), int(count))
            for candidate, count in sorted(candidate_counts.items(), key=lambda item: item[0].rank_key())
        ),
    )


def build_repeated_candidate_plans(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    required_bits: float,
    candidates: tuple[_CandidateView, ...],
) -> tuple[_CoveragePlan, ...]:
    plans = []
    for candidate in sorted(candidates, key=lambda candidate: repeated_candidate_rank(problem, required_bits, candidate)):
        n_slots = plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate)
        if math.isinf(float(n_slots)):
            continue
        plan = make_plan_from_candidate_counts(
            problem,
            user_id=int(user_id),
            required_bits=float(required_bits),
            candidate_counts={candidate: int(n_slots)},
        )
        if plan is not None:
            plans.append(plan)
    return tuple(plans)


def repeated_candidate_rank(
    problem: PreparedJointOfdmaProblem,
    required_bits: float,
    candidate: _CandidateView,
) -> tuple[float, float, float, tuple]:
    return (
        plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate),
        exact_single_slot_dc_w(problem, candidate),
        -float(candidate.bits_per_slot),
        candidate.rank_key(),
    )


def make_plan_from_candidate_counts(
    problem: PreparedJointOfdmaProblem,
    *,
    user_id: int,
    required_bits: float,
    candidate_counts: dict[_CandidateView, int],
) -> _CoveragePlan | None:
    cleaned_counts = {
        candidate: int(count)
        for candidate, count in candidate_counts.items()
        if int(count) > 0
    }
    if not cleaned_counts or sum(cleaned_counts.values()) > int(problem.frame_n_slots):
        return None

    row_instances = build_row_instances(problem, cleaned_counts)
    delivered_bits = sum(row_instance.delivered_bits for row_instance in row_instances)
    if float(delivered_bits) + TOL < float(required_bits):
        return None

    primary_candidate = choose_primary_plan_candidate(problem, row_instances)
    exact_serial_energy_j = float(problem.t_slot_s) * float(
        sum(int(row_instance.count) * float(row_instance.exact_single_slot_dc_w) for row_instance in row_instances)
    )
    return _CoveragePlan(
        user_id=int(user_id),
        pa_id=int(primary_candidate.pa_id),
        candidate=primary_candidate,
        n_slots=int(sum(row_instance.count for row_instance in row_instances)),
        delivered_bits=float(delivered_bits),
        overdelivery_bits=float(delivered_bits) - float(required_bits),
        area_prb_slots=int(sum(row_instance.total_prbs for row_instance in row_instances)),
        total_p_out_w=float(sum(row_instance.total_p_out_w for row_instance in row_instances)),
        exact_serial_energy_j=float(exact_serial_energy_j),
        row_instances=row_instances,
    )


def build_row_instances(
    problem: PreparedJointOfdmaProblem,
    candidate_counts: dict[_CandidateView, int],
) -> tuple[_UserPlanRowInstance, ...]:
    return tuple(
        _UserPlanRowInstance(
            candidate=candidate,
            count=int(count),
            exact_single_slot_dc_w=exact_single_slot_dc_w(problem, candidate),
        )
        for candidate, count in sorted(candidate_counts.items(), key=lambda item: item[0].rank_key())
    )


def choose_primary_plan_candidate(
    problem: PreparedJointOfdmaProblem,
    row_instances: tuple[_UserPlanRowInstance, ...],
) -> _CandidateView:
    return sorted(
        row_instances,
        key=lambda row_instance: (
            -int(row_instance.count),
            -float(row_instance.candidate.bits_per_slot),
            exact_single_slot_dc_w(problem, row_instance.candidate),
            row_instance.candidate.rank_key(),
        ),
    )[0].candidate


__all__ = [
    "MAX_USER_GREEDY_PLANS",
    "build_coverage_plans_by_user",
]
