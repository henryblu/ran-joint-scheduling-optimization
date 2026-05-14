from __future__ import annotations

from collections import defaultdict, deque
import math

from .models import PreparedJointOfdmaProblem
from .packer import slot_dc_power_w, slot_pa_output_limit_w
from .plan_types import _CandidateView, _CoveragePlan, _UserCandidateRow, _UserPlanRowInstance
from .resource_bounds import _UserAreaLowerBound


TOL = 1e-12
MAX_CANDIDATES_PER_USER_GEAR = 100


def build_candidate_views(
    problem: PreparedJointOfdmaProblem,
) -> dict[int, tuple[_CandidateView, ...]]:
    user_candidates = {}
    for user_id in sorted(problem.user_slot_spaces):
        candidates = []
        user_slot_space = _apply_active_pa_dc_contract(problem, problem.user_slot_spaces[int(user_id)])
        sorted_rows = user_slot_space.sort_values(
            [
                "pa_id",
                "n_prb",
                "mcs",
                "layers",
                "bits_per_slot",
                "p_dc_active_w",
                "p_out_total_w",
            ],
            ascending=[True, True, True, True, True, True, True],
        )
        for candidate_id, row in enumerate(sorted_rows.itertuples(index=False)):
            if float(row.p_out_total_w) > slot_pa_output_limit_w(problem, pa_id=int(row.pa_id)) + TOL:
                raise ValueError("Prepared OFDMA slot rows must respect the selected PA output limit.")
            candidates.append(
                _CandidateView(
                    user_id=int(user_id),
                    candidate_id=int(candidate_id),
                    pa_id=int(row.pa_id),
                    n_prb=int(row.n_prb),
                    layers=int(row.layers),
                    mcs=int(row.mcs),
                    bits_per_slot=float(row.bits_per_slot),
                    p_dc_active_w=float(row.p_dc_active_w),
                    p_out_total_w=float(row.p_out_total_w),
                )
            )
        user_candidates[int(user_id)] = tuple(candidates)
    return user_candidates


def _apply_active_pa_dc_contract(
    problem: PreparedJointOfdmaProblem,
    candidate_table,
):
    """Recompute row active DC power from the single-row slot RF output."""

    if candidate_table.empty:
        return candidate_table.copy()

    corrected_table = candidate_table.copy()
    corrected_table["p_dc_active_w"] = [
        slot_dc_power_w(
            problem,
            pa_id=int(row.pa_id),
            aggregate_rf_output_w=float(row.p_out_total_w),
        )
        for row in corrected_table.itertuples(index=False)
    ]
    return corrected_table.reset_index(drop=True)


def prune_scenario_candidates(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    user_candidates: dict[int, tuple[_CandidateView, ...]],
) -> dict[int, tuple[_CandidateView, ...]]:
    pruned_candidates = {}
    for user_id, candidates in sorted(user_candidates.items()):
        candidates_by_pa = defaultdict(list)
        for candidate in candidates:
            candidates_by_pa[int(candidate.pa_id)].append(candidate)

        kept_candidates = []
        for pa_id in sorted(candidates_by_pa):
            non_dominated = [
                candidate
                for candidate in candidates_by_pa[int(pa_id)]
                if not any(
                    other.candidate_id != candidate.candidate_id and candidate_dominates(other, candidate)
                    for other in candidates_by_pa[int(pa_id)]
                )
            ]
            kept_candidates.extend(
                cap_candidate_diversity(
                    problem,
                    required_bits=float(required_bits_by_user.get(int(user_id), 0.0)),
                    candidates=tuple(sorted(non_dominated, key=lambda candidate: candidate.rank_key())),
                )
            )

        pruned_candidates[int(user_id)] = tuple(sorted(kept_candidates, key=lambda candidate: candidate.rank_key()))
    return pruned_candidates


def candidate_dominates(
    left: _CandidateView,
    right: _CandidateView,
) -> bool:
    if int(left.pa_id) != int(right.pa_id):
        return False
    if float(left.bits_per_slot) + TOL < float(right.bits_per_slot):
        return False
    if int(left.n_prb) > int(right.n_prb):
        return False
    if float(left.p_out_total_w) > float(right.p_out_total_w) + TOL:
        return False
    return (
        float(left.bits_per_slot) > float(right.bits_per_slot) + TOL
        or int(left.n_prb) < int(right.n_prb)
        or float(left.p_out_total_w) + TOL < float(right.p_out_total_w)
    )


def cap_candidate_diversity(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
    candidates: tuple[_CandidateView, ...],
) -> tuple[_CandidateView, ...]:
    if len(candidates) <= MAX_CANDIDATES_PER_USER_GEAR:
        return candidates

    ranking_functions = build_candidate_cap_rankings(problem, required_bits=float(required_bits))

    selected_candidates = []
    selected_ids = set()
    for candidate in build_slot_bucket_representatives(
        problem,
        required_bits=float(required_bits),
        candidates=candidates,
    ):
        if int(candidate.candidate_id) in selected_ids:
            continue
        selected_candidates.append(candidate)
        selected_ids.add(int(candidate.candidate_id))
        if len(selected_candidates) >= MAX_CANDIDATES_PER_USER_GEAR:
            return tuple(sorted(selected_candidates, key=lambda candidate: candidate.rank_key()))

    ranked_lists = [deque(sorted(candidates, key=ranking)) for ranking in ranking_functions]
    while len(selected_candidates) < MAX_CANDIDATES_PER_USER_GEAR:
        if not add_next_ranked_candidate(
            ranked_lists=ranked_lists,
            selected_candidates=selected_candidates,
            selected_ids=selected_ids,
        ):
            break

    if len(selected_candidates) >= MAX_CANDIDATES_PER_USER_GEAR:
        return tuple(sorted(selected_candidates, key=lambda candidate: candidate.rank_key()))

    for candidate in candidates:
        if int(candidate.candidate_id) in selected_ids:
            continue
        selected_candidates.append(candidate)
        selected_ids.add(int(candidate.candidate_id))
        if len(selected_candidates) >= MAX_CANDIDATES_PER_USER_GEAR:
            break
    return tuple(sorted(selected_candidates, key=lambda candidate: candidate.rank_key()))


def build_candidate_cap_rankings(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
):
    ranking_functions = [
        lambda candidate: (
            plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate),
            exact_single_slot_dc_w(problem, candidate),
            -float(candidate.bits_per_slot),
            candidate.rank_key(),
        ),
        lambda candidate: (-float(candidate.bits_per_slot), candidate.rank_key()),
        lambda candidate: (exact_single_slot_dc_w(problem, candidate), candidate.rank_key()),
        lambda candidate: (float(candidate.p_out_total_w), candidate.rank_key()),
        lambda candidate: (int(candidate.n_prb), candidate.rank_key()),
        lambda candidate: (exact_single_slot_dc_w(problem, candidate) / max(float(candidate.bits_per_slot), TOL), candidate.rank_key()),
        lambda candidate: (-float(candidate.bits_per_slot) / max(int(candidate.n_prb), 1), candidate.rank_key()),
    ]
    if float(required_bits) <= TOL:
        return ranking_functions

    ranking_functions.append(
        lambda candidate: (
            plan_overdelivery_bits(problem, required_bits=float(required_bits), candidate=candidate),
            exact_single_slot_dc_w(problem, candidate),
            int(candidate.n_prb),
            candidate.rank_key(),
        )
    )
    return ranking_functions


def build_slot_bucket_representatives(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
    candidates: tuple[_CandidateView, ...],
) -> tuple[_CandidateView, ...]:
    candidates_by_slot_count = defaultdict(list)
    for candidate in candidates:
        n_slots = plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate)
        if math.isinf(float(n_slots)):
            continue
        candidates_by_slot_count[int(n_slots)].append(candidate)

    representatives = []
    selected_ids = set()
    for n_slots in sorted(candidates_by_slot_count):
        bucket = tuple(candidates_by_slot_count[int(n_slots)])
        bucket_representatives = (
            min(bucket, key=lambda candidate: (exact_single_slot_dc_w(problem, candidate), candidate.rank_key())),
            min(bucket, key=lambda candidate: (float(candidate.p_out_total_w), candidate.rank_key())),
            min(bucket, key=lambda candidate: (int(candidate.n_prb), candidate.rank_key())),
            max(bucket, key=lambda candidate: (float(candidate.bits_per_slot), _reverse_rank_key(candidate))),
            min(bucket, key=lambda candidate: (exact_single_slot_dc_w(problem, candidate) / max(float(candidate.bits_per_slot), TOL), candidate.rank_key())),
        )
        for candidate in bucket_representatives:
            if int(candidate.candidate_id) in selected_ids:
                continue
            representatives.append(candidate)
            selected_ids.add(int(candidate.candidate_id))
    return tuple(representatives)


def _reverse_rank_key(
    candidate: _CandidateView,
) -> tuple:
    return tuple(-value if isinstance(value, (int, float)) else value for value in candidate.rank_key())


def add_next_ranked_candidate(
    *,
    ranked_lists: list[deque[_CandidateView]],
    selected_candidates: list[_CandidateView],
    selected_ids: set[int],
) -> bool:
    for ranked_candidates in ranked_lists:
        while ranked_candidates and int(ranked_candidates[0].candidate_id) in selected_ids:
            ranked_candidates.popleft()
        if not ranked_candidates:
            continue
        candidate = ranked_candidates.popleft()
        selected_candidates.append(candidate)
        selected_ids.add(int(candidate.candidate_id))
        return True
    return False


def plan_n_slots(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
    candidate: _CandidateView,
) -> float:
    if float(required_bits) <= TOL:
        return 0.0

    n_slots = int(math.ceil(float(required_bits) / float(candidate.bits_per_slot) - TOL))
    if n_slots < 1 or n_slots > int(problem.frame_n_slots):
        return float("inf")
    return float(n_slots)


def exact_single_slot_dc_w(
    problem: PreparedJointOfdmaProblem,
    candidate: _CandidateView,
) -> float:
    return slot_dc_power_w(
        problem,
        pa_id=int(candidate.pa_id),
        aggregate_rf_output_w=float(candidate.p_out_total_w),
    )


def plan_overdelivery_bits(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits: float,
    candidate: _CandidateView,
) -> float:
    if float(required_bits) <= TOL:
        return 0.0

    n_slots = plan_n_slots(problem, required_bits=float(required_bits), candidate=candidate)
    if math.isinf(float(n_slots)):
        return float("inf")
    return float(n_slots) * float(candidate.bits_per_slot) - float(required_bits)


def validate_problem_for_greedy_search(
    problem: PreparedJointOfdmaProblem,
    *,
    user_candidates: dict[int, tuple[_UserCandidateRow, ...]],
    required_bits_by_user: dict[int, float],
    area_lower_bounds: dict[int, _UserAreaLowerBound],
) -> str | None:
    for user_id in sorted(required_bits_by_user):
        if float(required_bits_by_user[int(user_id)]) <= TOL:
            continue
        rows = user_candidates.get(int(user_id), ())
        if not rows:
            return f"No feasible slot-normalized operating points were found for user {int(user_id)}."

    for user_id in sorted(required_bits_by_user):
        if float(required_bits_by_user[int(user_id)]) <= TOL:
            continue
        max_bits_per_slot = max(float(row.bits_per_slot) for row in user_candidates[int(user_id)])
        if float(max_bits_per_slot) * float(problem.frame_n_slots) + TOL < float(required_bits_by_user[int(user_id)]):
            return (
                f"User {int(user_id)} cannot meet the frame payload target even when using its "
                "highest-payload slot row in every available slot."
            )

    available_prb_area = int(problem.frame_n_slots) * int(problem.prb_max)
    required_prb_area = 0
    for user_id in sorted(required_bits_by_user):
        if float(required_bits_by_user[int(user_id)]) <= TOL:
            continue
        minimum_area = area_lower_bounds[int(user_id)].min_area_for_bits(float(required_bits_by_user[int(user_id)]))
        if minimum_area is None:
            return (
                f"User {int(user_id)} cannot meet the frame payload target within the optimistic OFDMA "
                "PRB-area lower bound."
            )
        required_prb_area += int(minimum_area)

    if int(required_prb_area) <= int(available_prb_area) + TOL:
        return None
    return (
        "The requested frame payloads exceed the optimistic OFDMA PRB-area lower bound: "
        f"required area = {float(required_prb_area):.3f} PRB-slots, "
        f"available area = {float(available_prb_area):.3f} PRB-slots."
    )
__all__ = [
    "MAX_CANDIDATES_PER_USER_GEAR",
    "build_candidate_views",
    "candidate_dominates",
    "exact_single_slot_dc_w",
    "plan_overdelivery_bits",
    "prune_scenario_candidates",
    "validate_problem_for_greedy_search",
]
