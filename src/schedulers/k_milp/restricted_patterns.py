from __future__ import annotations

"""Restricted one-UE and dual-UE pattern generation for MILP diagnostics."""

from itertools import combinations, product

from .models import MilpCandidateRow, OfdmaMilpProblem, OfdmaSlotPattern
from .pattern_count import build_slot_pattern


def build_restricted_pair_slot_patterns(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> tuple[OfdmaSlotPattern, ...]:
    """Enumerate feasible one-UE and dual-UE slot patterns for pair-bounded MILP solves."""

    patterns: list[OfdmaSlotPattern] = []
    pattern_id = 0
    for pa_id in allowed_pa_ids:
        rows_by_user = build_rows_by_user_for_pa(problem, pa_id=int(pa_id))
        pattern_id = append_one_ue_patterns(
            problem,
            patterns=patterns,
            rows_by_user=rows_by_user,
            pa_id=int(pa_id),
            next_pattern_id=int(pattern_id),
        )
        pattern_id = append_dual_ue_patterns(
            problem,
            patterns=patterns,
            rows_by_user=rows_by_user,
            pa_id=int(pa_id),
            next_pattern_id=int(pattern_id),
        )
    return tuple(patterns)


def build_rows_by_user_for_pa(
    problem: OfdmaMilpProblem,
    *,
    pa_id: int,
) -> dict[int, tuple[MilpCandidateRow, ...]]:
    """Group candidate rows for one PA family by user."""

    return {
        int(user_id): tuple(
            row
            for row in problem.candidate_rows_by_user[int(user_id)]
            if int(row.pa_id) == int(pa_id)
        )
        for user_id in sorted(problem.candidate_rows_by_user)
    }


def append_one_ue_patterns(
    problem: OfdmaMilpProblem,
    *,
    patterns: list[OfdmaSlotPattern],
    rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
    pa_id: int,
    next_pattern_id: int,
) -> int:
    """Append every feasible one-user pattern and return the next pattern id."""

    pattern_id = int(next_pattern_id)
    for row in one_ue_rows(rows_by_user):
        pattern = build_slot_pattern(problem, pattern_id=pattern_id, pa_id=int(pa_id), rows=(row,))
        if pattern is None:
            continue
        patterns.append(pattern)
        pattern_id += 1
    return int(pattern_id)


def append_dual_ue_patterns(
    problem: OfdmaMilpProblem,
    *,
    patterns: list[OfdmaSlotPattern],
    rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
    pa_id: int,
    next_pattern_id: int,
) -> int:
    """Append every feasible two-user pattern and return the next pattern id."""

    pattern_id = int(next_pattern_id)
    dual_ue_patterns = []
    for left_row, right_row in dual_ue_row_pairs(rows_by_user):
        pattern = build_slot_pattern(
            problem,
            pattern_id=pattern_id,
            pa_id=int(pa_id),
            rows=(left_row, right_row),
        )
        if pattern is None:
            continue
        dual_ue_patterns.append(pattern)
    for pattern in prune_same_support_dominated_dual_ue_patterns(problem, tuple(dual_ue_patterns)):
        patterns.append(
            OfdmaSlotPattern(
                pattern_id=int(pattern_id),
                pa_id=pattern.pa_id,
                rows=pattern.rows,
                used_prbs=pattern.used_prbs,
                aggregate_p_out_w=pattern.aggregate_p_out_w,
                dc_power_w=pattern.dc_power_w,
                slot_energy_j=pattern.slot_energy_j,
                delivered_bits_by_user=pattern.delivered_bits_by_user,
            )
        )
        pattern_id += 1
    return int(pattern_id)


def one_ue_rows(
    rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
) -> tuple[MilpCandidateRow, ...]:
    """Flatten one-UE row choices in deterministic user order."""

    return tuple(
        row
        for user_id in sorted(rows_by_user)
        for row in rows_by_user[int(user_id)]
    )


def dual_ue_row_pairs(
    rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
) -> tuple[tuple[MilpCandidateRow, MilpCandidateRow], ...]:
    """Return all two-user row choices in deterministic user-pair order."""

    return tuple(
        row_pair
        for left_user_id, right_user_id in combinations(sorted(rows_by_user), 2)
        for row_pair in product(rows_by_user[int(left_user_id)], rows_by_user[int(right_user_id)])
    )


def prune_same_support_dominated_dual_ue_patterns(
    problem: OfdmaMilpProblem,
    patterns: tuple[OfdmaSlotPattern, ...],
) -> tuple[OfdmaSlotPattern, ...]:
    """Remove dominated dual-UE patterns only within the same UE-pair support."""

    kept_by_support: dict[tuple[int, int], list[OfdmaSlotPattern]] = {}
    for pattern in sorted(patterns, key=pattern_rank):
        support = tuple(sorted(int(row.user_id) for row in pattern.rows))
        incumbent_patterns = kept_by_support.setdefault(support, [])
        if any(pattern_dominates(problem, incumbent, pattern) for incumbent in incumbent_patterns):
            continue
        kept_by_support[support] = [
            incumbent
            for incumbent in incumbent_patterns
            if not pattern_dominates(problem, pattern, incumbent)
        ]
        kept_by_support[support].append(pattern)
    return tuple(
        pattern
        for support in sorted(kept_by_support)
        for pattern in sorted(kept_by_support[support], key=pattern_rank)
    )


def pattern_dominates(
    problem: OfdmaMilpProblem,
    candidate: OfdmaSlotPattern,
    incumbent: OfdmaSlotPattern,
) -> bool:
    candidate_support = tuple(sorted(candidate.delivered_bits_by_user))
    incumbent_support = tuple(sorted(incumbent.delivered_bits_by_user))
    if candidate_support != incumbent_support:
        return False

    useful_bits_not_lower = all(
        effective_pattern_bits(problem, candidate, user_id=int(user_id))
        + 1e-9
        >= effective_pattern_bits(problem, incumbent, user_id=int(user_id))
        for user_id in candidate_support
    )
    if not useful_bits_not_lower:
        return False
    if float(candidate.slot_energy_j) > float(incumbent.slot_energy_j) + 1e-12:
        return False
    return (
        any(
            effective_pattern_bits(problem, candidate, user_id=int(user_id))
            > effective_pattern_bits(problem, incumbent, user_id=int(user_id)) + 1e-9
            for user_id in candidate_support
        )
        or float(candidate.slot_energy_j) + 1e-12 < float(incumbent.slot_energy_j)
    )


def effective_pattern_bits(
    problem: OfdmaMilpProblem,
    pattern: OfdmaSlotPattern,
    *,
    user_id: int,
) -> float:
    return min(
        float(pattern.delivered_bits_by_user.get(int(user_id), 0.0)),
        float(problem.demand_bits_by_user[int(user_id)]),
    )


def pattern_rank(pattern: OfdmaSlotPattern) -> tuple[float, int, float, tuple[int, ...], tuple[int, ...]]:
    return (
        float(pattern.slot_energy_j),
        int(pattern.used_prbs),
        float(pattern.aggregate_p_out_w),
        tuple(sorted(int(row.user_id) for row in pattern.rows)),
        tuple(int(row.local_row_id) for row in pattern.rows),
    )


def count_one_ue_patterns(patterns: tuple[OfdmaSlotPattern, ...]) -> int:
    return sum(1 for pattern in patterns if len(pattern.rows) == 1)


def count_dual_ue_patterns(patterns: tuple[OfdmaSlotPattern, ...]) -> int:
    return sum(1 for pattern in patterns if len(pattern.rows) == 2)


__all__ = ["build_restricted_pair_slot_patterns"]
