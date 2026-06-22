from __future__ import annotations

"""TDMA-contained one-UE baseline plus M-dual OFDMA pattern generation."""

from dataclasses import dataclass
from itertools import combinations

from models import BatchUserParameterSpace
from schedulers.k_milp.tdma_plan_frontier import TdmaSlotRow, build_tdma_slot_rows, prune_dominated_tdma_slot_rows
from schedulers.k_milp.tdma_space import prepare_joint_schedule_problem

from .admission import AdmissionStats, build_admitted_ofdma_problem
from .models import MilpCandidateRow, OfdmaMilpProblem, OfdmaSlotPattern
from .restricted_patterns import append_dual_ue_patterns, build_rows_by_user_for_pa


@dataclass(frozen=True)
class ContainedPatternStats:
    """Private diagnostics for the TDMA-contained pattern family."""

    max_dual_rows_per_user: int
    one_ue_baseline_rows_by_user: dict[int, int]
    dual_raw_rows_by_user: dict[int, int]
    dual_admitted_rows_by_user: dict[int, int]
    raw_dual_ue_pair_bound: int
    one_ue_pattern_count: int
    valid_dual_ue_pattern_count: int


def build_tdma_contained_slot_patterns(
    problem: OfdmaMilpProblem,
    batch_space: BatchUserParameterSpace,
    *,
    allowed_pa_ids: tuple[int, ...],
    max_dual_rows_per_user: int,
) -> tuple[tuple[OfdmaSlotPattern, ...], ContainedPatternStats]:
    """Build one pattern family containing TDMA one-UE rows plus M-dual pair rows."""

    baseline_rows_by_user = build_one_ue_baseline_rows_by_user(
        batch_space,
        allowed_pa_ids=allowed_pa_ids,
    )
    one_ue_patterns = build_one_ue_baseline_patterns(
        problem,
        baseline_rows_by_user=baseline_rows_by_user,
    )
    admitted_problem, admission_stats = build_admitted_ofdma_problem(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        max_rows_per_user=int(max_dual_rows_per_user),
    )
    dual_ue_patterns = build_dual_ue_extension_patterns(
        problem,
        admitted_problem=admitted_problem,
        allowed_pa_ids=allowed_pa_ids,
        next_pattern_id=len(one_ue_patterns),
    )
    return (
        (*one_ue_patterns, *dual_ue_patterns),
        build_contained_pattern_stats(
            baseline_rows_by_user=baseline_rows_by_user,
            admission_stats=admission_stats,
            admitted_problem=admitted_problem,
            allowed_pa_ids=allowed_pa_ids,
            one_ue_patterns=one_ue_patterns,
            dual_ue_patterns=dual_ue_patterns,
        ),
    )


def build_one_ue_baseline_rows_by_user(
    batch_space: BatchUserParameterSpace,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> dict[int, tuple[MilpCandidateRow, ...]]:
    """Return TDMA-pruned one-UE rows converted to the OFDMA pattern row model."""

    tdma_problem = prepare_joint_schedule_problem(batch_space)
    allowed_pa_id_set = set(int(pa_id) for pa_id in allowed_pa_ids)
    rows_by_user = {}
    global_id = 0
    for user_id, slot_space in sorted(tdma_problem.user_candidate_spaces.items()):
        slot_rows = tuple(
            row for row in build_tdma_slot_rows(slot_space)
            if int(row.pa_id) in allowed_pa_id_set
        )
        tdma_rows = tuple(
            row
            for pa_id in allowed_pa_ids
            for row in prune_dominated_tdma_slot_rows(
                tuple(slot_row for slot_row in slot_rows if int(slot_row.pa_id) == int(pa_id))
            )
        )
        converted_rows = []
        for local_row_id, tdma_row in enumerate(sorted(tdma_rows, key=tdma_row_rank)):
            converted_rows.append(
                MilpCandidateRow(
                    global_id=int(global_id),
                    user_id=int(tdma_row.user_id),
                    local_row_id=int(local_row_id),
                    pa_id=int(tdma_row.pa_id),
                    n_prb=int(tdma_row.n_prb),
                    layers=int(tdma_row.layers),
                    mcs=int(tdma_row.mcs),
                    bits_per_slot=float(tdma_row.bits_per_slot),
                    p_out_total_w=float(tdma_row.p_out_total_w),
                    p_dc_active_w=float(tdma_row.p_dc_active_w),
                )
            )
            global_id += 1
        rows_by_user[int(user_id)] = tuple(converted_rows)
    return rows_by_user


def build_one_ue_baseline_patterns(
    problem: OfdmaMilpProblem,
    *,
    baseline_rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
) -> tuple[OfdmaSlotPattern, ...]:
    """Build one-UE baseline patterns using TDMA slot energy exactly."""

    patterns = []
    pattern_id = 0
    for user_id in sorted(baseline_rows_by_user):
        for row in baseline_rows_by_user[int(user_id)]:
            patterns.append(
                OfdmaSlotPattern(
                    pattern_id=int(pattern_id),
                    pa_id=int(row.pa_id),
                    rows=(row,),
                    used_prbs=int(row.n_prb),
                    aggregate_p_out_w=float(row.p_out_total_w),
                    dc_power_w=float(row.p_dc_active_w),
                    slot_energy_j=float(row.p_dc_active_w) * float(problem.t_slot_s),
                    delivered_bits_by_user={int(row.user_id): float(row.bits_per_slot)},
                )
            )
            pattern_id += 1
    return tuple(patterns)


def build_dual_ue_extension_patterns(
    problem: OfdmaMilpProblem,
    *,
    admitted_problem: OfdmaMilpProblem,
    allowed_pa_ids: tuple[int, ...],
    next_pattern_id: int,
) -> tuple[OfdmaSlotPattern, ...]:
    """Build dual-UE extension patterns from M-dual admitted OFDMA rows."""

    patterns: list[OfdmaSlotPattern] = []
    pattern_id = int(next_pattern_id)
    for pa_id in allowed_pa_ids:
        rows_by_user = build_rows_by_user_for_pa(admitted_problem, pa_id=int(pa_id))
        pattern_id = append_dual_ue_patterns(
            problem,
            patterns=patterns,
            rows_by_user=rows_by_user,
            pa_id=int(pa_id),
            next_pattern_id=int(pattern_id),
        )
    return tuple(patterns)


def build_contained_pattern_stats(
    *,
    baseline_rows_by_user: dict[int, tuple[MilpCandidateRow, ...]],
    admission_stats: AdmissionStats,
    admitted_problem: OfdmaMilpProblem,
    allowed_pa_ids: tuple[int, ...],
    one_ue_patterns: tuple[OfdmaSlotPattern, ...],
    dual_ue_patterns: tuple[OfdmaSlotPattern, ...],
) -> ContainedPatternStats:
    return ContainedPatternStats(
        max_dual_rows_per_user=int(admission_stats.max_rows_per_user),
        one_ue_baseline_rows_by_user={
            int(user_id): int(len(rows))
            for user_id, rows in sorted(baseline_rows_by_user.items())
        },
        dual_raw_rows_by_user={
            int(user_id): int(count)
            for user_id, count in sorted(admission_stats.raw_rows_by_user.items())
        },
        dual_admitted_rows_by_user={
            int(user_id): int(count)
            for user_id, count in sorted(admission_stats.admitted_rows_by_user.items())
        },
        raw_dual_ue_pair_bound=estimate_raw_dual_ue_pair_bound(admitted_problem, allowed_pa_ids=allowed_pa_ids),
        one_ue_pattern_count=int(len(one_ue_patterns)),
        valid_dual_ue_pattern_count=int(len(dual_ue_patterns)),
    )


def estimate_raw_dual_ue_pair_bound(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> int:
    total = 0
    user_ids = tuple(sorted(problem.candidate_rows_by_user))
    for pa_id in allowed_pa_ids:
        counts = {
            int(user_id): sum(1 for row in problem.candidate_rows_by_user[int(user_id)] if int(row.pa_id) == int(pa_id))
            for user_id in user_ids
        }
        for left_user_id, right_user_id in combinations(user_ids, 2):
            total += int(counts[int(left_user_id)]) * int(counts[int(right_user_id)])
    return int(total)


def tdma_row_rank(row: TdmaSlotRow) -> tuple[int, int, int, int, float, float, float]:
    return (
        int(row.pa_id),
        int(row.n_prb),
        int(row.layers),
        int(row.mcs),
        float(row.bits_per_slot),
        float(row.p_out_total_w),
        float(row.p_dc_active_w),
    )


__all__ = [
    "ContainedPatternStats",
    "build_one_ue_baseline_patterns",
    "build_one_ue_baseline_rows_by_user",
    "build_tdma_contained_slot_patterns",
    "estimate_raw_dual_ue_pair_bound",
]
