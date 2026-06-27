from __future__ import annotations

"""Pair-aware PRB split-template generation for restricted K2 OFDMA menus."""

from dataclasses import dataclass
from itertools import combinations, product
import math

from .models import MilpCandidateRow, OfdmaMilpProblem, OfdmaSlotPattern
from .pattern_count import build_slot_pattern
from .restricted_patterns import build_rows_by_user_for_pa, prune_same_support_dominated_dual_ue_patterns


TOL = 1e-12
SPLIT_TEMPLATE_FRACTIONS = (0.1, 0.3)


@dataclass(frozen=True)
class SplitTemplatePatternStats:
    """Diagnostics for the pair-aware split-template dual pattern family."""

    split_template_count: int
    max_rows_per_capacity: int
    raw_rows_by_user: dict[int, int]
    template_rows_by_user: dict[int, int]
    raw_dual_ue_pair_count: int
    feasible_dual_ue_pattern_count: int
    retained_dual_ue_pattern_count: int
    overflow_fallback_count: int


def build_prb_split_templates(
    *,
    prb_max: int,
    prb_step: int,
) -> tuple[tuple[int, int], ...]:
    """Return asymmetric PRB split templates that fit the usable PRB budget."""

    usable_quanta = int(prb_max) // int(prb_step)
    half_low = int(math.floor(float(usable_quanta) / 2.0))
    half_high = int(math.ceil(float(usable_quanta) / 2.0))
    left_quanta = [
        int(math.ceil(float(fraction) * float(usable_quanta)))
        for fraction in SPLIT_TEMPLATE_FRACTIONS
    ]
    left_quanta.extend([half_low, half_high])
    left_quanta.extend(
        int(usable_quanta) - int(math.ceil(float(fraction) * float(usable_quanta)))
        for fraction in reversed(SPLIT_TEMPLATE_FRACTIONS)
    )
    return tuple(
        (int(left) * int(prb_step), (int(usable_quanta) - int(left)) * int(prb_step))
        for left in left_quanta
        if 0 < int(left) < int(usable_quanta)
    )


def build_split_template_dual_ue_patterns(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    prb_step: int,
    next_pattern_id: int,
) -> tuple[tuple[OfdmaSlotPattern, ...], SplitTemplatePatternStats]:
    """Build pair-aware dual-UE patterns from fixed PRB split templates."""

    split_templates = build_prb_split_templates(
        prb_max=int(problem.prb_max),
        prb_step=int(prb_step),
    )
    retained_patterns: list[OfdmaSlotPattern] = []
    raw_dual_ue_pair_count = 0
    feasible_dual_patterns = []
    overflow_fallback_count = 0
    raw_rows_by_user: dict[int, int] = {}
    template_row_ids_by_user: dict[int, set[int]] = {
        int(user_id): set()
        for user_id in sorted(problem.candidate_rows_by_user)
    }

    for pa_id in allowed_pa_ids:
        rows_by_user = build_rows_by_user_for_pa(problem, pa_id=int(pa_id))
        for user_id, rows in rows_by_user.items():
            raw_rows_by_user[int(user_id)] = raw_rows_by_user.get(int(user_id), 0) + int(len(rows))

        for left_user_id, right_user_id in combinations(sorted(rows_by_user), 2):
            for left_capacity, right_capacity in split_templates:
                left_rows, left_overflows = select_capacity_representative_rows(
                    rows_by_user[int(left_user_id)],
                    demand_bits=float(problem.demand_bits_by_user[int(left_user_id)]),
                    prb_capacity=int(left_capacity),
                )
                right_rows, right_overflows = select_capacity_representative_rows(
                    rows_by_user[int(right_user_id)],
                    demand_bits=float(problem.demand_bits_by_user[int(right_user_id)]),
                    prb_capacity=int(right_capacity),
                )
                overflow_fallback_count += int(left_overflows) + int(right_overflows)
                template_row_ids_by_user[int(left_user_id)].update(int(row.global_id) for row in left_rows)
                template_row_ids_by_user[int(right_user_id)].update(int(row.global_id) for row in right_rows)

                for left_row, right_row in product(left_rows, right_rows):
                    raw_dual_ue_pair_count += 1
                    pattern = build_slot_pattern(
                        problem,
                        pattern_id=int(next_pattern_id),
                        pa_id=int(pa_id),
                        rows=(left_row, right_row),
                    )
                    if pattern is None:
                        continue
                    feasible_dual_patterns.append(pattern)

    pattern_id = int(next_pattern_id)
    for pattern in prune_same_support_dominated_dual_ue_patterns(problem, tuple(feasible_dual_patterns)):
        retained_patterns.append(
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

    stats = SplitTemplatePatternStats(
        split_template_count=int(len(split_templates)),
        max_rows_per_capacity=2,
        raw_rows_by_user=dict(sorted(raw_rows_by_user.items())),
        template_rows_by_user={
            int(user_id): int(len(row_ids))
            for user_id, row_ids in sorted(template_row_ids_by_user.items())
        },
        raw_dual_ue_pair_count=int(raw_dual_ue_pair_count),
        feasible_dual_ue_pattern_count=int(len(feasible_dual_patterns)),
        retained_dual_ue_pattern_count=int(len(retained_patterns)),
        overflow_fallback_count=int(overflow_fallback_count),
    )
    return tuple(retained_patterns), stats


def select_capacity_representative_rows(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    prb_capacity: int,
) -> tuple[tuple[MilpCandidateRow, ...], int]:
    """Select throughput and RF-efficient representatives under a PRB capacity."""

    if not rows:
        return (), 0

    fitting_rows = tuple(row for row in rows if int(row.n_prb) <= int(prb_capacity))
    overflow_used = 0
    if not fitting_rows:
        fitting_rows = (min(rows, key=overflow_row_rank),)
        overflow_used = 1

    selected = (
        min(fitting_rows, key=lambda row: throughput_representative_rank(row, demand_bits=float(demand_bits))),
        min(fitting_rows, key=lambda row: rf_efficient_representative_rank(row, demand_bits=float(demand_bits))),
    )
    selected_by_id = {}
    for row in selected:
        selected_by_id.setdefault(int(row.global_id), row)
    return tuple(selected_by_id.values()), int(overflow_used)


def useful_bits(row: MilpCandidateRow, *, demand_bits: float) -> float:
    return min(float(row.bits_per_slot), float(demand_bits))


def throughput_representative_rank(
    row: MilpCandidateRow,
    *,
    demand_bits: float,
) -> tuple[float, float, int, int]:
    return (
        -useful_bits(row, demand_bits=float(demand_bits)),
        float(row.p_out_total_w),
        int(row.n_prb),
        int(row.local_row_id),
    )


def rf_efficient_representative_rank(
    row: MilpCandidateRow,
    *,
    demand_bits: float,
) -> tuple[float, float, int, int]:
    efficiency = useful_bits(row, demand_bits=float(demand_bits)) / max(float(row.p_out_total_w), TOL)
    return (
        -float(efficiency),
        -useful_bits(row, demand_bits=float(demand_bits)),
        int(row.n_prb),
        int(row.local_row_id),
    )


def overflow_row_rank(row: MilpCandidateRow) -> tuple[int, float, int]:
    return (int(row.n_prb), float(row.p_out_total_w), int(row.local_row_id))


__all__ = [
    "SplitTemplatePatternStats",
    "build_prb_split_templates",
    "build_split_template_dual_ue_patterns",
    "select_capacity_representative_rows",
]
