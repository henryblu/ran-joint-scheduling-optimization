from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .tdma_models import USER_CANDIDATE_COLUMNS


TOL = 1e-12


@dataclass(frozen=True)
class TdmaSlotRow:
    """One feasible single-slot TDMA allocation row."""

    user_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    bits_per_slot: float
    p_dc_active_w: float
    p_out_total_w: float

    def rank_key(self) -> tuple[int, int, int, int, int, float, float, float]:
        return (
            int(self.user_id),
            int(self.pa_id),
            int(self.n_prb),
            int(self.mcs),
            int(self.layers),
            float(self.bits_per_slot),
            float(self.p_dc_active_w),
            float(self.p_out_total_w),
        )


@dataclass(frozen=True)
class TdmaUserPlan:
    """One per-user TDMA frame plan assembled from single-slot rows."""

    user_id: int
    n_slots: int
    delivered_bits: float
    schedule_cost: float
    slot_rows: tuple[TdmaSlotRow, ...]

    def delivered_rate_bps(self, frame_duration_s: float) -> float:
        return float(self.delivered_bits) / float(frame_duration_s)

    def rank_key(
        self,
    ) -> tuple[float, int, float, tuple[tuple[int, int, int, int, int, float, float, float], ...]]:
        return (
            float(self.schedule_cost),
            int(self.n_slots),
            -float(self.delivered_bits),
            tuple(row.rank_key() for row in self.slot_rows),
        )


@dataclass(frozen=True)
class _PlanCandidate:
    """Lightweight plan candidate used before final slot expansion."""

    n_slots: int
    delivered_bits: float
    schedule_cost: float
    row_counts: tuple[tuple[TdmaSlotRow, int], ...]


def build_user_tdma_plan_frontier(
    slot_space: pd.DataFrame,
    *,
    required_bits: float,
    frame_n_slots: int,
) -> tuple[TdmaUserPlan, ...]:
    """Build the exact mixed TDMA frontier over the bounded frame slot budget."""

    slot_rows = prune_dominated_tdma_slot_rows(build_tdma_slot_rows(slot_space))
    if not slot_rows:
        return ()

    candidates_by_slot_count = build_exact_mixed_plan_candidates(
        slot_rows=slot_rows,
        required_bits=float(required_bits),
        frame_n_slots=int(frame_n_slots),
    )
    return prune_user_plan_frontier(
        tuple(
            build_plan_from_candidate(user_id=int(slot_rows[0].user_id), candidate=candidate)
            for candidate in candidates_by_slot_count.values()
        )
    )


def build_tdma_slot_rows(slot_space: pd.DataFrame) -> tuple[TdmaSlotRow, ...]:
    return tuple(
        TdmaSlotRow(
            user_id=int(row.user_id),
            pa_id=int(row.pa_id),
            n_prb=int(row.n_prb),
            layers=int(row.layers),
            mcs=int(row.mcs),
            bits_per_slot=float(row.bits_per_slot),
            p_dc_active_w=float(row.p_dc_active_w),
            p_out_total_w=float(row.p_out_total_w),
        )
        for row in slot_space[USER_CANDIDATE_COLUMNS].itertuples(index=False)
        if float(row.bits_per_slot) > TOL
    )


def prune_dominated_tdma_slot_rows(slot_rows: tuple[TdmaSlotRow, ...]) -> tuple[TdmaSlotRow, ...]:
    ranked_rows = sorted(
        slot_rows,
        key=lambda row: (
            float(row.p_dc_active_w),
            -float(row.bits_per_slot),
            row.rank_key(),
        ),
    )
    kept_rows = []
    best_bits_per_slot = -1.0
    for row in ranked_rows:
        if float(row.bits_per_slot) <= float(best_bits_per_slot) + TOL:
            continue
        kept_rows.append(row)
        best_bits_per_slot = float(row.bits_per_slot)
    return tuple(kept_rows)


def build_exact_mixed_plan_candidates(
    *,
    slot_rows: tuple[TdmaSlotRow, ...],
    required_bits: float,
    frame_n_slots: int,
) -> dict[int, _PlanCandidate]:
    states_by_slot_count = {
        slot_count: ()
        for slot_count in range(int(frame_n_slots) + 1)
    }
    states_by_slot_count[0] = (
        _PlanCandidate(n_slots=0, delivered_bits=0.0, schedule_cost=0.0, row_counts=()),
    )

    for row in sorted(slot_rows, key=exact_plan_dp_row_order):
        for slot_count in range(1, int(frame_n_slots) + 1):
            expanded_states = tuple(
                add_slot_to_candidate(
                    candidate=previous_candidate,
                    row=row,
                    frame_n_slots=int(frame_n_slots),
                )
                for previous_candidate in states_by_slot_count[int(slot_count) - 1]
            )
            if not expanded_states:
                continue

            states_by_slot_count[int(slot_count)] = prune_partial_plan_candidates(
                (*states_by_slot_count[int(slot_count)], *expanded_states),
                required_bits=float(required_bits),
                frame_n_slots=int(frame_n_slots),
                max_extension_bits=float(row.bits_per_slot),
            )

    best_candidate_by_slot_count: dict[int, _PlanCandidate] = {}
    for slot_count, slot_states in states_by_slot_count.items():
        if int(slot_count) <= 0:
            continue

        feasible_candidate = select_best_feasible_plan_candidate(
            slot_states,
            required_bits=float(required_bits),
        )
        if feasible_candidate is None:
            continue

        best_candidate_by_slot_count[int(slot_count)] = feasible_candidate

    return best_candidate_by_slot_count


def add_slot_to_candidate(
    *,
    candidate: _PlanCandidate,
    row: TdmaSlotRow,
    frame_n_slots: int,
) -> _PlanCandidate:
    if candidate.row_counts and candidate.row_counts[-1][0] == row:
        previous_row, previous_count = candidate.row_counts[-1]
        row_counts = (
            *candidate.row_counts[:-1],
            (previous_row, int(previous_count) + 1),
        )
    else:
        row_counts = (*candidate.row_counts, (row, 1))

    return _PlanCandidate(
        n_slots=int(candidate.n_slots) + 1,
        delivered_bits=float(candidate.delivered_bits) + float(row.bits_per_slot),
        schedule_cost=(
            float(candidate.schedule_cost)
            + float(row.p_dc_active_w) / float(frame_n_slots)
        ),
        row_counts=row_counts,
    )


def prune_partial_plan_candidates(
    candidates: tuple[_PlanCandidate, ...],
    *,
    required_bits: float,
    frame_n_slots: int,
    max_extension_bits: float,
) -> tuple[_PlanCandidate, ...]:
    kept_candidates = []
    best_effective_bits = -1.0
    for candidate in sorted(
        candidates,
        key=lambda candidate: partial_candidate_rank(candidate, required_bits=float(required_bits)),
    ):
        if not partial_candidate_can_reach_required_bits(
            candidate,
            required_bits=float(required_bits),
            frame_n_slots=int(frame_n_slots),
            max_extension_bits=float(max_extension_bits),
        ):
            continue

        effective_bits = min(float(candidate.delivered_bits), float(required_bits))
        if float(effective_bits) <= float(best_effective_bits) + TOL:
            continue

        kept_candidates.append(candidate)
        best_effective_bits = float(effective_bits)

    return tuple(kept_candidates)


def select_best_feasible_plan_candidate(
    candidates: tuple[_PlanCandidate, ...],
    *,
    required_bits: float,
) -> _PlanCandidate | None:
    feasible_candidates = tuple(
        candidate
        for candidate in candidates
        if float(candidate.delivered_bits) + TOL >= float(required_bits)
    )
    if not feasible_candidates:
        return None

    return min(feasible_candidates, key=same_slot_candidate_rank)


def prune_user_plan_frontier(plans: tuple[TdmaUserPlan, ...]) -> tuple[TdmaUserPlan, ...]:
    kept_plans = []
    minimum_slot_count = 10**9
    for plan in sorted(plans, key=lambda candidate: candidate.rank_key()):
        if int(plan.n_slots) >= int(minimum_slot_count):
            continue
        kept_plans.append(plan)
        minimum_slot_count = int(plan.n_slots)
    return tuple(kept_plans)


def same_slot_candidate_rank(candidate: _PlanCandidate) -> tuple[float, float, tuple]:
    return (
        float(candidate.schedule_cost),
        -float(candidate.delivered_bits),
        tuple(
            (row.rank_key(), int(count))
            for row, count in sorted(
                candidate.row_counts,
                key=lambda item: item[0].rank_key(),
            )
        ),
    )


def partial_candidate_rank(candidate: _PlanCandidate, *, required_bits: float) -> tuple[float, float, float]:
    return (
        float(candidate.schedule_cost),
        -min(float(candidate.delivered_bits), float(required_bits)),
        -float(candidate.delivered_bits),
    )


def partial_candidate_can_reach_required_bits(
    candidate: _PlanCandidate,
    *,
    required_bits: float,
    frame_n_slots: int,
    max_extension_bits: float,
) -> bool:
    remaining_slots = int(frame_n_slots) - int(candidate.n_slots)
    reachable_bits = (
        float(candidate.delivered_bits)
        + float(remaining_slots) * float(max_extension_bits)
    )
    return float(reachable_bits) + TOL >= float(required_bits)


def exact_plan_dp_row_order(
    row: TdmaSlotRow,
) -> tuple[float, float, tuple[int, int, int, int, int, float, float, float]]:
    return (
        -float(row.bits_per_slot),
        float(row.p_dc_active_w),
        row.rank_key(),
    )


def build_plan_from_candidate(
    *,
    user_id: int,
    candidate: _PlanCandidate,
) -> TdmaUserPlan:
    slot_rows = tuple(
        row
        for row, count in sorted(candidate.row_counts, key=lambda item: item[0].rank_key())
        for _ in range(int(count))
    )
    return TdmaUserPlan(
        user_id=int(user_id),
        n_slots=int(len(slot_rows)),
        delivered_bits=float(sum(row.bits_per_slot for row in slot_rows)),
        schedule_cost=float(candidate.schedule_cost),
        slot_rows=slot_rows,
    )


__all__ = [
    "TdmaSlotRow",
    "TdmaUserPlan",
    "build_user_tdma_plan_frontier",
]
