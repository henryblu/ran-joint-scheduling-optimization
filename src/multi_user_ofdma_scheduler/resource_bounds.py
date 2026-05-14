from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .greedy_search import _LocalInsertion, _MutableSlotState, _UserCandidateRow


TOL = 1e-12


@dataclass(frozen=True)
class _UserAreaLowerBound:
    max_bits_by_area: tuple[float, ...]

    def min_area_for_bits(self, required_bits: float) -> int | None:
        if float(required_bits) <= TOL:
            return 0
        area = bisect_left(self.max_bits_by_area, float(required_bits) - TOL)
        if area >= len(self.max_bits_by_area):
            return None
        return int(area)


@dataclass(frozen=True)
class _UserSlotCompatibility:
    """Compact existence frontier for user rows that can still fit one open slot."""

    pa_frontiers: dict[int, tuple[int, ...]]
    pa_prefix_min_rf_w: dict[int, tuple[float, ...]]

    def has_compatible_row(
        self,
        *,
        slot_pa_id: int | None,
        free_prbs: int,
        rf_headroom_w: float,
    ) -> bool:
        if int(free_prbs) <= 0:
            return False

        if slot_pa_id is None:
            return any(
                _frontier_supports_slot(
                    prb_frontier=prb_frontier,
                    prefix_min_rf_w=self.pa_prefix_min_rf_w[int(pa_id)],
                    free_prbs=int(free_prbs),
                    rf_headroom_w=float(rf_headroom_w),
                )
                for pa_id, prb_frontier in self.pa_frontiers.items()
            )

        prb_frontier = self.pa_frontiers.get(int(slot_pa_id))
        if prb_frontier is None:
            return False
        return _frontier_supports_slot(
            prb_frontier=prb_frontier,
            prefix_min_rf_w=self.pa_prefix_min_rf_w[int(slot_pa_id)],
            free_prbs=int(free_prbs),
            rf_headroom_w=float(rf_headroom_w),
        )


def build_user_area_lower_bounds(
    user_candidates: dict[int, tuple[_UserCandidateRow, ...]],
    *,
    max_area: int,
) -> dict[int, _UserAreaLowerBound]:
    """Build one unbounded integer DP per user over PRB-slot area."""

    area_limit = max(0, int(max_area))
    lower_bounds = {}
    for user_id, candidates in user_candidates.items():
        positive_candidates = tuple(
            candidate
            for candidate in candidates
            if int(candidate.n_prb) > 0 and float(candidate.bits_per_slot) > TOL
        )
        max_bits_by_area = [0.0] * (area_limit + 1)
        for area in range(1, area_limit + 1):
            max_bits_by_area[area] = max(
                [
                    float(max_bits_by_area[area - 1]),
                    *(
                        float(max_bits_by_area[area - int(candidate.n_prb)]) + float(candidate.bits_per_slot)
                        for candidate in positive_candidates
                        if int(candidate.n_prb) <= int(area)
                    ),
                ]
            )
        lower_bounds[int(user_id)] = _UserAreaLowerBound(max_bits_by_area=tuple(max_bits_by_area))
    return lower_bounds


def build_user_slot_compatibility(
    user_candidates: dict[int, tuple[_UserCandidateRow, ...]],
) -> dict[int, _UserSlotCompatibility]:
    """Build one compact slot-fit frontier per user and PA."""

    compatibility_by_user = {}
    for user_id, candidates in user_candidates.items():
        rows_by_pa = {}
        for candidate in candidates:
            if int(candidate.n_prb) <= 0 or float(candidate.bits_per_slot) <= TOL:
                continue
            rows_by_pa.setdefault(int(candidate.pa_id), []).append((int(candidate.n_prb), float(candidate.p_out_total_w)))

        pa_frontiers = {}
        pa_prefix_min_rf_w = {}
        for pa_id, rows in rows_by_pa.items():
            frontier_rows = _build_slot_compatibility_frontier(rows)
            pa_frontiers[int(pa_id)] = tuple(int(n_prb) for n_prb, _ in frontier_rows)
            pa_prefix_min_rf_w[int(pa_id)] = tuple(float(p_out_total_w) for _, p_out_total_w in frontier_rows)

        compatibility_by_user[int(user_id)] = _UserSlotCompatibility(
            pa_frontiers=pa_frontiers,
            pa_prefix_min_rf_w=pa_prefix_min_rf_w,
        )
    return compatibility_by_user


def evaluate_remaining_resources_for_user_after_insertion(
    *,
    insertion: _LocalInsertion,
    remaining_bits_after: float,
    current_user_area_demand: int,
    current_total_area_demand: int,
    area_lower_bound: _UserAreaLowerBound,
    user_slot_compatibility: _UserSlotCompatibility,
    slot_states: list[_MutableSlotState],
    current_global_remaining_supply: int,
    problem_prb_max: int,
    pa_max_output_w_by_id: dict[int, float],
) -> tuple[int, bool] | None:
    """Apply the optimistic PRB-area screen using only the inserted user's changed demand."""

    remaining_area_after = (
        0
        if float(remaining_bits_after) <= TOL
        else area_lower_bound.min_area_for_bits(float(remaining_bits_after))
    )
    if remaining_area_after is None:
        return None

    global_remaining_supply = int(current_global_remaining_supply) - int(insertion.candidate.n_prb)
    global_remaining_demand = int(current_total_area_demand) - int(current_user_area_demand) + int(remaining_area_after)
    if float(global_remaining_demand) > float(global_remaining_supply) + TOL:
        return None

    if int(remaining_area_after) == 0:
        return 0, True

    user_supply_after = estimate_user_supply_after_insertion(
        problem_prb_max=int(problem_prb_max),
        pa_max_output_w_by_id=pa_max_output_w_by_id,
        user_id=int(insertion.user_id),
        user_slot_compatibility=user_slot_compatibility,
        slot_states=slot_states,
        insertion=insertion,
    )
    return int(user_supply_after) - int(remaining_area_after), False


def estimate_user_supply_after_insertion(
    *,
    problem_prb_max: int,
    pa_max_output_w_by_id: dict[int, float],
    user_id: int,
    user_slot_compatibility: _UserSlotCompatibility,
    slot_states: list[_MutableSlotState],
    insertion: _LocalInsertion,
) -> int:
    """Count optimistic free PRBs in slots where the user still has a PA-compatible row."""

    user_supply_after = 0
    for slot_state in slot_states:
        free_prbs = int(problem_prb_max) - int(slot_state.used_prbs)
        aggregate_rf_output_w = float(slot_state.aggregate_rf_output_w)
        scheduled_user_ids = slot_state.scheduled_users
        slot_pa_id = slot_state.pa_id

        if int(slot_state.slot_id) == int(insertion.slot_id):
            free_prbs -= int(insertion.candidate.n_prb)
            aggregate_rf_output_w = float(insertion.new_aggregate_rf_output_w)
            scheduled_user_ids = set(slot_state.scheduled_users)
            scheduled_user_ids.add(int(insertion.user_id))
            slot_pa_id = int(insertion.candidate.pa_id)

        if int(free_prbs) <= 0 or int(user_id) in scheduled_user_ids:
            continue

        rf_headroom_w = _slot_rf_headroom(
            pa_max_output_w_by_id=pa_max_output_w_by_id,
            slot_pa_id=slot_pa_id,
            aggregate_rf_output_w=float(aggregate_rf_output_w),
        )
        if not user_slot_compatibility.has_compatible_row(
            slot_pa_id=None if slot_pa_id is None else int(slot_pa_id),
            free_prbs=int(free_prbs),
            rf_headroom_w=float(rf_headroom_w),
        ):
            continue
        user_supply_after += int(free_prbs)
    return int(user_supply_after)


def _build_slot_compatibility_frontier(rows: list[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Keep only PRB thresholds where the best reachable RF load improves."""

    rows = sorted(rows, key=lambda row: (int(row[0]), float(row[1])))
    frontier = []
    best_rf_w = float("inf")
    for n_prb, p_out_total_w in rows:
        if float(p_out_total_w) >= float(best_rf_w) - TOL:
            continue
        frontier.append((int(n_prb), float(p_out_total_w)))
        best_rf_w = float(p_out_total_w)
    return tuple(frontier)


def _frontier_supports_slot(
    *,
    prb_frontier: tuple[int, ...],
    prefix_min_rf_w: tuple[float, ...],
    free_prbs: int,
    rf_headroom_w: float,
) -> bool:
    """Return whether the frontier contains any row that fits the current PRB and RF budget."""

    if int(free_prbs) <= 0:
        return False
    limit = bisect_left(prb_frontier, int(free_prbs) + 1) - 1
    if int(limit) < 0:
        return False
    return float(prefix_min_rf_w[int(limit)]) <= float(rf_headroom_w) + TOL


def _slot_rf_headroom(
    *,
    pa_max_output_w_by_id: dict[int, float],
    slot_pa_id: int | None,
    aggregate_rf_output_w: float,
) -> float:
    """Return the RF output headroom remaining in one slot."""

    if slot_pa_id is None:
        return max((float(p_max_w) for p_max_w in pa_max_output_w_by_id.values()), default=0.0)
    return max(0.0, float(pa_max_output_w_by_id[int(slot_pa_id)]) - float(aggregate_rf_output_w))


__all__ = [
    "_UserAreaLowerBound",
    "_UserSlotCompatibility",
    "build_user_area_lower_bounds",
    "build_user_slot_compatibility",
    "evaluate_remaining_resources_for_user_after_insertion",
]
