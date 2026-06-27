from __future__ import annotations

"""M-bounded candidate admission for the OFDMA pattern-count oracle."""

from dataclasses import dataclass, replace
import math

from .models import MilpCandidateRow, OfdmaMilpProblem


TOL = 1e-12
PAIR_OUTPUT_BALANCE_WEIGHT = 0.25
SMALL_UE_FULL_SERVICE_FRACTION = 1.0
SMALL_UE_HALF_SLOT_OUTPUT_ABS_TOL_W = 1.0
SMALL_UE_HALF_SLOT_OUTPUT_RATIO = 1.5


@dataclass(frozen=True)
class AdmissionStats:
    """Private diagnostic counts for one admitted OFDMA problem."""

    max_rows_per_user: int
    raw_rows_by_user: dict[int, int]
    admitted_rows_by_user: dict[int, int]


@dataclass(frozen=True)
class RoleCandidate:
    row: MilpCandidateRow
    priority: int


def build_admitted_ofdma_problem(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    max_rows_per_user: int,
) -> tuple[OfdmaMilpProblem, AdmissionStats]:
    """Return an admitted problem with at most M rows per UE per allowed PA family."""

    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    allowed_rows_by_user = {
        int(user_id): tuple(row for row in rows if int(row.pa_id) in allowed_pa_id_set)
        for user_id, rows in sorted(problem.candidate_rows_by_user.items())
    }
    admitted_by_user = {
        int(user_id): admit_user_rows_by_pa(
            rows=rows,
            demand_bits=float(problem.demand_bits_by_user[int(user_id)]),
            allowed_pa_ids=allowed_pa_ids,
            max_rows=int(max_rows_per_user),
            prb_max=int(problem.prb_max),
            n_tx_chains=int(problem.n_tx_chains),
            pa_catalog=problem.pa_catalog,
        )
        for user_id, rows in sorted(allowed_rows_by_user.items())
    }
    admitted_rows = rebuild_candidate_row_ids(admitted_by_user)
    admitted_rows_by_user = {
        int(user_id): tuple(row for row in admitted_rows if int(row.user_id) == int(user_id))
        for user_id in sorted(admitted_by_user)
    }
    admitted_problem = replace(
        problem,
        candidate_rows=admitted_rows,
        candidate_rows_by_user=admitted_rows_by_user,
    )
    return admitted_problem, AdmissionStats(
        max_rows_per_user=int(max_rows_per_user),
        raw_rows_by_user={
            int(user_id): int(len(rows))
            for user_id, rows in sorted(allowed_rows_by_user.items())
        },
        admitted_rows_by_user={
            int(user_id): int(len(rows))
            for user_id, rows in sorted(admitted_rows_by_user.items())
        },
    )


def admit_user_rows_by_pa(
    *,
    rows: tuple[MilpCandidateRow, ...],
    demand_bits: float,
    allowed_pa_ids: tuple[int, ...],
    max_rows: int,
    prb_max: int,
    n_tx_chains: int,
    pa_catalog: tuple,
) -> tuple[MilpCandidateRow, ...]:
    admitted = []
    for pa_id in allowed_pa_ids:
        pa_rows = tuple(row for row in rows if int(row.pa_id) == int(pa_id))
        p_out_cap_w = float(n_tx_chains) * float(pa_catalog[int(pa_id)].p_max_w)
        admitted.extend(
            admit_user_rows(
                rows=pa_rows,
                demand_bits=float(demand_bits),
                max_rows=int(max_rows),
                prb_max=int(prb_max),
                p_out_cap_w=float(p_out_cap_w),
            )
        )
    return tuple(sorted(admitted, key=row_base_rank))


def admit_user_rows(
    *,
    rows: tuple[MilpCandidateRow, ...],
    demand_bits: float,
    max_rows: int,
    prb_max: int,
    p_out_cap_w: float,
) -> tuple[MilpCandidateRow, ...]:
    if not rows:
        return ()

    if len(rows) <= int(max_rows):
        return tuple(sorted(rows, key=row_base_rank))

    selected = selected_role_candidates(
        rows,
        demand_bits=float(demand_bits),
        max_rows=int(max_rows),
        prb_max=int(prb_max),
        p_out_cap_w=float(p_out_cap_w),
    )
    selected_by_row_id: dict[int, RoleCandidate] = {}
    for candidate in selected:
        row_id = int(candidate.row.global_id)
        incumbent = selected_by_row_id.get(row_id)
        if incumbent is not None and incumbent.priority <= int(candidate.priority):
            continue
        selected_by_row_id[row_id] = candidate

    admitted = sorted(
        selected_by_row_id.values(),
        key=lambda candidate: admitted_role_rank(candidate, demand_bits=float(demand_bits)),
    )
    if len(admitted) < int(max_rows):
        admitted_ids = {int(candidate.row.global_id) for candidate in admitted}
        for row in sorted(
            rows,
            key=lambda row: fill_rank(
                row,
                demand_bits=float(demand_bits),
                prb_max=int(prb_max),
                p_out_cap_w=float(p_out_cap_w),
            ),
        ):
            if int(row.global_id) in admitted_ids:
                continue
            admitted.append(RoleCandidate(row=row, priority=90))
            admitted_ids.add(int(row.global_id))
            if len(admitted) >= int(max_rows):
                break

    return tuple(candidate.row for candidate in admitted[: int(max_rows)])


def selected_role_candidates(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    max_rows: int,
    prb_max: int,
    p_out_cap_w: float,
) -> tuple[RoleCandidate, ...]:
    if is_half_slot_efficient_small_demand_ue(rows, demand_bits=float(demand_bits), prb_max=int(prb_max)):
        return selected_small_demand_candidates(
            rows,
            demand_bits=float(demand_bits),
            max_rows=int(max_rows),
            prb_max=int(prb_max),
        )

    selected: list[RoleCandidate] = []
    for target_index, target_prb in enumerate(build_even_prb_targets(prb_max=int(prb_max), row_count=int(max_rows))):
        target_rows = ranked_pair_representative_rows(
            rows,
            demand_bits=float(demand_bits),
            target_prb=int(target_prb),
            prb_max=int(prb_max),
            p_out_cap_w=float(p_out_cap_w),
        )
        selected.extend(
            RoleCandidate(row=row, priority=int(target_index) + 10 + 100 * int(rank_index))
            for rank_index, row in enumerate(target_rows)
        )
    return tuple(selected)


def selected_small_demand_candidates(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    max_rows: int,
    prb_max: int,
) -> tuple[RoleCandidate, ...]:
    return tuple(
        RoleCandidate(
            row=lowest_output_row_at_demand_fraction(
                rows,
                demand_bits=float(demand_bits),
                demand_fraction_target=float(target),
                prb_max=int(prb_max),
            ),
            priority=int(target_index) + 10,
        )
        for target_index, target in enumerate(build_even_demand_fraction_targets(row_count=int(max_rows)))
    )


def build_even_prb_targets(*, prb_max: int, row_count: int) -> tuple[int, ...]:
    step = float(prb_max) / float(int(row_count) + 1)
    return tuple(int(math.floor(float(step) * float(index))) for index in range(1, int(row_count) + 1))


def build_even_demand_fraction_targets(*, row_count: int) -> tuple[float, ...]:
    return tuple(float(index) / float(row_count) for index in range(1, int(row_count) + 1))


def ranked_pair_representative_rows(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    target_prb: int,
    prb_max: int,
    p_out_cap_w: float,
) -> tuple[MilpCandidateRow, ...]:
    available_prb = nearest_available_prb_count(rows, target_prb=int(target_prb))
    target_rows = tuple(row for row in rows if int(row.n_prb) == int(available_prb))
    prb_fraction = float(available_prb) / max(float(prb_max), TOL)
    bounded_rows = tuple(
        row
        for row in target_rows
        if float(row.p_out_total_w) / max(float(p_out_cap_w), TOL) <= float(prb_fraction) + TOL
    )
    ranked_rows = bounded_rows if bounded_rows else target_rows
    return tuple(
        sorted(
            ranked_rows,
            key=lambda row: pair_candidate_score_rank(
                row,
                demand_bits=float(demand_bits),
                target_prb=int(target_prb),
                prb_max=int(prb_max),
                p_out_cap_w=float(p_out_cap_w),
            ),
        )
    )


def nearest_available_prb_count(rows: tuple[MilpCandidateRow, ...], *, target_prb: int) -> int:
    return min(
        (int(row.n_prb) for row in rows),
        key=lambda n_prb: (abs(int(n_prb) - int(target_prb)), int(n_prb)),
    )


def select_pair_representative_row(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    target_prb: int,
    prb_max: int,
    p_out_cap_w: float,
) -> MilpCandidateRow:
    return ranked_pair_representative_rows(
        rows,
        demand_bits=float(demand_bits),
        target_prb=int(target_prb),
        prb_max=int(prb_max),
        p_out_cap_w=float(p_out_cap_w),
    )[0]


def lowest_output_row_at_demand_fraction(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    demand_fraction_target: float,
    prb_max: int,
) -> MilpCandidateRow:
    required_bits = float(demand_bits) * float(demand_fraction_target)
    fitting_rows = tuple(row for row in rows if useful_bits(row, demand_bits=float(demand_bits)) + TOL >= required_bits)
    pairable_rows = tuple(row for row in fitting_rows if int(row.n_prb) <= int(pairable_prb_cap(prb_max=int(prb_max))))
    ranked_rows = pairable_rows if pairable_rows else fitting_rows
    return min(ranked_rows, key=lambda row: demand_milestone_rank(row, demand_bits=float(demand_bits)))


def pairable_prb_cap(*, prb_max: int) -> int:
    return int(prb_max) // 2


def full_service_rows(rows: tuple[MilpCandidateRow, ...], *, demand_bits: float) -> tuple[MilpCandidateRow, ...]:
    return tuple(row for row in rows if demand_fraction(row, demand_bits=float(demand_bits)) >= SMALL_UE_FULL_SERVICE_FRACTION - TOL)


def lowest_output_full_service_row(rows: tuple[MilpCandidateRow, ...], *, demand_bits: float) -> MilpCandidateRow | None:
    candidates = full_service_rows(rows, demand_bits=float(demand_bits))
    if not candidates:
        return None
    return min(candidates, key=lambda row: demand_milestone_rank(row, demand_bits=float(demand_bits)))


def is_half_slot_efficient_small_demand_ue(
    rows: tuple[MilpCandidateRow, ...],
    *,
    demand_bits: float,
    prb_max: int,
) -> bool:
    full_slot_best = lowest_output_full_service_row(rows, demand_bits=float(demand_bits))
    if full_slot_best is None:
        return False

    pairable_rows = tuple(row for row in rows if int(row.n_prb) <= pairable_prb_cap(prb_max=int(prb_max)))
    half_slot_best = lowest_output_full_service_row(pairable_rows, demand_bits=float(demand_bits))
    if half_slot_best is None:
        return False

    allowed_output_w = max(
        float(SMALL_UE_HALF_SLOT_OUTPUT_ABS_TOL_W),
        float(SMALL_UE_HALF_SLOT_OUTPUT_RATIO) * float(full_slot_best.p_out_total_w),
    )
    return float(half_slot_best.p_out_total_w) <= float(allowed_output_w) + TOL


def rebuild_candidate_row_ids(rows_by_user: dict[int, tuple[MilpCandidateRow, ...]]) -> tuple[MilpCandidateRow, ...]:
    rows: list[MilpCandidateRow] = []
    global_id = 0
    for user_id in sorted(rows_by_user):
        for local_row_id, row in enumerate(sorted(rows_by_user[int(user_id)], key=row_base_rank)):
            rows.append(
                replace(
                    row,
                    global_id=int(global_id),
                    local_row_id=int(local_row_id),
                )
            )
            global_id += 1
    return tuple(rows)


def useful_bits(row: MilpCandidateRow, *, demand_bits: float) -> float:
    return min(float(row.bits_per_slot), float(demand_bits))


def demand_fraction(row: MilpCandidateRow, *, demand_bits: float) -> float:
    return useful_bits(row, demand_bits=float(demand_bits)) / max(float(demand_bits), TOL)


def pair_candidate_score_rank(
    row: MilpCandidateRow,
    *,
    demand_bits: float,
    target_prb: int,
    prb_max: int,
    p_out_cap_w: float,
) -> tuple[float, float, float, float, int, int, int, int]:
    row_demand_fraction = demand_fraction(row, demand_bits=float(demand_bits))
    output_cap_fraction = float(row.p_out_total_w) / max(float(p_out_cap_w), TOL)
    prb_fraction = float(row.n_prb) / max(float(prb_max), TOL)
    balance_error = abs(float(output_cap_fraction) - float(prb_fraction))
    score = float(row_demand_fraction) - float(PAIR_OUTPUT_BALANCE_WEIGHT) * float(balance_error)
    return (
        -float(score),
        float(balance_error),
        -float(row_demand_fraction),
        float(output_cap_fraction),
        abs(int(row.n_prb) - int(target_prb)),
        int(row.n_prb),
        int(row.pa_id),
        int(row.local_row_id),
    )


def demand_milestone_rank(row: MilpCandidateRow, *, demand_bits: float) -> tuple[float, int, float, float, int, int]:
    return (
        float(row.p_out_total_w),
        int(row.n_prb),
        float(row.p_dc_active_w),
        -useful_bits(row, demand_bits=float(demand_bits)),
        int(row.pa_id),
        int(row.local_row_id),
    )


def admitted_role_rank(candidate: RoleCandidate, *, demand_bits: float) -> tuple[int, float, int, float, int, int]:
    row = candidate.row
    return (
        int(candidate.priority),
        -useful_bits(row, demand_bits=float(demand_bits)),
        int(row.n_prb),
        float(row.p_out_total_w),
        int(row.pa_id),
        int(row.local_row_id),
    )


def fill_rank(
    row: MilpCandidateRow,
    *,
    demand_bits: float,
    prb_max: int,
    p_out_cap_w: float,
) -> tuple[float, float, int, float, int, int]:
    resource_cost = float(row.n_prb) / max(float(prb_max), TOL) + float(row.p_out_total_w) / max(float(p_out_cap_w), TOL)
    return (
        -useful_bits(row, demand_bits=float(demand_bits)),
        float(resource_cost),
        int(row.n_prb),
        float(row.p_out_total_w),
        int(row.pa_id),
        int(row.local_row_id),
    )


def row_base_rank(row: MilpCandidateRow) -> tuple[int, int, int, float, float, float, int]:
    return (
        int(row.pa_id),
        int(row.n_prb),
        int(row.mcs),
        float(row.bits_per_slot),
        float(row.p_out_total_w),
        float(row.p_dc_active_w),
        int(row.local_row_id),
    )


__all__ = [
    "AdmissionStats",
    "build_admitted_ofdma_problem",
    "useful_bits",
]
