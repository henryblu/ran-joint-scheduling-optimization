from __future__ import annotations

"""M-bounded candidate admission for the OFDMA pattern-count oracle."""

from dataclasses import dataclass, replace
import math

from .models import MilpCandidateRow, OfdmaMilpProblem


TOL = 1e-12
DEMAND_KNOT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


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
) -> tuple[MilpCandidateRow, ...]:
    admitted = []
    for pa_id in allowed_pa_ids:
        pa_rows = tuple(row for row in rows if int(row.pa_id) == int(pa_id))
        admitted.extend(
            admit_user_rows(
                rows=pa_rows,
                demand_bits=float(demand_bits),
                max_rows=int(max_rows),
            )
        )
    return tuple(sorted(admitted, key=row_base_rank))


def admit_user_rows(
    *,
    rows: tuple[MilpCandidateRow, ...],
    demand_bits: float,
    max_rows: int,
) -> tuple[MilpCandidateRow, ...]:
    if not rows:
        return ()

    if len(rows) <= int(max_rows):
        return tuple(sorted(rows, key=row_base_rank))

    selected = selected_role_candidates(rows, demand_bits=float(demand_bits), max_rows=int(max_rows))
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
        for row in sorted(rows, key=lambda row: fill_rank(row, demand_bits=float(demand_bits))):
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
) -> tuple[RoleCandidate, ...]:
    top_count = max(1, int(math.ceil(float(max_rows) / 4.0)))
    selected: list[RoleCandidate] = []
    selected.extend(
        RoleCandidate(row=row, priority=10)
        for row in sorted(rows, key=lambda row: bit_density_rank(row, demand_bits=float(demand_bits)))[:top_count]
    )
    selected.extend(
        RoleCandidate(row=row, priority=20)
        for row in sorted(rows, key=lambda row: rf_efficiency_rank(row, demand_bits=float(demand_bits)))[:top_count]
    )
    for fraction in DEMAND_KNOT_FRACTIONS:
        threshold = float(fraction) * float(demand_bits)
        threshold_rows = tuple(
            row
            for row in rows
            if useful_bits(row, demand_bits=float(demand_bits)) + TOL >= float(threshold)
        )
        if not threshold_rows:
            continue
        selected.append(RoleCandidate(row=min(threshold_rows, key=row_prb_rank), priority=30))
        selected.append(RoleCandidate(row=min(threshold_rows, key=row_rf_rank), priority=40))

    selected.append(RoleCandidate(row=min(rows, key=row_prb_rank), priority=50))
    selected.append(RoleCandidate(row=min(rows, key=row_rf_rank), priority=60))
    selected.append(
        RoleCandidate(
            row=max(rows, key=lambda row: (useful_bits(row, demand_bits=float(demand_bits)), -int(row.n_prb))),
            priority=70,
        )
    )
    return tuple(selected)


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


def bit_density_rank(row: MilpCandidateRow, *, demand_bits: float) -> tuple[float, int, float, int, int]:
    density = useful_bits(row, demand_bits=float(demand_bits)) / max(float(row.n_prb), TOL)
    return (-float(density), int(row.n_prb), float(row.p_out_total_w), int(row.pa_id), int(row.local_row_id))


def rf_efficiency_rank(row: MilpCandidateRow, *, demand_bits: float) -> tuple[float, float, int, int, int]:
    efficiency = useful_bits(row, demand_bits=float(demand_bits)) / max(float(row.p_out_total_w), TOL)
    return (-float(efficiency), float(row.p_out_total_w), int(row.n_prb), int(row.pa_id), int(row.local_row_id))


def row_prb_rank(row: MilpCandidateRow) -> tuple[int, float, int, int]:
    return (int(row.n_prb), float(row.p_out_total_w), int(row.pa_id), int(row.local_row_id))


def row_rf_rank(row: MilpCandidateRow) -> tuple[float, int, int, int]:
    return (float(row.p_out_total_w), int(row.n_prb), int(row.pa_id), int(row.local_row_id))


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


def fill_rank(row: MilpCandidateRow, *, demand_bits: float) -> tuple[float, int, float, int, int]:
    return (
        -useful_bits(row, demand_bits=float(demand_bits)),
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
