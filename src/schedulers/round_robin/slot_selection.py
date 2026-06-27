from __future__ import annotations

"""Slot-local candidate selection for the OFDMA round-robin baseline."""

import math

from configs.pa import pa_slot_dc_power

from .models import RoundRobinCandidateRow, RoundRobinProblem


TOL = 1e-9
OUTPUT_BALANCE_WEIGHT = 0.25


def select_allocation_row(
    problem: RoundRobinProblem,
    *,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    allowed_pa_ids: tuple[int, ...],
    user_id: int,
    target_prbs: int,
    remaining_bits: float,
) -> RoundRobinCandidateRow | None:
    """Return the best replacement row under the user's rolling PRB and power targets."""

    if int(target_prbs) <= 0 or float(remaining_bits) <= TOL:
        return None

    rows = candidate_replacement_rows(
        problem,
        slot_rows=slot_rows,
        allowed_pa_ids=allowed_pa_ids,
        user_id=int(user_id),
        target_prbs=int(target_prbs),
        remaining_bits=float(remaining_bits),
    )
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: replacement_candidate_rank(
            problem,
            row,
            remaining_bits=float(remaining_bits),
        ),
    )


def candidate_replacement_rows(
    problem: RoundRobinProblem,
    *,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    allowed_pa_ids: tuple[int, ...],
    user_id: int,
    target_prbs: int,
    remaining_bits: float,
) -> tuple[RoundRobinCandidateRow, ...]:
    allowed_pa_id_set = slot_pa_id_set(slot_rows, allowed_pa_ids=allowed_pa_ids)
    prb_headroom = int(problem.prb_max) - sum(int(row.n_prb) for row in slot_rows)

    replacement_rows: list[RoundRobinCandidateRow] = []
    for row in problem.candidate_rows_by_user[int(user_id)]:
        if int(row.pa_id) not in allowed_pa_id_set:
            continue
        if int(row.n_prb) > int(target_prbs):
            continue
        replacement_row = useful_replacement_row(
            problem,
            row=row,
            n_prb=useful_replacement_prbs(
                problem,
                row,
                slot_rows=slot_rows,
                remaining_bits=float(remaining_bits),
                prb_headroom=int(prb_headroom),
            ),
        )
        if replacement_row is None:
            continue
        if not row_output_fits_prb_fraction(problem, replacement_row):
            continue
        if row_keeps_slot_schedulable(problem, slot_rows=slot_rows, row=replacement_row):
            replacement_rows.append(replacement_row)
    return tuple(replacement_rows)


def slot_pa_id_set(
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    *,
    allowed_pa_ids: tuple[int, ...],
) -> set[int]:
    if slot_rows:
        return {int(slot_rows[0].pa_id)}
    return {int(pa_id) for pa_id in allowed_pa_ids}


def useful_replacement_prbs(
    problem: RoundRobinProblem,
    row: RoundRobinCandidateRow,
    *,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    remaining_bits: float,
    prb_headroom: int,
) -> int:
    pa_output_headroom_w = available_slot_pa_output_headroom_w(
        problem,
        slot_rows=slot_rows,
        pa_id=int(row.pa_id),
    )
    return min(
        int(row.n_prb),
        int(prb_headroom),
        prbs_needed_for_bits(row, remaining_bits=float(remaining_bits)),
        prbs_that_fit_pa_output_headroom(row, pa_output_headroom_w=float(pa_output_headroom_w)),
    )


def useful_replacement_row(
    problem: RoundRobinProblem,
    *,
    row: RoundRobinCandidateRow,
    n_prb: int,
) -> RoundRobinCandidateRow | None:
    if int(n_prb) <= 0:
        return None
    if int(n_prb) >= int(row.n_prb):
        return row

    exact_row = matching_selected_shape_row(
        problem,
        selected_row=row,
        n_prb=int(n_prb),
    )
    if exact_row is not None:
        return exact_row
    return scaled_allocation_row(problem, selected_row=row, n_prb=int(n_prb))


def replacement_candidate_rank(
    problem: RoundRobinProblem,
    row: RoundRobinCandidateRow,
    *,
    remaining_bits: float,
) -> tuple[float, float, float, int, float, float, int, int, int, int]:
    useful_fraction = min(float(row.bits_per_slot), float(remaining_bits)) / max(float(remaining_bits), TOL)
    balance_error = row_output_balance_error(problem, row)
    score = float(useful_fraction) - float(OUTPUT_BALANCE_WEIGHT) * float(balance_error)
    return (
        -float(score),
        float(balance_error),
        -float(useful_fraction),
        int(row.n_prb),
        float(row.p_out_total_w),
        float(row.p_dc_active_w),
        -int(row.mcs),
        -int(row.layers),
        int(row.pa_id),
        int(row.local_row_id),
    )


def row_output_fits_prb_fraction(problem: RoundRobinProblem, row: RoundRobinCandidateRow) -> bool:
    return row_output_cap_fraction(problem, row) <= row_prb_fraction(problem, row) + TOL


def row_output_balance_error(problem: RoundRobinProblem, row: RoundRobinCandidateRow) -> float:
    return abs(float(row_output_cap_fraction(problem, row)) - float(row_prb_fraction(problem, row)))


def row_output_cap_fraction(problem: RoundRobinProblem, row: RoundRobinCandidateRow) -> float:
    pa_output_cap_w = float(problem.n_tx_chains) * float(problem.pa_catalog[int(row.pa_id)].p_max_w)
    return float(row.p_out_total_w) / max(float(pa_output_cap_w), TOL)


def row_prb_fraction(problem: RoundRobinProblem, row: RoundRobinCandidateRow) -> float:
    return float(row.n_prb) / max(float(problem.prb_max), TOL)


def prbs_needed_for_bits(row: RoundRobinCandidateRow, *, remaining_bits: float) -> int:
    bits_per_prb = float(row.bits_per_slot) / float(row.n_prb)
    return max(1, int(math.ceil(float(remaining_bits) / float(bits_per_prb) - TOL)))


def prbs_that_fit_pa_output_headroom(
    row: RoundRobinCandidateRow,
    *,
    pa_output_headroom_w: float,
) -> int:
    if float(row.p_out_total_w) <= TOL:
        return int(row.n_prb)
    return max(
        0,
        int(math.floor(float(pa_output_headroom_w) * float(row.n_prb) / float(row.p_out_total_w) + TOL)),
    )


def available_slot_pa_output_headroom_w(
    problem: RoundRobinProblem,
    *,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    pa_id: int,
) -> float:
    pa = problem.pa_catalog[int(pa_id)]
    aggregate_p_out_w = sum(float(row.p_out_total_w) for row in slot_rows)
    return float(problem.n_tx_chains) * float(pa.p_max_w) - float(aggregate_p_out_w)


def row_keeps_slot_schedulable(
    problem: RoundRobinProblem,
    *,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
    row: RoundRobinCandidateRow,
) -> bool:
    candidate_slot_rows = tuple(slot_rows) + (row,)
    if sum(int(slot_row.n_prb) for slot_row in candidate_slot_rows) > int(problem.prb_max):
        return False

    aggregate_p_out_w = sum(float(slot_row.p_out_total_w) for slot_row in candidate_slot_rows)
    pa = problem.pa_catalog[int(row.pa_id)]
    if float(aggregate_p_out_w) > float(problem.n_tx_chains) * float(pa.p_max_w) + TOL:
        return False

    if not bool(problem.use_psd_constraint):
        return True

    occupied_bandwidth_hz = (
        sum(int(slot_row.n_prb) for slot_row in candidate_slot_rows)
        * 12.0
        * float(problem.delta_f_hz)
    )
    aggregate_psd_w_per_hz = float(aggregate_p_out_w) / max(float(occupied_bandwidth_hz), TOL)
    return float(aggregate_psd_w_per_hz) <= float(problem.psd_max_w_per_hz) + TOL


def matching_selected_shape_row(
    problem: RoundRobinProblem,
    *,
    selected_row: RoundRobinCandidateRow,
    n_prb: int,
) -> RoundRobinCandidateRow | None:
    for row in problem.candidate_rows_by_user[int(selected_row.user_id)]:
        if (
            int(row.pa_id) == int(selected_row.pa_id)
            and int(row.layers) == int(selected_row.layers)
            and int(row.mcs) == int(selected_row.mcs)
            and int(row.n_prb) == int(n_prb)
        ):
            return row
    return None


def scaled_allocation_row(
    problem: RoundRobinProblem,
    *,
    selected_row: RoundRobinCandidateRow,
    n_prb: int,
) -> RoundRobinCandidateRow:
    scale = float(n_prb) / float(selected_row.n_prb)
    p_out_total_w = float(selected_row.p_out_total_w) * float(scale)
    p_dc_active_w = pa_slot_dc_power(
        problem.pa_catalog[int(selected_row.pa_id)],
        p_out_total_w=float(p_out_total_w),
        n_tx_chains=int(problem.n_tx_chains),
        prb_fraction=float(n_prb) / float(problem.prb_max),
    )
    return RoundRobinCandidateRow(
        global_id=int(selected_row.global_id),
        user_id=int(selected_row.user_id),
        local_row_id=int(selected_row.local_row_id),
        pa_id=int(selected_row.pa_id),
        n_prb=int(n_prb),
        layers=int(selected_row.layers),
        mcs=int(selected_row.mcs),
        bits_per_slot=float(selected_row.bits_per_slot) * float(scale),
        p_out_total_w=float(p_out_total_w),
        p_dc_active_w=float(p_dc_active_w),
    )


def compute_slot_dc_power_w(
    problem: RoundRobinProblem,
    slot_rows: tuple[RoundRobinCandidateRow, ...],
) -> float:
    if not slot_rows:
        return 0.0
    pa_id = int(slot_rows[0].pa_id)
    used_prbs = sum(int(row.n_prb) for row in slot_rows)
    aggregate_p_out_w = sum(float(row.p_out_total_w) for row in slot_rows)
    return pa_slot_dc_power(
        problem.pa_catalog[int(pa_id)],
        p_out_total_w=float(aggregate_p_out_w),
        n_tx_chains=int(problem.n_tx_chains),
        prb_fraction=float(used_prbs) / float(problem.prb_max),
    )


__all__ = [
    "compute_slot_dc_power_w",
    "select_allocation_row",
]
