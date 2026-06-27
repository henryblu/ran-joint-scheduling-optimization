from __future__ import annotations

"""Rolling-quantum allocation for the OFDMA round-robin baseline."""

from schedulers.feasibility_bounds import (
    InfeasibilityCertificate,
    log_feasibility_certificate,
    row_menu_certificate,
)

from .models import RoundRobinAttemptResult, RoundRobinCandidateRow, RoundRobinProblem
from .slot_selection import (
    compute_slot_dc_power_w,
    row_output_balance_error,
    row_output_fits_prb_fraction,
    select_allocation_row,
)


TOL = 1e-9


def run_round_robin_attempt(
    problem: RoundRobinProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
) -> RoundRobinAttemptResult:
    """Allocate candidate-backed UE rows in cyclic order with dynamic fair shares."""

    user_ids = tuple(sorted(problem.required_rate_by_user))
    fair_prb_share = int(problem.prb_max) // max(len(user_ids), 1)
    certificate = round_robin_attempt_certificate(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        fair_prb_share=int(fair_prb_share),
    )
    if certificate is not None:
        log_feasibility_certificate(
            certificate,
            scheduler_mode="ofdma_round_robin",
            policy=problem.switch_policy.value,
            attempt_name=attempt_name,
        )
        return build_failed_attempt(
            problem,
            attempt_name=attempt_name,
            allowed_pa_ids=allowed_pa_ids,
            fair_prb_share=int(fair_prb_share),
            selected_rows_by_user={},
            failure_reason=certificate.reason,
        )

    slot_rows_by_slot, delivered_bits_by_user, round_robin_cycle_count = allocate_slots(
        problem,
        allowed_pa_ids=allowed_pa_ids,
    )
    selected_rows_by_user = first_allocated_rows_by_user(slot_rows_by_slot)
    unsatisfied_user_ids = tuple(
        int(user_id)
        for user_id in user_ids
        if float(delivered_bits_by_user[int(user_id)]) + TOL < float(problem.demand_bits_by_user[int(user_id)])
    )
    if unsatisfied_user_ids:
        return build_failed_attempt(
            problem,
            attempt_name=attempt_name,
            allowed_pa_ids=allowed_pa_ids,
            fair_prb_share=int(fair_prb_share),
            selected_rows_by_user=selected_rows_by_user,
            slot_rows_by_slot=slot_rows_by_slot,
            delivered_bits_by_user=delivered_bits_by_user,
            round_robin_cycle_count=int(round_robin_cycle_count),
            failure_reason="Round-robin allocation exhausted the finite frame before satisfying all users.",
        )

    frame_energy_j = compute_frame_energy_j(problem, slot_rows_by_slot)
    return RoundRobinAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=True,
        fair_prb_share=int(fair_prb_share),
        selected_pa_id=selected_attempt_pa_id(selected_rows_by_user),
        selected_rows_by_user=dict(selected_rows_by_user),
        slot_rows_by_slot=slot_rows_by_slot,
        delivered_bits_by_user=delivered_bits_by_user,
        unsatisfied_user_ids=(),
        frame_energy_j=float(frame_energy_j),
        active_slot_count=count_active_slots(slot_rows_by_slot),
        allocation_count=count_allocations(slot_rows_by_slot),
        round_robin_cycle_count=int(round_robin_cycle_count),
        failure_reason=None,
    )


def round_robin_attempt_certificate(
    problem: RoundRobinProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    fair_prb_share: int,
) -> InfeasibilityCertificate | None:
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    bits_by_user = {
        int(user_id): tuple(
            float(row.bits_per_slot)
            for row in candidate_rows
            if int(row.pa_id) in allowed_pa_id_set
        )
        for user_id, candidate_rows in sorted(problem.candidate_rows_by_user.items())
    }
    return row_menu_certificate(
        demand_bits_by_user=problem.demand_bits_by_user,
        bits_by_user=bits_by_user,
        frame_n_slots=int(problem.frame_n_slots),
        max_users_per_slot=max(len(problem.required_rate_by_user), 1),
    )


def select_rows_by_user(
    problem: RoundRobinProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    fair_prb_share: int,
) -> dict[int, RoundRobinCandidateRow]:
    """Select the initial audit row used for diagnostics and tests."""

    selected_rows: dict[int, RoundRobinCandidateRow] = {}
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    for user_id, candidate_rows in sorted(problem.candidate_rows_by_user.items()):
        eligible_rows = tuple(
            row
            for row in candidate_rows
            if int(row.pa_id) in allowed_pa_id_set
            and int(row.n_prb) <= int(fair_prb_share)
        )
        if not eligible_rows:
            continue
        balanced_rows = tuple(row for row in eligible_rows if row_output_fits_prb_fraction(problem, row))
        ranked_rows = balanced_rows if balanced_rows else eligible_rows
        selected_rows[int(user_id)] = min(
            ranked_rows,
            key=lambda row: row_selection_rank(problem, row),
        )
    return selected_rows


def allocate_slots(
    problem: RoundRobinProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> tuple[tuple[tuple[RoundRobinCandidateRow, ...], ...], dict[int, float], int]:
    """Allocate UE rows by rolling PRB quantum with a carried cursor."""

    user_ids = tuple(sorted(problem.required_rate_by_user))
    delivered_bits_by_user = {int(user_id): 0.0 for user_id in user_ids}
    slot_rows_by_slot: list[tuple[RoundRobinCandidateRow, ...]] = []
    round_robin_cycle_count = 0
    next_user_index = 0
    quantum_prbs = rolling_quantum_prbs(problem, allowed_pa_ids=allowed_pa_ids)
    for _slot_id in range(int(problem.frame_n_slots)):
        slot_rows, next_user_index = allocate_one_slot(
            problem,
            user_ids=user_ids,
            allowed_pa_ids=allowed_pa_ids,
            delivered_bits_by_user=delivered_bits_by_user,
            next_user_index=int(next_user_index),
            quantum_prbs=int(quantum_prbs),
        )
        if slot_rows:
            round_robin_cycle_count += 1
        slot_rows_by_slot.append(slot_rows)
        if all_users_satisfied(problem, delivered_bits_by_user):
            break

    while len(slot_rows_by_slot) < int(problem.frame_n_slots):
        slot_rows_by_slot.append(())
    return tuple(slot_rows_by_slot), delivered_bits_by_user, int(round_robin_cycle_count)


def rolling_quantum_prbs(
    problem: RoundRobinProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
) -> int:
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    return min(
        int(row.n_prb)
        for row in problem.candidate_rows
        if int(row.pa_id) in allowed_pa_id_set and float(row.bits_per_slot) > TOL
    )


def allocate_one_slot(
    problem: RoundRobinProblem,
    *,
    user_ids: tuple[int, ...],
    allowed_pa_ids: tuple[int, ...],
    delivered_bits_by_user: dict[int, float],
    next_user_index: int,
    quantum_prbs: int,
) -> tuple[tuple[RoundRobinCandidateRow, ...], int]:
    slot_rows_by_user: dict[int, RoundRobinCandidateRow] = {}
    target_prbs_by_user: dict[int, int] = {}
    slot_user_order: list[int] = []
    rejected_since_last_accept = 0
    max_considerations = max(len(user_ids), 1) * max(
        1,
        int(problem.prb_max + int(quantum_prbs) - 1) // int(quantum_prbs),
    )

    for _consideration in range(int(max_considerations)):
        if all_users_satisfied(problem, delivered_bits_by_user):
            break
        if slot_prbs_used(slot_rows_by_user) >= int(problem.prb_max):
            break

        user_id = int(user_ids[int(next_user_index)])
        next_user_index = (int(next_user_index) + 1) % max(len(user_ids), 1)
        if user_is_satisfied(problem, delivered_bits_by_user, user_id=int(user_id)):
            rejected_since_last_accept += 1
            continue
        if not user_can_receive_next_quantum(
            problem,
            user_ids=user_ids,
            slot_rows_by_user=slot_rows_by_user,
            delivered_bits_by_user=delivered_bits_by_user,
            user_id=int(user_id),
            quantum_prbs=int(quantum_prbs),
        ):
            rejected_since_last_accept += 1
            continue

        target_prbs_by_user[int(user_id)] = min(
            int(problem.prb_max),
            int(target_prbs_by_user.get(int(user_id), 0)) + int(quantum_prbs),
        )
        replacement_row = select_slot_replacement_row(
            problem,
            slot_rows_by_user=slot_rows_by_user,
            delivered_bits_by_user=delivered_bits_by_user,
            allowed_pa_ids=allowed_pa_ids,
            user_id=int(user_id),
            target_prbs=int(target_prbs_by_user[int(user_id)]),
        )
        if replacement_row is None:
            rejected_since_last_accept += 1
            if stop_slot_after_rejections(
                problem,
                user_ids=user_ids,
                target_prbs_by_user=target_prbs_by_user,
                rejected_since_last_accept=int(rejected_since_last_accept),
            ):
                break
            continue

        previous_row = slot_rows_by_user.get(int(user_id))
        previous_bits = 0.0 if previous_row is None else float(previous_row.bits_per_slot)
        if float(replacement_row.bits_per_slot) <= float(previous_bits) + TOL:
            rejected_since_last_accept += 1
            continue

        slot_rows_by_user[int(user_id)] = replacement_row
        if int(user_id) not in slot_user_order:
            slot_user_order.append(int(user_id))
        delivered_bits_by_user[int(user_id)] += float(replacement_row.bits_per_slot) - float(previous_bits)
        rejected_since_last_accept = 0

    slot_rows = tuple(
        slot_rows_by_user[int(user_id)]
        for user_id in slot_user_order
        if int(user_id) in slot_rows_by_user
    )
    if slot_rows:
        next_user_index = next_slot_start_index(
            user_ids,
            slot_rows_by_user=slot_rows_by_user,
        )
    return slot_rows, int(next_user_index)


def select_slot_replacement_row(
    problem: RoundRobinProblem,
    *,
    slot_rows_by_user: dict[int, RoundRobinCandidateRow],
    delivered_bits_by_user: dict[int, float],
    allowed_pa_ids: tuple[int, ...],
    user_id: int,
    target_prbs: int,
) -> RoundRobinCandidateRow | None:
    previous_row = slot_rows_by_user.get(int(user_id))
    previous_bits = 0.0 if previous_row is None else float(previous_row.bits_per_slot)
    remaining_bits = (
        float(problem.demand_bits_by_user[int(user_id)])
        - float(delivered_bits_by_user.get(int(user_id), 0.0))
        + float(previous_bits)
    )
    slot_rows_without_user = tuple(
        row
        for row_user_id, row in slot_rows_by_user.items()
        if int(row_user_id) != int(user_id)
    )
    return select_allocation_row(
        problem,
        slot_rows=slot_rows_without_user,
        allowed_pa_ids=allowed_pa_ids,
        user_id=int(user_id),
        target_prbs=int(target_prbs),
        remaining_bits=float(remaining_bits),
    )


def slot_prbs_used(slot_rows_by_user: dict[int, RoundRobinCandidateRow]) -> int:
    return int(sum(int(row.n_prb) for row in slot_rows_by_user.values()))


def user_can_receive_next_quantum(
    problem: RoundRobinProblem,
    *,
    user_ids: tuple[int, ...],
    slot_rows_by_user: dict[int, RoundRobinCandidateRow],
    delivered_bits_by_user: dict[int, float],
    user_id: int,
    quantum_prbs: int,
) -> bool:
    current_prbs = slot_user_prbs(slot_rows_by_user, user_id=int(user_id))
    min_prbs = min(
        slot_user_prbs(slot_rows_by_user, user_id=int(other_user_id))
        for other_user_id in user_ids
        if not user_is_satisfied(problem, delivered_bits_by_user, user_id=int(other_user_id))
    )
    return int(current_prbs) < int(min_prbs) + int(quantum_prbs)


def slot_user_prbs(
    slot_rows_by_user: dict[int, RoundRobinCandidateRow],
    *,
    user_id: int,
) -> int:
    row = slot_rows_by_user.get(int(user_id))
    if row is None:
        return 0
    return int(row.n_prb)


def next_slot_start_index(
    user_ids: tuple[int, ...],
    *,
    slot_rows_by_user: dict[int, RoundRobinCandidateRow],
) -> int:
    max_prbs = max(int(row.n_prb) for row in slot_rows_by_user.values())
    for user_index, user_id in enumerate(user_ids):
        row = slot_rows_by_user.get(int(user_id))
        if row is not None and int(row.n_prb) == int(max_prbs):
            return (int(user_index) + 1) % max(len(user_ids), 1)
    return 0


def stop_slot_after_rejections(
    problem: RoundRobinProblem,
    *,
    user_ids: tuple[int, ...],
    target_prbs_by_user: dict[int, int],
    rejected_since_last_accept: int,
) -> bool:
    if int(rejected_since_last_accept) < len(user_ids):
        return False
    return all(
        int(target_prbs_by_user.get(int(user_id), 0)) >= int(problem.prb_max)
        for user_id in user_ids
    )


def build_failed_attempt(
    problem: RoundRobinProblem,
    *,
    attempt_name: str,
    allowed_pa_ids: tuple[int, ...],
    fair_prb_share: int,
    selected_rows_by_user: dict[int, RoundRobinCandidateRow],
    failure_reason: str,
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...] | None = None,
    delivered_bits_by_user: dict[int, float] | None = None,
    round_robin_cycle_count: int = 0,
) -> RoundRobinAttemptResult:
    user_ids = tuple(sorted(problem.required_rate_by_user))
    resolved_slot_rows = empty_slot_rows(problem) if slot_rows_by_slot is None else slot_rows_by_slot
    resolved_delivered_bits = (
        {int(user_id): 0.0 for user_id in user_ids}
        if delivered_bits_by_user is None
        else dict(delivered_bits_by_user)
    )
    unsatisfied_user_ids = tuple(
        int(user_id)
        for user_id in user_ids
        if float(resolved_delivered_bits.get(int(user_id), 0.0)) + TOL < float(problem.demand_bits_by_user[int(user_id)])
    )
    return RoundRobinAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=False,
        fair_prb_share=int(fair_prb_share),
        selected_pa_id=selected_attempt_pa_id(selected_rows_by_user),
        selected_rows_by_user=dict(selected_rows_by_user),
        slot_rows_by_slot=resolved_slot_rows,
        delivered_bits_by_user=resolved_delivered_bits,
        unsatisfied_user_ids=unsatisfied_user_ids,
        frame_energy_j=None,
        active_slot_count=count_active_slots(resolved_slot_rows),
        allocation_count=count_allocations(resolved_slot_rows),
        round_robin_cycle_count=int(round_robin_cycle_count),
        failure_reason=str(failure_reason),
    )


def compute_frame_energy_j(
    problem: RoundRobinProblem,
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...],
) -> float:
    return float(problem.t_slot_s) * sum(
        compute_slot_dc_power_w(problem, slot_rows)
        for slot_rows in slot_rows_by_slot
        if slot_rows
    )


def row_selection_rank(
    problem: RoundRobinProblem,
    row: RoundRobinCandidateRow,
) -> tuple[float, float, int, float, int, int, int, int, int]:
    return (
        float(row_output_balance_error(problem, row)),
        -float(row.bits_per_slot),
        int(row.n_prb),
        float(row.p_dc_active_w),
        -int(row.mcs),
        -int(row.layers),
        int(row.pa_id),
        int(row.mcs),
        int(row.layers),
        int(row.local_row_id),
    )


def selected_attempt_pa_id(selected_rows_by_user: dict[int, RoundRobinCandidateRow]) -> int | None:
    selected_pa_ids = tuple(sorted({int(row.pa_id) for row in selected_rows_by_user.values()}))
    if not selected_pa_ids:
        return None
    return int(selected_pa_ids[0])


def first_allocated_rows_by_user(
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...],
) -> dict[int, RoundRobinCandidateRow]:
    rows_by_user: dict[int, RoundRobinCandidateRow] = {}
    for slot_rows in slot_rows_by_slot:
        for row in slot_rows:
            rows_by_user.setdefault(int(row.user_id), row)
    return rows_by_user


def all_users_satisfied(
    problem: RoundRobinProblem,
    delivered_bits_by_user: dict[int, float],
) -> bool:
    return all(
        user_is_satisfied(problem, delivered_bits_by_user, user_id=int(user_id))
        for user_id in problem.required_rate_by_user
    )


def user_is_satisfied(
    problem: RoundRobinProblem,
    delivered_bits_by_user: dict[int, float],
    *,
    user_id: int,
) -> bool:
    return (
        float(delivered_bits_by_user.get(int(user_id), 0.0)) + TOL
        >= float(problem.demand_bits_by_user[int(user_id)])
    )


def empty_slot_rows(problem: RoundRobinProblem) -> tuple[tuple[RoundRobinCandidateRow, ...], ...]:
    return tuple(() for _slot_id in range(int(problem.frame_n_slots)))


def count_active_slots(
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...],
) -> int:
    return int(sum(1 for slot_rows in slot_rows_by_slot if slot_rows))


def count_allocations(
    slot_rows_by_slot: tuple[tuple[RoundRobinCandidateRow, ...], ...],
) -> int:
    return int(sum(len(slot_rows) for slot_rows in slot_rows_by_slot))


__all__ = [
    "compute_slot_dc_power_w",
    "round_robin_attempt_certificate",
    "run_round_robin_attempt",
    "select_rows_by_user",
]
