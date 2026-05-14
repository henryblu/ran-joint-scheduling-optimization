from __future__ import annotations

from configs.pa import pa_slot_dc_power

from .models import PreparedJointOfdmaProblem
from .plan_types import _CoveragePlan, _MutableSlotState, _PackedFrame, _ScheduleToken


TOL = 1e-12


def build_schedule_tokens(
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> tuple[_ScheduleToken, ...]:
    tokens = []
    for user_id, plan in sorted(selected_plans_by_user.items()):
        token_index = 0
        for row_instance in plan.iter_row_instances():
            for _ in range(int(row_instance.count)):
                tokens.append(
                    _ScheduleToken(
                        user_id=int(user_id),
                        token_index=int(token_index),
                        candidate=row_instance.candidate,
                    )
                )
                token_index += 1
    return tuple(tokens)


def pack_selected_plans_serially(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> _PackedFrame | None:
    slot_states = build_serial_slot_states(
        problem,
        selected_plans_by_user=selected_plans_by_user,
    )
    if slot_states is None:
        return None
    return finalize_packed_frame(
        problem,
        required_bits_by_user=required_bits_by_user,
        slot_states=slot_states,
    )


def pack_serial_then_compact(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> _PackedFrame | None:
    slot_states = build_serial_slot_states(
        problem,
        selected_plans_by_user=selected_plans_by_user,
    )
    if slot_states is None:
        return None

    compact_slot_states(problem, slot_states=slot_states)
    return finalize_packed_frame(
        problem,
        required_bits_by_user=required_bits_by_user,
        slot_states=slot_states,
    )


def build_serial_slot_states(
    problem: PreparedJointOfdmaProblem,
    *,
    selected_plans_by_user: dict[int, _CoveragePlan],
) -> list[_MutableSlotState] | None:
    tokens = build_schedule_tokens(selected_plans_by_user)
    if len(tokens) > int(problem.frame_n_slots):
        return None

    slot_states = [
        _MutableSlotState(slot_id=int(slot_id))
        for slot_id in range(int(problem.frame_n_slots))
    ]
    for slot_id, token in enumerate(tokens):
        slot_state = slot_states[int(slot_id)]
        slot_state.tokens.append(token)
        recompute_slot_state(problem, slot_state)
    return slot_states


def compact_slot_states(
    problem: PreparedJointOfdmaProblem,
    *,
    slot_states: list[_MutableSlotState],
) -> None:
    max_iterations = max(1, 2 * sum(len(slot.tokens) for slot in slot_states))
    for _ in range(int(max_iterations)):
        move = find_best_energy_improving_move(problem, slot_states=slot_states)
        if move is None:
            return
        source_slot_id, token_index, target_slot_id = move
        token = slot_states[int(source_slot_id)].tokens.pop(int(token_index))
        slot_states[int(target_slot_id)].tokens.append(token)
        recompute_slot_state(problem, slot_states[int(source_slot_id)])
        recompute_slot_state(problem, slot_states[int(target_slot_id)])


def find_best_energy_improving_move(
    problem: PreparedJointOfdmaProblem,
    *,
    slot_states: list[_MutableSlotState],
) -> tuple[int, int, int] | None:
    best_move = min(
        iter_energy_improving_moves(problem, slot_states=slot_states),
        default=None,
        key=lambda move: move[0],
    )
    if best_move is None:
        return None
    _, source_slot_id, token_index, target_slot_id = best_move
    return int(source_slot_id), int(token_index), int(target_slot_id)


def iter_energy_improving_moves(
    problem: PreparedJointOfdmaProblem,
    *,
    slot_states: list[_MutableSlotState],
):
    for source_slot in slot_states:
        for token_index, token in enumerate(source_slot.tokens):
            yield from iter_token_energy_improving_moves(
                problem,
                source_slot=source_slot,
                token_index=int(token_index),
                token=token,
                slot_states=slot_states,
            )


def iter_token_energy_improving_moves(
    problem: PreparedJointOfdmaProblem,
    *,
    source_slot: _MutableSlotState,
    token_index: int,
    token: _ScheduleToken,
    slot_states: list[_MutableSlotState],
):
    for target_slot in slot_states:
        if int(source_slot.slot_id) == int(target_slot.slot_id):
            continue

        delta_w = move_energy_delta_w(
            problem,
            source_slot=source_slot,
            target_slot=target_slot,
            token=token,
        )
        if delta_w is None or float(delta_w) >= -TOL:
            continue

        yield (
            (
                float(delta_w),
                int(source_slot.slot_id),
                int(token_index),
                int(target_slot.slot_id),
                token.stable_key(),
            ),
            int(source_slot.slot_id),
            int(token_index),
            int(target_slot.slot_id),
        )


def move_energy_delta_w(
    problem: PreparedJointOfdmaProblem,
    *,
    source_slot: _MutableSlotState,
    target_slot: _MutableSlotState,
    token: _ScheduleToken,
) -> float | None:
    if target_slot.pa_id is not None and int(target_slot.pa_id) != int(token.pa_id):
        return None
    if int(token.user_id) in target_slot.scheduled_users:
        return None
    if int(target_slot.used_prbs) + int(token.n_prb) > int(problem.prb_max):
        return None

    target_aggregate_p_out_w = float(target_slot.aggregate_p_out_w) + float(token.p_out_total_w)
    if target_aggregate_p_out_w > slot_pa_output_limit_w(problem, pa_id=int(token.pa_id)) + TOL:
        return None

    source_tokens = [candidate_token for candidate_token in source_slot.tokens if candidate_token is not token]
    source_dc_power_w = slot_dc_for_tokens(problem, source_tokens)
    target_dc_power_w = slot_dc_power_w(
        problem,
        pa_id=int(token.pa_id),
        aggregate_rf_output_w=float(target_aggregate_p_out_w),
    )
    return (
        float(source_dc_power_w)
        + float(target_dc_power_w)
        - float(source_slot.dc_power_w)
        - float(target_slot.dc_power_w)
    )


def slot_dc_for_tokens(
    problem: PreparedJointOfdmaProblem,
    tokens: list[_ScheduleToken],
) -> float:
    if not tokens:
        return 0.0
    pa_id = int(tokens[0].pa_id)
    aggregate_p_out_w = sum(float(token.p_out_total_w) for token in tokens)
    return slot_dc_power_w(
        problem,
        pa_id=int(pa_id),
        aggregate_rf_output_w=float(aggregate_p_out_w),
    )


def recompute_slot_state(
    problem: PreparedJointOfdmaProblem,
    slot_state: _MutableSlotState,
) -> None:
    if not slot_state.tokens:
        slot_state.pa_id = None
        slot_state.used_prbs = 0
        slot_state.aggregate_p_out_w = 0.0
        slot_state.dc_power_w = 0.0
        slot_state.scheduled_users = set()
        return

    slot_state.pa_id = int(slot_state.tokens[0].pa_id)
    slot_state.used_prbs = int(sum(int(token.n_prb) for token in slot_state.tokens))
    slot_state.aggregate_p_out_w = float(sum(float(token.p_out_total_w) for token in slot_state.tokens))
    slot_state.dc_power_w = slot_dc_power_w(
        problem,
        pa_id=int(slot_state.pa_id),
        aggregate_rf_output_w=float(slot_state.aggregate_p_out_w),
    )
    slot_state.scheduled_users = {int(token.user_id) for token in slot_state.tokens}


def finalize_packed_frame(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    slot_states: list[_MutableSlotState],
) -> _PackedFrame | None:
    delivered_bits_by_user = {
        int(user_id): 0.0
        for user_id in sorted(required_bits_by_user)
    }
    for slot_state in slot_states:
        for token in slot_state.tokens:
            delivered_bits_by_user[int(token.user_id)] = (
                float(delivered_bits_by_user[int(token.user_id)])
                + float(token.bits_per_slot)
            )

    if any(
        float(delivered_bits_by_user[int(user_id)]) + TOL < float(required_bits)
        for user_id, required_bits in sorted(required_bits_by_user.items())
    ):
        return None

    slot_schedules = tuple(slot_state.to_schedule() for slot_state in slot_states)
    frame_energy_j = float(problem.t_slot_s) * float(sum(slot.dc_power_w for slot in slot_schedules))
    frame_duration_s = float(problem.frame_n_slots) * float(problem.t_slot_s)
    return _PackedFrame(
        slot_schedules=slot_schedules,
        delivered_bits_by_user=delivered_bits_by_user,
        frame_energy_j=float(frame_energy_j),
        average_frame_dc_power_w=float(frame_energy_j) / max(float(frame_duration_s), TOL),
        average_frame_rf_output_w=float(sum(slot.aggregate_p_out_w for slot in slot_schedules)) / max(int(problem.frame_n_slots), 1),
    )


def validate_final_packed_frame(
    problem: PreparedJointOfdmaProblem,
    *,
    required_bits_by_user: dict[int, float],
    packed_frame: _PackedFrame,
) -> None:
    delivered_bits_by_user = {
        int(user_id): 0.0
        for user_id in sorted(required_bits_by_user)
    }
    expected_frame_energy_j = 0.0
    if len(packed_frame.slot_schedules) != int(problem.frame_n_slots):
        raise ValueError("The packed OFDMA frame does not contain the expected number of slots.")

    for slot in packed_frame.slot_schedules:
        if not slot.allocations:
            if slot.pa_id is not None or slot.used_prbs != 0:
                raise ValueError("Blank OFDMA slots must not carry a PA gear or PRB usage.")
            if abs(float(slot.aggregate_p_out_w)) > TOL or abs(float(slot.dc_power_w)) > TOL:
                raise ValueError("Blank OFDMA slots must have zero RF output and zero DC power.")
            continue

        pa_ids = {int(allocation.pa_id) for allocation in slot.allocations}
        if len(pa_ids) != 1 or int(slot.pa_id) not in pa_ids:
            raise ValueError("Mixed PA gears are not allowed inside an OFDMA slot.")

        scheduled_users = [int(allocation.user_id) for allocation in slot.allocations]
        if len(scheduled_users) != len(set(scheduled_users)):
            raise ValueError("The same user cannot appear twice in one OFDMA slot.")

        expected_used_prbs = int(sum(int(allocation.n_prb) for allocation in slot.allocations))
        if int(slot.used_prbs) != int(expected_used_prbs):
            raise ValueError("Reported OFDMA slot PRB usage does not match the slot allocations.")
        if int(slot.used_prbs) > int(problem.prb_max):
            raise ValueError("An OFDMA slot exceeded the frame PRB budget.")

        expected_aggregate_p_out_w = float(sum(float(allocation.p_out_total_w) for allocation in slot.allocations))
        if abs(float(slot.aggregate_p_out_w) - float(expected_aggregate_p_out_w)) > TOL:
            raise ValueError("Reported OFDMA slot RF output does not match the slot allocations.")
        if float(slot.aggregate_p_out_w) > slot_pa_output_limit_w(problem, pa_id=int(slot.pa_id)) + TOL:
            raise ValueError("An OFDMA slot exceeded the selected PA output limit.")

        expected_dc_power_w = slot_dc_power_w(
            problem,
            pa_id=int(slot.pa_id),
            aggregate_rf_output_w=float(slot.aggregate_p_out_w),
        )
        if abs(float(slot.dc_power_w) - float(expected_dc_power_w)) > TOL:
            raise ValueError("OFDMA slot DC power must be recomputed from aggregate RF output.")

        for allocation in slot.allocations:
            delivered_bits_by_user[int(allocation.user_id)] += float(allocation.bits_per_slot)
        expected_frame_energy_j += float(slot.dc_power_w) * float(problem.t_slot_s)

    for user_id, required_bits in sorted(required_bits_by_user.items()):
        if float(delivered_bits_by_user[int(user_id)]) + TOL < float(required_bits):
            raise ValueError("The packed OFDMA frame returned an unsatisfied user.")

    if abs(float(expected_frame_energy_j) - float(packed_frame.frame_energy_j)) > TOL:
        raise ValueError("The packed OFDMA frame energy does not match the sum of slot energies.")


def slot_dc_power_w(
    problem: PreparedJointOfdmaProblem,
    *,
    pa_id: int,
    aggregate_rf_output_w: float,
) -> float:
    if float(aggregate_rf_output_w) <= TOL:
        return 0.0

    return pa_slot_dc_power(
        problem.pa_catalog[int(pa_id)],
        p_out_total_w=float(aggregate_rf_output_w),
        n_tx_chains=int(problem.n_tx_chains),
    )


def slot_pa_output_limit_w(
    problem: PreparedJointOfdmaProblem,
    *,
    pa_id: int,
) -> float:
    return float(problem.n_tx_chains) * float(problem.pa_catalog[int(pa_id)].p_max_w)


__all__ = [
    "pack_selected_plans_serially",
    "pack_serial_then_compact",
    "slot_pa_output_limit_w",
    "slot_dc_power_w",
    "validate_final_packed_frame",
]
