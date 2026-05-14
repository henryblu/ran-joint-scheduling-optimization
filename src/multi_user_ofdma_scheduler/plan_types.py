from __future__ import annotations

from dataclasses import dataclass, field

from models import SlotAllocation, SlotSchedule


@dataclass(frozen=True)
class _CandidateView:
    """Trusted slot row used by the OFDMA planner.

    p_out_total_w is total RF output for this allocation in one active slot.
    p_dc_active_w is the single-row active PA DC draw and is diagnostic once
    rows are packed; packed slot DC power is computed from aggregate RF output.
    """

    user_id: int
    candidate_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    bits_per_slot: float
    p_dc_active_w: float
    p_out_total_w: float

    def rank_key(self) -> tuple[int, int, int, int, float, float, float, int]:
        return (
            int(self.pa_id),
            int(self.n_prb),
            int(self.mcs),
            int(self.layers),
            float(self.bits_per_slot),
            float(self.p_out_total_w),
            float(self.p_dc_active_w),
            int(self.candidate_id),
        )

    def to_allocation(self) -> SlotAllocation:
        return SlotAllocation(
            user_id=int(self.user_id),
            pa_id=int(self.pa_id),
            n_prb=int(self.n_prb),
            layers=int(self.layers),
            mcs=int(self.mcs),
            bits_per_slot=float(self.bits_per_slot),
            p_out_total_w=float(self.p_out_total_w),
            p_dc_active_w=float(self.p_dc_active_w),
        )


_UserCandidateRow = _CandidateView


@dataclass(frozen=True)
class _UserPlanRowInstance:
    candidate: _CandidateView
    count: int
    exact_single_slot_dc_w: float

    @property
    def user_id(self) -> int:
        return int(self.candidate.user_id)

    @property
    def pa_id(self) -> int:
        return int(self.candidate.pa_id)

    @property
    def delivered_bits(self) -> float:
        return int(self.count) * float(self.candidate.bits_per_slot)

    @property
    def total_p_out_w(self) -> float:
        return int(self.count) * float(self.candidate.p_out_total_w)

    @property
    def total_prbs(self) -> int:
        return int(self.count) * int(self.candidate.n_prb)

    def rank_key(self) -> tuple[int, int, int, int, int, float, float, float]:
        return (
            int(self.candidate.pa_id),
            int(self.count),
            int(self.candidate.n_prb),
            int(self.candidate.mcs),
            int(self.candidate.layers),
            -float(self.candidate.bits_per_slot),
            float(self.candidate.p_out_total_w),
            float(self.candidate.candidate_id),
        )


@dataclass(frozen=True)
class _CoveragePlan:
    user_id: int
    pa_id: int
    candidate: _CandidateView
    n_slots: int
    delivered_bits: float
    overdelivery_bits: float
    area_prb_slots: int
    total_p_out_w: float
    exact_serial_energy_j: float = 0.0
    row_instances: tuple[_UserPlanRowInstance, ...] = ()

    @property
    def total_exact_serial_energy_j(self) -> float:
        return float(self.exact_serial_energy_j)

    def iter_row_instances(self) -> tuple[_UserPlanRowInstance, ...]:
        if self.row_instances:
            return self.row_instances
        return (
            _UserPlanRowInstance(
                candidate=self.candidate,
                count=int(self.n_slots),
                exact_single_slot_dc_w=0.0,
            ),
        )

    def uses_pa_ids(self) -> tuple[int, ...]:
        return tuple(sorted({int(row_instance.pa_id) for row_instance in self.iter_row_instances()}))

    def signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(row_instance.candidate.candidate_id), int(row_instance.count))
            for row_instance in self.iter_row_instances()
        )

    def rank_key(self) -> tuple[int, float, float, int, tuple[tuple[int, int], ...]]:
        return (
            int(self.n_slots),
            float(self.exact_serial_energy_j),
            float(self.overdelivery_bits),
            int(self.area_prb_slots),
            self.signature(),
        )


@dataclass(frozen=True)
class _ScheduleToken:
    user_id: int
    token_index: int
    candidate: _CandidateView

    @property
    def pa_id(self) -> int:
        return int(self.candidate.pa_id)

    @property
    def n_prb(self) -> int:
        return int(self.candidate.n_prb)

    @property
    def bits_per_slot(self) -> float:
        return float(self.candidate.bits_per_slot)

    @property
    def p_out_total_w(self) -> float:
        return float(self.candidate.p_out_total_w)

    def stable_key(self) -> tuple[int, int, int, int, int, float, float, int, int]:
        return (
            int(self.user_id),
            int(self.pa_id),
            int(self.n_prb),
            int(self.candidate.mcs),
            int(self.candidate.layers),
            float(self.bits_per_slot),
            float(self.p_out_total_w),
            int(self.candidate.candidate_id),
            int(self.token_index),
        )


@dataclass
class _MutableSlotState:
    slot_id: int
    pa_id: int | None = None
    used_prbs: int = 0
    aggregate_p_out_w: float = 0.0
    dc_power_w: float = 0.0
    tokens: list[_ScheduleToken] = field(default_factory=list)
    scheduled_users: set[int] = field(default_factory=set)

    def to_schedule(self) -> SlotSchedule:
        allocations = tuple(
            sorted(
                (token.candidate.to_allocation() for token in self.tokens),
                key=lambda allocation: (
                    allocation.user_id,
                    allocation.pa_id,
                    allocation.n_prb,
                    allocation.mcs,
                    allocation.layers,
                    allocation.bits_per_slot,
                    allocation.p_out_total_w,
                ),
            )
        )
        return SlotSchedule(
            slot_index=int(self.slot_id),
            active=bool(allocations),
            pa_id=None if self.pa_id is None else int(self.pa_id),
            used_prbs=int(self.used_prbs),
            aggregate_p_out_w=float(self.aggregate_p_out_w),
            dc_power_w=float(self.dc_power_w),
            allocations=allocations,
        )


@dataclass(frozen=True)
class _PackedFrame:
    slot_schedules: tuple[SlotSchedule, ...]
    delivered_bits_by_user: dict[int, float]
    frame_energy_j: float
    average_frame_dc_power_w: float
    average_frame_rf_output_w: float

    def rank_key(self) -> tuple[float, int, tuple[tuple[int, int, float, tuple[tuple[int, int, int, int, int, float, float, int], ...]], ...]]:
        return (
            float(self.frame_energy_j),
            int(sum(slot.active for slot in self.slot_schedules)),
            tuple(_slot_schedule_signature(slot) for slot in self.slot_schedules),
        )


def _slot_schedule_signature(
    slot: SlotSchedule,
) -> tuple[int, int, float, tuple[tuple[int, int, int, int, int, float, float, int], ...]]:
    return (
        -1 if slot.pa_id is None else int(slot.pa_id),
        int(slot.used_prbs),
        float(slot.aggregate_p_out_w),
        tuple(
            (
                int(allocation.user_id),
                int(allocation.pa_id),
                int(allocation.n_prb),
                int(allocation.mcs),
                int(allocation.layers),
                float(allocation.bits_per_slot),
                float(allocation.p_out_total_w),
                int(index),
            )
            for index, allocation in enumerate(slot.allocations)
        ),
    )


__all__ = [
    "_CandidateView",
    "_CoveragePlan",
    "_MutableSlotState",
    "_PackedFrame",
    "_ScheduleToken",
    "_UserCandidateRow",
    "_UserPlanRowInstance",
    "_slot_schedule_signature",
]
