from __future__ import annotations

"""Shared public multi-user scheduler result models."""

from dataclasses import dataclass, field
from typing import Any

from .scheduler import SchedulerMode


@dataclass(frozen=True)
class SchedulerPowerSummary:
    """Authoritative frame-level power and energy summary."""

    frame_energy_j: float
    average_frame_dc_power_w: float
    active_energy_j: float
    inactive_energy_j: float
    average_frame_rf_output_w: float


@dataclass(frozen=True)
class UserScheduleSummary:
    """Per-user delivery summary shared by scheduler backends."""

    user_id: int
    required_bits: float
    delivered_bits: float
    required_rate_bps: float
    delivered_rate_bps: float
    satisfied: bool


@dataclass(frozen=True)
class SlotAllocation:
    """One user allocation inside one reported slot schedule.

    p_out_total_w is the row's complete active-slot RF output contribution.
    p_dc_active_w is the diagnostic single-row active PA DC draw. OFDMA slot
    cost is recomputed from aggregate slot RF output, not by summing this field.
    """

    user_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    bits_per_slot: float
    p_out_total_w: float
    p_dc_active_w: float


@dataclass(frozen=True)
class SlotSchedule:
    """One canonical slot schedule entry shared by TDMA and OFDMA results."""

    slot_index: int
    active: bool
    pa_id: int | None
    used_prbs: int
    aggregate_p_out_w: float
    dc_power_w: float
    allocations: tuple[SlotAllocation, ...]


@dataclass(frozen=True)
class MultiUserScheduleResult:
    """Shared public scheduler result returned by the multi-user dispatcher."""

    scheduler_mode: SchedulerMode
    feasible: bool
    infeasible_reason: str | None
    power_summary: SchedulerPowerSummary
    user_summaries: tuple[UserScheduleSummary, ...]
    slot_schedules: tuple[SlotSchedule, ...]
    solver_details: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "MultiUserScheduleResult",
    "SchedulerPowerSummary",
    "SlotAllocation",
    "SlotSchedule",
    "UserScheduleSummary",
]
