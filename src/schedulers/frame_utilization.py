from __future__ import annotations

"""Raw frame-utilization diagnostics derived from shared slot schedules."""

from models import MultiUserScheduleResult


def summarize_frame_utilization(
    result: MultiUserScheduleResult,
    *,
    frame_prb_count: int,
    frame_tx_chain_count: int,
) -> dict[str, float | int | bool | str]:
    """Derive raw frame resource use from the public schedule truth."""

    frame_slot_count = int(len(result.slot_schedules))
    active_slot_count = int(sum(slot.active for slot in result.slot_schedules))
    used_prb_slot_area = int(sum(int(slot.used_prbs) for slot in result.slot_schedules))
    allocation_count = int(sum(len(slot.allocations) for slot in result.slot_schedules))
    multi_user_slot_count = int(sum(1 for slot in result.slot_schedules if len(slot.allocations) > 1))
    used_spatial_volume = int(
        sum(
            int(allocation.n_prb) * int(allocation.layers)
            for slot in result.slot_schedules
            for allocation in slot.allocations
        )
    )
    mcs_weight = int(
        sum(
            int(allocation.n_prb) * int(allocation.layers)
            for slot in result.slot_schedules
            for allocation in slot.allocations
        )
    )
    weighted_mcs_total = float(
        sum(
            float(allocation.mcs) * float(allocation.n_prb) * float(allocation.layers)
            for slot in result.slot_schedules
            for allocation in slot.allocations
        )
    )
    return {
        "mode": str(result.scheduler_mode.value),
        "feasible": bool(result.feasible),
        "users": int(len(result.user_summaries)),
        "slots": int(frame_slot_count),
        "prb": int(frame_prb_count),
        "chains": int(frame_tx_chain_count),
        "time": float(active_slot_count) / float(frame_slot_count),
        "prb_util": float(used_prb_slot_area) / float(frame_slot_count * int(frame_prb_count)),
        "spatial": float(used_spatial_volume) / float(frame_slot_count * int(frame_prb_count) * int(frame_tx_chain_count)),
        "alloc": int(allocation_count),
        "alloc_slot": float(allocation_count) / float(active_slot_count) if active_slot_count else 0.0,
        "multi_slot": float(multi_user_slot_count) / float(active_slot_count) if active_slot_count else 0.0,
        "mcs": float(weighted_mcs_total) / float(mcs_weight) if mcs_weight else 0.0,
        "reason": "" if result.infeasible_reason is None else str(result.infeasible_reason),
    }


def frame_utilization_log_fields(
    result: MultiUserScheduleResult,
    *,
    frame_prb_count: int,
    frame_tx_chain_count: int,
) -> list[tuple[str, str]]:
    """Format the raw utilization summary for compact scheduler logs."""

    summary = summarize_frame_utilization(
        result,
        frame_prb_count=int(frame_prb_count),
        frame_tx_chain_count=int(frame_tx_chain_count),
    )
    fields = [
        ("mode", str(summary["mode"])),
        ("feasible", str(bool(summary["feasible"]))),
        ("users", str(int(summary["users"]))),
        ("slots", str(int(summary["slots"]))),
        ("prb", str(int(summary["prb"]))),
        ("chains", str(int(summary["chains"]))),
        ("time", _format_ratio(float(summary["time"]))),
        ("prb_util", _format_ratio(float(summary["prb_util"]))),
        ("spatial", _format_ratio(float(summary["spatial"]))),
        ("alloc", str(int(summary["alloc"]))),
        ("alloc_slot", f"{float(summary['alloc_slot']):.2f}"),
        ("multi_slot", _format_ratio(float(summary["multi_slot"]))),
        ("mcs", f"{float(summary['mcs']):.2f}"),
    ]
    if not bool(summary["feasible"]):
        fields.append(("reason", _format_reason(str(summary["reason"]))))
    return fields


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}"


def _format_reason(reason: str) -> str:
    return str(reason).replace(" ", "_")


__all__ = [
    "frame_utilization_log_fields",
    "summarize_frame_utilization",
]
