from __future__ import annotations

"""Scheduler-local infeasibility certificates over optimistic frame bounds."""

from dataclasses import dataclass
import logging
import math

from run_reporting import build_console_message, current_run_scope


TOL = 1e-9
LOGGER = logging.getLogger("snapshot_run")


@dataclass(frozen=True)
class InfeasibilityCertificate:
    """One necessary-condition proof that a scheduler attempt cannot serve the frame."""

    bound_name: str
    reason: str
    details: dict[str, object]


def positive_user_count_certificate(
    *,
    demand_bits_by_user: dict[int, float],
    frame_n_slots: int,
) -> InfeasibilityCertificate | None:
    positive_user_count = sum(1 for demand_bits in demand_bits_by_user.values() if float(demand_bits) > TOL)
    if int(positive_user_count) <= int(frame_n_slots):
        return None
    return InfeasibilityCertificate(
        bound_name="tdma_user_count_bound",
        reason=(
            "Bound-certified infeasible: tdma_user_count_bound "
            f"(positive_users={int(positive_user_count)} frame_slots={int(frame_n_slots)})."
        ),
        details={
            "positive_users": int(positive_user_count),
            "frame_slots": int(frame_n_slots),
        },
    )


def row_menu_certificate(
    *,
    demand_bits_by_user: dict[int, float],
    bits_by_user: dict[int, tuple[float, ...]],
    frame_n_slots: int,
    max_users_per_slot: int,
) -> InfeasibilityCertificate | None:
    for user_id in sorted(demand_bits_by_user):
        demand_bits = float(demand_bits_by_user[int(user_id)])
        positive_bits = tuple(float(bits) for bits in bits_by_user.get(int(user_id), ()) if float(bits) > TOL)
        if demand_bits <= TOL:
            continue
        if not positive_bits:
            return InfeasibilityCertificate(
                bound_name="no_positive_payload_row",
                reason=(
                    "Bound-certified infeasible: no_positive_payload_row "
                    f"(user_id={int(user_id)} demand_bits={demand_bits:.12g})."
                ),
                details={
                    "user_id": int(user_id),
                    "demand_bits": float(demand_bits),
                },
            )
        max_bits_per_slot = max(positive_bits)
        if demand_bits > float(frame_n_slots) * float(max_bits_per_slot) + TOL:
            return InfeasibilityCertificate(
                bound_name="per_user_capacity_bound",
                reason=(
                    "Bound-certified infeasible: per_user_capacity_bound "
                    f"(user_id={int(user_id)} demand_bits={demand_bits:.12g} "
                    f"frame_capacity_bits={float(frame_n_slots) * float(max_bits_per_slot):.12g})."
                ),
                details={
                    "user_id": int(user_id),
                    "demand_bits": float(demand_bits),
                    "max_bits_per_slot": float(max_bits_per_slot),
                    "frame_slots": int(frame_n_slots),
                    "frame_capacity_bits": float(frame_n_slots) * float(max_bits_per_slot),
                },
            )

    min_required_appearances = sum(
        min_user_appearances(
            demand_bits=float(demand_bits_by_user[int(user_id)]),
            bits_by_user=bits_by_user.get(int(user_id), ()),
        )
        for user_id in sorted(demand_bits_by_user)
    )
    appearance_capacity = int(frame_n_slots) * int(max_users_per_slot)
    if int(min_required_appearances) <= int(appearance_capacity):
        return None
    return InfeasibilityCertificate(
        bound_name="slot_lower_bound",
        reason=(
            "Bound-certified infeasible: slot_lower_bound "
            f"(min_required_appearances={int(min_required_appearances)} "
            f"appearance_capacity={int(appearance_capacity)})."
        ),
        details={
            "min_required_appearances": int(min_required_appearances),
            "appearance_capacity": int(appearance_capacity),
            "frame_slots": int(frame_n_slots),
            "max_users_per_slot": int(max_users_per_slot),
        },
    )


def min_user_appearances(*, demand_bits: float, bits_by_user: tuple[float, ...]) -> int:
    if float(demand_bits) <= TOL:
        return 0
    positive_bits = tuple(float(bits) for bits in bits_by_user if float(bits) > TOL)
    if not positive_bits:
        return math.inf
    return int(math.ceil(float(demand_bits) / max(positive_bits) - TOL))


def log_feasibility_certificate(
    certificate: InfeasibilityCertificate,
    *,
    scheduler_mode: str,
    policy: str,
    attempt_name: str,
) -> None:
    fields = [
        ("mode", str(scheduler_mode)),
        ("policy", str(policy)),
        ("attempt", str(attempt_name)),
        ("bound", str(certificate.bound_name)),
    ]
    fields.extend((str(key), str(value)) for key, value in certificate.details.items())
    LOGGER.info(
        build_console_message(
            level_tag="INFO",
            scope=current_run_scope(),
            stage="feasibility_bound",
            event="infeasible",
            fields=fields,
        )
    )


__all__ = [
    "InfeasibilityCertificate",
    "log_feasibility_certificate",
    "positive_user_count_certificate",
    "row_menu_certificate",
]
