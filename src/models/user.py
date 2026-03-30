"""Shared user-request models used across presets and orchestration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserRequest:
    """Immutable user request used by shared presets and higher-level orchestration."""

    user_id: int
    distance_m: float
    required_rate_bps: float


__all__ = ["UserRequest"]
