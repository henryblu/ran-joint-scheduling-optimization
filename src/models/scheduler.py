from __future__ import annotations

"""Shared source-facing scheduler mode selection enums."""

from enum import Enum


class SchedulerMode(Enum):
    ROUND_ROBIN = "round_robin"
    K_MILP = "k_milp"


__all__ = ["SchedulerMode"]