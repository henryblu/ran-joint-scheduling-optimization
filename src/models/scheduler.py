from __future__ import annotations

"""Shared scheduler mode selection enums."""

from enum import Enum


class SchedulerMode(Enum):
    TDMA = "tdma"
    OFDMA = "ofdma"


__all__ = ["SchedulerMode"]
