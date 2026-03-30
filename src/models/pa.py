"""Shared PA data models and enums."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PAState(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    OFF = "off"


class PASwitchPolicy(Enum):
    STANDBY = "standby"
    HARD_OFF = "hard_off"


@dataclass(frozen=True)
class PAParams:
    """Measured PA representation."""

    p_max_w: float
    p_idle_w: float
    eta_max: float
    g_pa_eff_linear: float
    kappa_distortion: float
    backoff_db: float
    pa_name: str = ""
    scenario_label: str = ""
    curve_pout_w: np.ndarray | None = None
    curve_pdc_w: np.ndarray | None = None
    curve_pin_w: np.ndarray | None = None
    source_csv: str = ""


__all__ = [
    "PAParams",
    "PAState",
    "PASwitchPolicy",
]
