from __future__ import annotations

"""Shared PA data models and enums."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PASwitchPolicy(Enum):
    BASELINE_8W_ONLY = "baseline_8w_only"
    HARD_OFF = "hard_off"
    DUAL_SWITCHABLE = "dual_switchable"


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
    "PASwitchPolicy",
]
