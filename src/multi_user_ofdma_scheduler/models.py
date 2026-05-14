from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models import PAParams


@dataclass(frozen=True)
class PreparedJointOfdmaProblem:
    """Prepared OFDMA slot-scheduling input passed from space prep to a future solver.

    Steps:
    1. Keep the trusted per-user one-slot PHY operating menus from the batch artifact.
    2. Preserve the per-user frame payload requirements the future OFDMA scheduler must satisfy.
    3. Preserve the shared frame timing and slot PRB budget from the fixed radio geometry.
    """

    frame_n_slots: int
    t_slot_s: float
    prb_max: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]
    user_requirements: pd.DataFrame
    user_slot_spaces: dict[int, pd.DataFrame]

__all__ = [
    "PreparedJointOfdmaProblem",
]
