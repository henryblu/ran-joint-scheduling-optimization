from __future__ import annotations

"""Config-owned policy for final scheduler backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KMilpSolverConfig:
    """K-MILP solver policy resolved before a campaign run."""

    time_limit_s: float | None = None
    node_limit: int | None = None
    rel_gap: float | None = None
    max_users_per_slot: int | None = None
    k2_accept_rel_gap: float = 1e-3
    k2_cutoff_time_limit_s: float = 60.0
    dual_admitted_rows_per_user: int = 5
    dual_pattern_strategy: str = "admission"


K_MILP_SOLVER_CONFIG = KMilpSolverConfig()
ACTIVE_SNAPSHOT_SCOPE_ENV_VAR = "THESIS_ACTIVE_SNAPSHOT_SCOPE"


__all__ = [
    "ACTIVE_SNAPSHOT_SCOPE_ENV_VAR",
    "KMilpSolverConfig",
    "K_MILP_SOLVER_CONFIG",
]
