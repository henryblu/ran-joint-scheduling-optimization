from __future__ import annotations

"""Internal OFDMA MILP oracle models."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from models import PAParams, PASwitchPolicy


@dataclass(frozen=True)
class MilpCandidateRow:
    """One finite user-row option in the direct slot-indexed MILP."""

    global_id: int
    user_id: int
    local_row_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    bits_per_slot: float
    p_out_total_w: float
    p_dc_active_w: float


@dataclass(frozen=True)
class OfdmaMilpProblem:
    """Trusted flattened problem consumed by the independent MILP oracle."""

    frame_n_slots: int
    t_slot_s: float
    prb_max: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]
    user_requirements: pd.DataFrame
    candidate_rows: tuple[MilpCandidateRow, ...]
    candidate_rows_by_user: dict[int, tuple[MilpCandidateRow, ...]]
    required_rate_by_user: dict[int, float]
    demand_bits_by_user: dict[int, float]
    switch_policy: PASwitchPolicy


@dataclass(frozen=True)
class PaCurveSegment:
    """One active PA curve segment for per-chain RF output and DC input."""

    pa_id: int
    segment_id: int
    left_p_out_w: float
    right_p_out_w: float
    left_dc_w: float
    right_dc_w: float


@dataclass(frozen=True)
class MilpModelSize:
    """Small model-size summary for logging and solver details."""

    variable_count: int
    binary_variable_count: int
    continuous_variable_count: int
    constraint_count: int
    nonzero_count: int


@dataclass(frozen=True)
class MilpVariableIndex:
    """Variable index maps used to build and decode one MILP attempt."""

    x: dict[tuple[int, int], int]
    z: dict[tuple[int, int], int]
    delta: dict[tuple[int, int], int]
    beta: dict[tuple[int, int, int], int]
    theta: dict[tuple[int, int, int], int]
    w: dict[int, int]
    v: dict[tuple[int, int], int]


@dataclass(frozen=True)
class MilpBuild:
    """Built SciPy MILP data plus decoding metadata."""

    c: Any
    integrality: Any
    bounds: Any
    constraints: Any
    variables: MilpVariableIndex
    model_size: MilpModelSize
    segments_by_pa: dict[int, tuple[PaCurveSegment, ...]]
    allowed_pa_ids: tuple[int, ...]
    build_elapsed_s: float


@dataclass(frozen=True)
class MilpAttemptResult:
    """One policy attempt result, including infeasible attempts."""

    attempt_name: str
    allowed_pa_ids: tuple[int, ...]
    success: bool
    solver_status: int
    solver_message: str
    objective_pwl_j: float | None
    objective_bound: float | None
    mip_gap: float | None
    solution: Any
    model_size: MilpModelSize
    segments_by_pa: dict[int, tuple[PaCurveSegment, ...]]
    variables: MilpVariableIndex
    build_elapsed_s: float
    solve_elapsed_s: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class OfdmaSlotPattern:
    """One feasible one-slot OFDMA allocation pattern for the count oracle."""

    pattern_id: int
    pa_id: int
    rows: tuple[MilpCandidateRow, ...]
    used_prbs: int
    aggregate_p_out_w: float
    dc_power_w: float
    slot_energy_j: float
    delivered_bits_by_user: dict[int, float]


@dataclass(frozen=True)
class PatternCountAttemptResult:
    """Result of the exact slot-pattern count MILP."""

    attempt_name: str
    allowed_pa_ids: tuple[int, ...]
    success: bool
    solver_status: int
    solver_message: str
    objective_j: float | None
    objective_bound: float | None
    mip_gap: float | None
    solution: Any
    patterns: tuple[OfdmaSlotPattern, ...]
    build_elapsed_s: float
    solve_elapsed_s: float
    model_size: MilpModelSize


__all__ = [
    "MilpAttemptResult",
    "MilpBuild",
    "MilpCandidateRow",
    "MilpModelSize",
    "MilpVariableIndex",
    "OfdmaMilpProblem",
    "OfdmaSlotPattern",
    "PaCurveSegment",
    "PatternCountAttemptResult",
]
