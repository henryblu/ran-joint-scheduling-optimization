from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from models import PAParams
from single_user_solver.models import PreparedSingleUserContext, SingleUserRequest


BATCH_USER_REQUIREMENT_COLUMNS = [
    "user_id",
    "required_rate_bps",
]


BATCH_USER_PARAMETER_SPACE_COLUMNS = [
    "pa_id",
    "bandwidth_hz",
    "n_prb",
    "layers",
    "mcs",
    "rate_active_bps",
    "p_dc_active_w",
    "p_out_total_w",
]


@dataclass(frozen=True)
class SingleUserScenario:
    """Prepared single-user study scenario used by notebook-facing helpers."""

    request: SingleUserRequest
    context: PreparedSingleUserContext


@dataclass(frozen=True)
class BatchUserParameterSpace:
    """Trusted batch single-user artifact consumed by the TDMA scheduler."""

    user_requirements: pd.DataFrame
    user_parameter_spaces: dict[int, pd.DataFrame]
    frame_n_slots: int
    n_tx_chains: int
    pa_catalog: tuple[PAParams, ...]


@dataclass
class SingleUserStudyResult:
    """Notebook-facing result tables for one frontier study sweep."""

    frontier_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanatory_configs: pd.DataFrame = field(default_factory=pd.DataFrame)
    pa_characteristics: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_space_view: pd.DataFrame = field(default_factory=pd.DataFrame)
