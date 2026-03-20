from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SingleUserRequest:
    """One single-user search request."""

    distance_m: float
    required_rate_bps: float
    path_loss_db: float | None = None


@dataclass(frozen=True)
class SingleUserSearchOptions:
    """Execution options that shape one single-user search run."""

    fast_mode: bool = False
    prb_step: int | None = None
    bandwidth_space_hz: tuple[float, ...] | None = None
    n_slots_on_space: tuple[int, ...] | None = None
    include_infeasible: bool = False
    parallel: bool = False
    max_workers: int | None = None
    use_cache: bool = True


@dataclass
class SingleUserProblem:
    """Built single-user problem with the request and resolved search inputs."""

    request: SingleUserRequest
    model_inputs: dict
    problem: object
    options: SingleUserSearchOptions


@dataclass
class SingleUserSearchResult:
    """Canonical single-user candidate ledger returned by the search module."""

    request: SingleUserRequest
    candidate_table: pd.DataFrame = field(default_factory=pd.DataFrame)
