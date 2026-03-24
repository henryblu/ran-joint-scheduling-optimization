from dataclasses import dataclass

from radio_core import Candidate, SchedulerVars


@dataclass(frozen=True)
class SingleUserRequest:
    """One single-user scheduler request."""

    distance_m: float
    required_rate_bps: float


@dataclass(frozen=True)
class SingleUserSearchOptions:
    """Execution options that shape one single-user candidate-space run."""
    prb_step: int
    bandwidth_space_hz: tuple[float, ...]
    n_slots_on_space: tuple[int, ...]
    use_cache: bool = True


@dataclass
class PreparedSingleUserContext:
    """Reusable single-user radio/search context for active-candidate evaluation."""

    model_inputs: dict
    deployment: object
    built_problem: object
    pa_catalog: list
    mcs_table: dict
    rrc_lookup: dict
    options: SingleUserSearchOptions


@dataclass(frozen=True)
class StaticCandidateSpec:
    """Scenario-invariant candidate metadata reused across user evaluations."""

    candidate_ordinal: int
    candidate: Candidate
    scheduler_vars: SchedulerVars
    pa_name: str
    bandwidth_hz: float
    alpha_f: float
    rate_ach_bps: float
    gamma_req_lin: float
    gamma_req_db: float


@dataclass(frozen=True)
class SingleUserStaticCandidateCatalog:
    """Cached static candidate catalog for one search-space shape."""

    candidates: tuple[StaticCandidateSpec, ...] = ()
