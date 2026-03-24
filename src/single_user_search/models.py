from dataclasses import dataclass
from typing import Mapping

from radio_core import Candidate, DeploymentParams, PAParams, RRCParams, ResolvedModelInputs, ResolvedSearchShape


SingleUserSearchOptions = ResolvedSearchShape


@dataclass(frozen=True)
class SingleUserRequest:
    """One notebook or API request for a single-user deployment."""

    distance_m: float
    required_rate_bps: float


@dataclass(frozen=True)
class SearchCatalog:
    """Static search-space data shared across deployment evaluations."""

    pa_catalog: tuple[PAParams, ...]
    rrc_catalog: tuple[RRCParams, ...]
    search_shape: ResolvedSearchShape
    rrc_lookup: Mapping[tuple[int, int], RRCParams]


@dataclass(frozen=True)
class PreparedSingleUserContext:
    """Resolved single-user context split into static catalog and deployment."""

    request: SingleUserRequest
    model_inputs: ResolvedModelInputs
    deployment: DeploymentParams
    search_catalog: SearchCatalog
    static_catalog_key: str
    active_table_key: str

    @property
    def mcs_table(self) -> dict[int, dict[str, float]]:
        return self.model_inputs.mcs_table

    @property
    def search_shape(self) -> ResolvedSearchShape:
        return self.search_catalog.search_shape

    @property
    def options(self) -> ResolvedSearchShape:
        return self.search_shape

    @property
    def pa_catalog(self) -> tuple[PAParams, ...]:
        return self.search_catalog.pa_catalog

    @property
    def rrc_catalog(self) -> tuple[RRCParams, ...]:
        return self.search_catalog.rrc_catalog

    @property
    def rrc_lookup(self) -> Mapping[tuple[int, int], RRCParams]:
        return self.search_catalog.rrc_lookup


@dataclass(frozen=True)
class StaticCandidateSpec:
    """Cached static candidate metadata reused across deployment evaluation."""

    candidate_ordinal: int
    candidate: Candidate
    rate_ach_bps: float
    gamma_req_lin: float
    gamma_req_db: float


@dataclass(frozen=True)
class SingleUserStaticCandidateCatalog:
    """Cached static candidate catalog for one search-space shape."""

    candidates: tuple[StaticCandidateSpec, ...] = ()
