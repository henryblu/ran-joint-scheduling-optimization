from dataclasses import dataclass
from typing import Mapping

from pa_models import PAParams
from radio_configs import RadioConfig
from radio_models import DeploymentParams


@dataclass(frozen=True)
class Candidate:
    """One discrete scheduler/RRC/PA candidate."""

    pa_id: int
    bwp_idx: int
    n_prb: int
    n_slots_on: int
    layers: int
    mcs: int


@dataclass(frozen=True)
class RRCParams:
    """RRC/BWP envelope for one bandwidth and PA pairing."""

    bwp_bw_hz: float
    bwp_index: int
    delta_f_hz: float
    prb_max_bwp: int
    max_layers: int
    max_mcs: int
    active_pa_id: int


@dataclass(frozen=True)
class SearchSpace:
    """Single-user search-owned discrete space metadata."""

    config: RadioConfig | None = None
    bandwidth_space_hz: tuple[float, ...] = ()
    n_slots_on_space: tuple[int, ...] = ()
    layers_space: tuple[int, ...] = ()
    mcs_space: tuple[int, ...] = ()
    prb_step: int = 1
    fingerprint: str = ""
    use_cache: bool = True


SingleUserSearchOptions = SearchSpace


@dataclass(frozen=True)
class SingleUserRequest:
    """One notebook or API request for a single-user deployment."""

    distance_m: float
    required_rate_bps: float


@dataclass(frozen=True)
class SearchCatalog:
    """Static search-space data shared across deployments."""

    pa_catalog: tuple[PAParams, ...]
    rrc_catalog: tuple[RRCParams, ...]
    search_shape: SearchSpace
    rrc_lookup: Mapping[tuple[int, int], RRCParams]


@dataclass(frozen=True)
class PreparedSingleUserContext:
    """Resolved single-user context split into static catalog and deployment."""

    request: SingleUserRequest
    model_inputs: RadioConfig
    deployment: DeploymentParams
    search_catalog: SearchCatalog
    static_catalog_key: str
    active_table_key: str

    @property
    def mcs_table(self) -> dict[int, dict[str, float]]:
        return {int(mcs): dict(row) for mcs, row in self.model_inputs.mcs_table.items()}

    @property
    def search_shape(self) -> SearchSpace:
        return self.search_catalog.search_shape

    @property
    def options(self) -> SearchSpace:
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
