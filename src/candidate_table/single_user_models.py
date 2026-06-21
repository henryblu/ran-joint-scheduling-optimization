from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from models import DeploymentParams, PAParams, RadioConfig


@dataclass(frozen=True)
class Candidate:
    """One discrete scheduler/channel/PA candidate."""

    pa_id: int
    n_prb: int
    n_slots_on: int
    layers: int
    mcs: int


@dataclass(frozen=True)
class RRCParams:
    """Single-carrier resource envelope for one PA family."""

    channel_bw_hz: float
    delta_f_hz: float
    prb_max: int
    max_layers: int
    max_mcs: int
    active_pa_id: int


@dataclass(frozen=True)
class SearchSpace:
    """Single-user search-owned discrete space metadata."""

    config: RadioConfig | None = None
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
    rrc_lookup: Mapping[int, RRCParams]


@dataclass(frozen=True)
class PreparedSingleUserContext:
    """Resolved single-user context split into static catalog and deployment."""

    request: SingleUserRequest
    model_inputs: RadioConfig
    deployment: DeploymentParams
    search_catalog: SearchCatalog
    static_catalog_key: str

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
    def rrc_lookup(self) -> Mapping[int, RRCParams]:
        return self.search_catalog.rrc_lookup


@dataclass(frozen=True)
class StaticCandidateSpec:
    """Cached static candidate metadata reused across deployment evaluation."""

    candidate_ordinal: int
    candidate: Candidate
    bits_per_slot: float
    rate_ach_bps: float
    gamma_req_lin: float
