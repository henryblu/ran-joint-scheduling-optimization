from types import MappingProxyType

import numpy as np

from .models import Candidate, RRCParams, SearchCatalog


def build_search_catalog(
    *,
    model_inputs,
    pa_catalog,
    search_shape,
):
    """Build the static search catalog shared across deployments."""

    config = getattr(model_inputs, "config", model_inputs)
    rrc_catalog = []
    for bwp_index, bwp_bw_hz in enumerate(search_shape.bandwidth_space_hz):
        prb_max = int(np.floor(bwp_bw_hz / (12.0 * config.delta_f_hz)))
        for pa_id in range(len(pa_catalog)):
            rrc_catalog.append(
                RRCParams(
                    bwp_bw_hz=bwp_bw_hz,
                    bwp_index=bwp_index,
                    delta_f_hz=config.delta_f_hz,
                    prb_max_bwp=prb_max,
                    max_layers=config.n_tx_chains,
                    max_mcs=max(search_shape.mcs_space),
                    active_pa_id=pa_id,
                )
            )

    frozen_rrc_catalog = tuple(rrc_catalog)
    rrc_lookup = MappingProxyType(
        {
            (rrc.active_pa_id, rrc.bwp_index): rrc
            for rrc in frozen_rrc_catalog
        }
    )
    return SearchCatalog(
        pa_catalog=tuple(pa_catalog),
        rrc_catalog=frozen_rrc_catalog,
        search_shape=search_shape,
        rrc_lookup=rrc_lookup,
    )


def iter_candidates(search_catalog):
    """Yield canonical candidate objects for the static search catalog."""

    ss = search_catalog.search_shape

    for rrc in search_catalog.rrc_catalog:
        for n_prb in range(1, rrc.prb_max_bwp + 1, ss.prb_step):
            for n_slots_on in ss.n_slots_on_space:
                for layers in ss.layers_space:
                    for mcs in ss.mcs_space:
                        yield Candidate(
                            pa_id=rrc.active_pa_id,
                            bwp_idx=rrc.bwp_index,
                            n_prb=n_prb,
                            n_slots_on=n_slots_on,
                            layers=layers,
                            mcs=mcs,
                        )


def resolve_candidate_context(search_catalog, candidate):
    """Resolve the static RRC envelope and PA for one candidate."""

    rrc = search_catalog.rrc_lookup[(candidate.pa_id, candidate.bwp_idx)]
    pa = search_catalog.pa_catalog[candidate.pa_id]
    return rrc, pa


def count_candidates_for_rrc(search_catalog, rrc):
    """Count scheduler combinations for one RRC envelope."""

    ss = search_catalog.search_shape
    n_prb_points = len(range(1, rrc.prb_max_bwp + 1, ss.prb_step))
    n_sched_points = (
        len(ss.n_slots_on_space)
        * len(ss.layers_space)
        * len(ss.mcs_space)
    )
    return n_prb_points * n_sched_points
