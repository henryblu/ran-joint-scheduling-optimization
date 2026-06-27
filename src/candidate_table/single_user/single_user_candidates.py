from itertools import product
from types import MappingProxyType

import numpy as np

from .single_user_models import Candidate, RRCParams, SearchCatalog


def build_search_catalog(
    *,
    model_inputs,
    pa_catalog,
    search_shape,
):
    """Build the static search catalog shared across deployments."""

    config = getattr(model_inputs, "config", model_inputs)
    rrc_catalog = []
    prb_max = int(np.floor(float(config.channel_bw_hz) / (12.0 * config.delta_f_hz)))
    for pa_id in range(len(pa_catalog)):
        rrc_catalog.append(
            RRCParams(
                channel_bw_hz=float(config.channel_bw_hz),
                delta_f_hz=config.delta_f_hz,
                prb_max=prb_max,
                max_layers=config.n_tx_chains,
                max_mcs=max(search_shape.mcs_space),
                active_pa_id=pa_id,
            )
        )

    frozen_rrc_catalog = tuple(rrc_catalog)
    rrc_lookup = MappingProxyType(
        {
            int(rrc.active_pa_id): rrc
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
        for n_prb, n_slots_on, layers, mcs in product(
            range(1, rrc.prb_max + 1, ss.prb_step),
            ss.n_slots_on_space,
            ss.layers_space,
            ss.mcs_space,
        ):
            yield Candidate(
                pa_id=rrc.active_pa_id,
                n_prb=n_prb,
                n_slots_on=n_slots_on,
                layers=layers,
                mcs=mcs,
            )


def resolve_candidate_context(search_catalog, candidate):
    """Resolve the static RRC envelope and PA for one candidate."""

    rrc = search_catalog.rrc_lookup[int(candidate.pa_id)]
    pa = search_catalog.pa_catalog[candidate.pa_id]
    return rrc, pa


def count_candidates_for_rrc(search_catalog, rrc):
    """Count scheduler combinations for one RRC envelope."""

    ss = search_catalog.search_shape
    n_prb_points = len(range(1, rrc.prb_max + 1, ss.prb_step))
    n_sched_points = (
        len(ss.n_slots_on_space)
        * len(ss.layers_space)
        * len(ss.mcs_space)
    )
    return n_prb_points * n_sched_points
