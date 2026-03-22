import numpy as np

from radio_core import Candidate, ModelOptions, Problem, RRCParams, SchedulerVars, SearchSpace


def build_discrete_problem(
    deployment,
    *,
    pa_catalog,
    scheduler_sweep,
    delta_f_hz_default,
    bandwidth_space,
    n_slots_on_space,
    prb_step,
):
    """Build the complete single-user search problem from resolved inputs."""

    rrc_catalog = _build_rrc_catalog(
        deployment,
        pa_catalog=pa_catalog,
        bandwidth_space=bandwidth_space,
        delta_f_hz_default=delta_f_hz_default,
        max_mcs=int(max(scheduler_sweep["mcs_space"])),
    )
    search_space = _build_search_space(
        deployment,
        scheduler_sweep=scheduler_sweep,
        n_slots_on_space=n_slots_on_space,
        prb_step=prb_step,
    )
    return Problem(
        deployment=deployment,
        pa_catalog=pa_catalog,
        rrc_catalog=rrc_catalog,
        search_space=search_space,
        options=ModelOptions(),
        rrc_lookup=_build_rrc_lookup(rrc_catalog),
    )


def iter_candidates(problem):
    """Yield scheduler candidates in the same order used by the search layer."""

    sp = problem.search_space
    for rrc in problem.rrc_catalog:
        n_prb_space = range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step))
        for n_prb in n_prb_space:
            for n_slots_on, layers, n_active_tx, mcs in _iter_valid_scheduler_configs(problem, rrc):
                yield Candidate(
                    pa_id=int(rrc.active_pa_id),
                    bwp_idx=int(rrc.bwp_index),
                    n_prb=int(n_prb),
                    n_slots_on=int(n_slots_on),
                    layers=int(layers),
                    n_active_tx=int(n_active_tx),
                    mcs=int(mcs),
                )


def count_candidates_for_rrc(problem, rrc):
    """Count raw scheduler combinations for one RRC envelope."""

    sp = problem.search_space
    n_prb_points = len(range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step)))
    scheduler_points = sum(1 for _ in _iter_valid_scheduler_configs(problem, rrc))
    return n_prb_points * scheduler_points


def build_scheduler_vars(candidate):
    """Create scheduler variables from one discrete candidate tuple."""

    return SchedulerVars(
        n_prb=int(candidate.n_prb),
        n_slots_on=int(candidate.n_slots_on),
        layers=int(candidate.layers),
        n_active_tx=int(candidate.n_active_tx),
        mcs=int(candidate.mcs),
    )


def resolve_candidate_context(problem, candidate):
    """Resolve the RRC envelope, scheduler vars, and PA for one candidate."""

    rrc = problem.rrc_lookup.get((int(candidate.pa_id), int(candidate.bwp_idx)))
    sched = build_scheduler_vars(candidate)
    pa = None if not (0 <= int(candidate.pa_id) < len(problem.pa_catalog)) else problem.pa_catalog[int(candidate.pa_id)]
    return rrc, sched, pa


def _build_rrc_catalog(deployment, *, pa_catalog, bandwidth_space, delta_f_hz_default, max_mcs):
    """Build the RRC/BWP envelopes explored by the search layer."""

    max_layers_default = deployment.n_tx_chains
    return [
        RRCParams(
            bwp_bw_hz=float(bwp_bw_hz),
            bwp_index=int(bwp_index),
            delta_f_hz=float(delta_f_hz_default),
            prb_max_bwp=int(np.floor(float(bwp_bw_hz) / (12.0 * float(delta_f_hz_default)))),
            max_layers=int(max_layers_default),
            max_mcs=int(max_mcs),
            active_pa_id=int(pa_id),
        )
        for bwp_index, bwp_bw_hz in enumerate(bandwidth_space)
        for pa_id in range(len(pa_catalog))
    ]


def _build_search_space(deployment, *, scheduler_sweep, n_slots_on_space, prb_step):
    """Define the discrete scheduler decision dimensions used by grid search."""

    resolved_n_slots_on_space = (
        list(n_slots_on_space)
        if n_slots_on_space is not None
        else list(range(1, deployment.n_slots_win + 1))
    )
    return SearchSpace(
        n_slots_on_space=resolved_n_slots_on_space,
        layers_space=list(scheduler_sweep["layers_space"]),
        n_active_tx_space=list(range(1, deployment.n_tx_chains + 1)),
        mcs_space=list(scheduler_sweep["mcs_space"]),
        prb_step=int(prb_step if prb_step is not None else scheduler_sweep["prb_step"]),
    )


def _build_rrc_lookup(rrc_catalog):
    """Build the direct RRC lookup used by the search layer."""

    return {
        (int(rrc.active_pa_id), int(rrc.bwp_index)): rrc
        for rrc in rrc_catalog
    }


def _iter_valid_scheduler_configs(problem, rrc):
    """Yield scheduler tuples that already satisfy structural limits."""

    sp = problem.search_space
    valid_mcs_space = list(_iter_valid_mcs(rrc, sp))
    for n_slots_on in sp.n_slots_on_space:
        for layers in _iter_valid_layers(rrc, sp):
            for n_active_tx in _iter_valid_active_tx_counts(problem, sp, layers):
                for mcs in valid_mcs_space:
                    yield n_slots_on, layers, n_active_tx, mcs


def _iter_valid_layers(rrc, search_space):
    """Yield layer counts admitted by the current RRC envelope."""

    for layers in search_space.layers_space:
        if 1 <= layers <= rrc.max_layers:
            yield layers


def _iter_valid_active_tx_counts(problem, search_space, layers):
    """Yield active-chain counts compatible with the current layer choice."""

    for n_active_tx in search_space.n_active_tx_space:
        if layers <= n_active_tx <= problem.deployment.n_tx_chains:
            yield n_active_tx


def _iter_valid_mcs(rrc, search_space):
    """Yield MCS values admitted by the current RRC envelope."""

    for mcs in search_space.mcs_space:
        if mcs <= rrc.max_mcs:
            yield mcs
