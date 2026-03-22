import numpy as np

from radio_core import ModelOptions, Problem

from .resource_grid import ResourceGridModel


class DownlinkProblemSpace:
    """Public downlink problem-space API for search-layer orchestration.

    The intended reading order is:
    - `build_problem` to materialize one search problem,
    - `iter_candidates` and `compute_candidate_rate` for downstream evaluation.
    """

    def __init__(self, mcs_table):
        self.mcs_table = dict(mcs_table)
        self.grid_model = ResourceGridModel(self.mcs_table)

    def build_problem(
        self,
        deployment,
        pa_catalog,
        scheduler_sweep,
        *,
        delta_f_hz_default,
        prb_step=None,
        bandwidth_space=None,
        n_slots_on_space=None,
    ):
        """Build the complete single-user search problem from deployment inputs.

        This method keeps the orchestration explicit:
        1. Resolve the bandwidth sweep that will define the BWP catalog.
        2. Build the RRC envelopes implied by those bandwidths and the PA catalog.
        3. Build the discrete scheduler search dimensions.
        4. Package everything into the `Problem` object consumed by the search layer.
        """

        resolved_bandwidth_space = self._resolve_bandwidth_space(scheduler_sweep, bandwidth_space)
        rrc_catalog = self.grid_model.build_bwp_catalog(
            deployment=deployment,
            pa_catalog=pa_catalog,
            bandwidth_space=resolved_bandwidth_space,
            delta_f_hz_default=delta_f_hz_default,
        )
        search_space = self.grid_model.build_search_space(
            deployment=deployment,
            sweep_settings=scheduler_sweep,
            prb_step=prb_step,
            n_slots_on_space=n_slots_on_space,
        )
        return Problem(
            deployment=deployment,
            pa_catalog=pa_catalog,
            rrc_catalog=rrc_catalog,
            search_space=search_space,
            options=ModelOptions(),
            rrc_lookup=self._build_rrc_lookup(rrc_catalog),
        )

    def iter_candidates(self, problem):
        """Yield scheduler candidates in the same order used by the search layer."""
        return self.grid_model.enumerate_scheduler_candidates(problem)

    def compute_candidate_rate(self, problem, candidate):
        """Compute the average achieved rate for a single discrete candidate."""
        rrc = self._find_rrc(problem, candidate)
        if rrc is None:
            raise ValueError(
                f"Candidate references unknown (pa_id, bwp_idx)=({candidate.pa_id}, {candidate.bwp_idx})"
            )
        sched = self.grid_model.build_scheduler_vars(candidate)
        return self.grid_model.compute_rate(problem.deployment, rrc, sched)

    @staticmethod
    def _resolve_bandwidth_space(scheduler_sweep, bandwidth_space):
        """Allow callers to override the bandwidth sweep without mutating presets."""
        if bandwidth_space is not None:
            return [float(v) for v in bandwidth_space]
        return [float(v) for v in scheduler_sweep["bandwidth_space_hz"]]

    @staticmethod
    def _build_rrc_lookup(rrc_catalog):
        """Build the direct RRC lookup used by candidate evaluation helpers."""
        return {
            (int(rrc.active_pa_id), int(rrc.bwp_index)): rrc
            for rrc in rrc_catalog
        }

    @staticmethod
    def _find_rrc(problem, candidate):
        """Return the RRC envelope referenced by a candidate, if it exists."""
        return problem.rrc_lookup.get((int(candidate.pa_id), int(candidate.bwp_idx)))
