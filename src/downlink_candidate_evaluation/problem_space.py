from itertools import islice

import numpy as np
import pandas as pd

from radio_core import ModelOptions, Problem

from .resource_grid import ResourceGridModel


class DownlinkProblemSpace:
    """Public downlink problem-space API for search-layer orchestration.

    The intended reading order is:
    - `build_problem` to materialize one search problem,
    - `describe_problem` and `estimate_search_space` for inspection,
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
        fast_mode=False,
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
            options=ModelOptions(fast_mode=bool(fast_mode)),
        )

    def estimate_search_space(self, problem, scenario_count=1):
        """Summarize the raw combinatorial size before feasibility pruning."""
        per_pa_counts = []
        for pa_id in range(len(problem.pa_catalog)):
            rrc_space = [rrc for rrc in problem.rrc_catalog if rrc.active_pa_id == pa_id]
            per_pa_counts.append(
                sum(self.grid_model.count_candidates_for_rrc(problem, rrc) for rrc in rrc_space)
            )
        raw_configs_per_scenario = sum(per_pa_counts)
        return {
            "pa_count": int(len(problem.pa_catalog)),
            "scenario_count": int(scenario_count),
            "raw_configs_per_pa_per_scenario": int(per_pa_counts[0]) if per_pa_counts else 0,
            "raw_configs_per_scenario": int(raw_configs_per_scenario),
            "raw_total_configs": int(raw_configs_per_scenario * scenario_count),
            "n_slots_on_values": len(problem.search_space.n_slots_on_space),
            "layers_values": len(problem.search_space.layers_space),
            "n_active_tx_values": len(problem.search_space.n_active_tx_space),
            "mcs_values": len(problem.search_space.mcs_space),
            "prb_step": int(problem.search_space.prb_step),
        }

    def describe_problem(self, problem, scenario_count=1):
        """Return notebook-friendly tables that explain what problem was built."""
        deployment_summary = pd.DataFrame(
            [
                {
                    "distance_m": float(problem.deployment.distance_m),
                    "path_loss_db": float(problem.deployment.path_loss_db),
                    "fc_hz": float(problem.deployment.fc_hz),
                    "n_tx_chains": int(problem.deployment.n_tx_chains),
                    "n_slots_win": int(problem.deployment.n_slots_win),
                    "delta_f_hz": float(problem.rrc_catalog[0].delta_f_hz) if problem.rrc_catalog else np.nan,
                }
            ]
        )
        rrc_catalog = pd.DataFrame(
            [
                {
                    "pa_id": int(rrc.active_pa_id),
                    "bwp_idx": int(rrc.bwp_index),
                    "bandwidth_hz": float(rrc.bwp_bw_hz),
                    "prb_max_bwp": int(rrc.prb_max_bwp),
                    "max_layers": int(rrc.max_layers),
                    "max_mcs": int(rrc.max_mcs),
                }
                for rrc in problem.rrc_catalog
            ]
        ).sort_values(["pa_id", "bandwidth_hz"]).reset_index(drop=True)
        search_space_summary = pd.DataFrame([self.estimate_search_space(problem, scenario_count=scenario_count)])
        return {
            "deployment_summary": deployment_summary,
            "rrc_catalog": rrc_catalog,
            "search_space_summary": search_space_summary,
        }

    def iter_candidates(self, problem):
        """Yield scheduler candidates in the same order used by the search layer."""
        return self.grid_model.enumerate_scheduler_candidates(problem)

    def preview_candidates(self, problem, limit=5):
        """Return the first few candidates as a DataFrame for quick inspection."""
        preview_rows = [candidate.__dict__ for candidate in islice(self.iter_candidates(problem), limit)]
        return pd.DataFrame(preview_rows)

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
    def _find_rrc(problem, candidate):
        """Return the RRC envelope referenced by a candidate, if it exists."""
        return next(
            (
                rrc
                for rrc in problem.rrc_catalog
                if rrc.active_pa_id == candidate.pa_id and rrc.bwp_index == candidate.bwp_idx
            ),
            None,
        )
