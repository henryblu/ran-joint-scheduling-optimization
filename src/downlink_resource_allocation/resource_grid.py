import numpy as np

from radio_core.model_types import Candidate, RRCParams, SchedulerVars, SearchSpace


class ResourceGridModel:
    """NR resource-grid accounting and scheduler candidate generation."""

    def __init__(self, mcs_table):
        self.mcs_table = mcs_table

    def build_bwp_catalog(self, deployment, pa_catalog, bandwidth_space, delta_f_hz_default):
        """Build BWP/RRC envelopes for each bandwidth and PA combination."""
        max_layers_default = deployment.n_tx_chains
        max_mcs_default = max(self.mcs_table)
        catalog = []
        for bwp_index, bwp_bw_hz in enumerate(bandwidth_space):
            for pa_id in range(len(pa_catalog)):
                catalog.append(
                    RRCParams(
                        bwp_bw_hz=float(bwp_bw_hz),
                        bwp_index=int(bwp_index),
                        delta_f_hz=float(delta_f_hz_default),
                        prb_max_bwp=int(np.floor(float(bwp_bw_hz) / (12.0 * float(delta_f_hz_default)))),
                        max_layers=int(max_layers_default),
                        max_mcs=int(max_mcs_default),
                        active_pa_id=int(pa_id),
                    )
                )
        return catalog

    def build_search_space(self, deployment, sweep_settings, prb_step=None, n_slots_on_space=None):
        """Define the discrete scheduler decision space for grid search."""
        if n_slots_on_space is None:
            n_slots_on_space = list(range(1, deployment.n_slots_win + 1))
        return SearchSpace(
            n_slots_on_space=list(n_slots_on_space),
            layers_space=list(sweep_settings["layers_space"]),
            n_active_tx_space=list(range(1, deployment.n_tx_chains + 1)),
            mcs_space=list(sweep_settings["mcs_space"]),
            prb_step=int(prb_step if prb_step is not None else sweep_settings["prb_step"]),
        )

    @staticmethod
    def slot_level_re_counts(deployment, sched):
        """Per-slot NR RE accounting with explicit pilot subtraction."""
        n_re_raw = sched.n_prb * 12 * deployment.n_sym_data
        n_dmrs_re_per_prb = 12 * deployment.n_dmrs_sym
        n_pilot = sched.n_prb * n_dmrs_re_per_prb
        n_re_data = max(n_re_raw - n_pilot, 1.0)
        return {
            "n_re_raw": float(n_re_raw),
            "n_dmrs_re_per_prb": float(n_dmrs_re_per_prb),
            "n_pilot": float(n_pilot),
            "n_re_data": float(n_re_data),
        }

    @staticmethod
    def scheduler_duty_cycle(deployment, sched):
        """Duty cycle from integer slot-on count inside fixed window."""
        return sched.n_slots_on / deployment.n_slots_win

    @staticmethod
    def occupied_bandwidth_hz(rrc, sched):
        """Occupied bandwidth from allocated PRBs."""
        return sched.n_prb * 12.0 * rrc.delta_f_hz

    @staticmethod
    def compute_k_active_re(deployment, sched):
        """Active resource elements used in the estimator model."""
        k_active = int(sched.n_prb * 12)
        return int(np.clip(k_active, 1, deployment.dft_size_N))

    @staticmethod
    def get_n_streams(sched):
        """Current single-user stream count."""
        return max(int(getattr(sched, "layers", 1)), 1)

    def compute_rate(self, deployment, rrc, sched):
        """Average rate in bps using explicit per-slot NR data RE accounting."""
        eta = self.mcs_table[sched.mcs]["eta"]
        re_counts = self.slot_level_re_counts(deployment, sched)
        bits_per_slot = re_counts["n_re_data"] * eta * sched.layers
        bits_in_window = sched.n_slots_on * bits_per_slot
        t_win = deployment.n_slots_win * deployment.t_slot_s
        return bits_in_window / t_win

    def count_candidates_for_rrc(self, problem, rrc):
        """Count the raw scheduler combinations generated for one RRC envelope."""
        sp = problem.search_space
        n_prb_points = len(range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step)))
        scheduler_points = 0
        for _n_slots_on in sp.n_slots_on_space:
            for layers in sp.layers_space:
                if layers < 1 or layers > rrc.max_layers:
                    continue
                for n_active_tx in sp.n_active_tx_space:
                    if n_active_tx < layers or n_active_tx > problem.deployment.n_tx_chains:
                        continue
                    for mcs in sp.mcs_space:
                        if mcs > rrc.max_mcs:
                            continue
                        scheduler_points += 1
        return n_prb_points * scheduler_points

    def enumerate_scheduler_candidates(self, problem):
        """Generate all scheduler candidates constrained by each RRC/BWP envelope."""
        sp = problem.search_space
        for rrc in problem.rrc_catalog:
            n_prb_space = range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step))
            for n_prb in n_prb_space:
                for n_slots_on in sp.n_slots_on_space:
                    for layers in sp.layers_space:
                        if layers < 1 or layers > rrc.max_layers:
                            continue
                        for n_active_tx in sp.n_active_tx_space:
                            if n_active_tx < layers or n_active_tx > problem.deployment.n_tx_chains:
                                continue
                            for mcs in sp.mcs_space:
                                if mcs > rrc.max_mcs:
                                    continue
                                yield Candidate(
                                    pa_id=rrc.active_pa_id,
                                    bwp_idx=rrc.bwp_index,
                                    n_prb=int(n_prb),
                                    n_slots_on=int(n_slots_on),
                                    layers=int(layers),
                                    n_active_tx=int(n_active_tx),
                                    mcs=int(mcs),
                                )

    @staticmethod
    def build_scheduler_vars(candidate):
        """Create a scheduler variable object from a candidate tuple."""
        return SchedulerVars(
            n_prb=int(candidate.n_prb),
            n_slots_on=int(candidate.n_slots_on),
            layers=int(candidate.layers),
            n_active_tx=int(candidate.n_active_tx),
            mcs=int(candidate.mcs),
        )
