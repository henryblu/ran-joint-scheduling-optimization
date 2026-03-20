import numpy as np

from radio_core.model_types import Candidate, RRCParams, SchedulerVars, SearchSpace


class ResourceGridModel:
    """NR resource-grid accounting and scheduler candidate generation.

    The public methods are ordered from high-level search-space construction down to
    low-level accounting helpers so the file reads in the same order as the workflow.
    """

    def __init__(self, mcs_table):
        self.mcs_table = mcs_table

    def build_bwp_catalog(self, deployment, pa_catalog, bandwidth_space, delta_f_hz_default):
        """Build the RRC/BWP envelopes explored by the search layer."""
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
        """Define the discrete scheduler decision dimensions used by grid search."""
        if n_slots_on_space is None:
            n_slots_on_space = list(range(1, deployment.n_slots_win + 1))
        return SearchSpace(
            n_slots_on_space=list(n_slots_on_space),
            layers_space=list(sweep_settings["layers_space"]),
            n_active_tx_space=list(range(1, deployment.n_tx_chains + 1)),
            mcs_space=list(sweep_settings["mcs_space"]),
            prb_step=int(prb_step if prb_step is not None else sweep_settings["prb_step"]),
        )

    def compute_rate(self, deployment, rrc, sched):
        """Compute the average rate implied by one scheduler choice."""
        eta = self.mcs_table[sched.mcs]["eta"]
        re_counts = self.slot_level_re_counts(deployment, sched)
        bits_per_slot = re_counts["n_re_data"] * eta * sched.layers
        bits_in_window = sched.n_slots_on * bits_per_slot
        t_win = deployment.n_slots_win * deployment.t_slot_s
        return bits_in_window / t_win

    def count_candidates_for_rrc(self, problem, rrc):
        """Count raw scheduler combinations for one RRC envelope.

        The validity filtering is delegated to helper iterators so the counting logic
        matches the candidate generation logic without repeating nested rejection tests.
        """
        sp = problem.search_space
        n_prb_points = len(range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step)))
        scheduler_points = sum(1 for _ in self._iter_valid_scheduler_configs(problem, rrc))
        return n_prb_points * scheduler_points

    def enumerate_scheduler_candidates(self, problem):
        """Generate all valid scheduler candidates for each RRC envelope."""
        sp = problem.search_space
        for rrc in problem.rrc_catalog:
            n_prb_space = range(1, rrc.prb_max_bwp + 1, max(1, sp.prb_step))
            for n_prb in n_prb_space:
                for n_slots_on, layers, n_active_tx, mcs in self._iter_valid_scheduler_configs(problem, rrc):
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

    @staticmethod
    def slot_level_re_counts(deployment, sched):
        """Per-slot NR resource-element accounting with explicit pilot subtraction."""
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
        """Duty cycle from integer slot-on count inside the fixed observation window."""
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

    def _iter_valid_scheduler_configs(self, problem, rrc):
        """Yield scheduler tuples that already satisfy structural limits."""
        sp = problem.search_space
        valid_mcs_space = list(self._iter_valid_mcs(rrc, sp))
        for n_slots_on in sp.n_slots_on_space:
            for layers in self._iter_valid_layers(rrc, sp):
                for n_active_tx in self._iter_valid_active_tx_counts(problem, sp, layers):
                    for mcs in valid_mcs_space:
                        yield n_slots_on, layers, n_active_tx, mcs

    @staticmethod
    def _iter_valid_layers(rrc, search_space):
        """Yield layer counts admitted by the current RRC envelope."""
        for layers in search_space.layers_space:
            if 1 <= layers <= rrc.max_layers:
                yield layers

    @staticmethod
    def _iter_valid_active_tx_counts(problem, search_space, layers):
        """Yield active-chain counts compatible with the current layer choice."""
        for n_active_tx in search_space.n_active_tx_space:
            if layers <= n_active_tx <= problem.deployment.n_tx_chains:
                yield n_active_tx

    @staticmethod
    def _iter_valid_mcs(rrc, search_space):
        """Yield MCS values admitted by the current RRC envelope."""
        for mcs in search_space.mcs_space:
            if mcs <= rrc.max_mcs:
                yield mcs
