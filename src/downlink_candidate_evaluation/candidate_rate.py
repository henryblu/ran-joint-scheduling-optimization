from .models import CandidateRateResult


class CandidateRateModel:
    """Compute the achieved rate and resource usage of one resolved candidate."""

    def __init__(self, mcs_table):
        self.mcs_table = dict(mcs_table)

    def compute_candidate_rate(self, deployment, rrc, sched):
        """Compute the average achieved rate for one resolved scheduler state."""

        re_counts = self._slot_level_re_counts(deployment, sched)
        # Rate equation: R_ach = (N_slots_on * N_RE,data * eta(m) * L) / T_win.
        eta = self.mcs_table[sched.mcs]["eta"]
        bits_per_slot = re_counts["n_re_data"] * eta * sched.layers
        bits_in_window = sched.n_slots_on * bits_per_slot
        t_win = deployment.n_slots_win * deployment.t_slot_s
        return CandidateRateResult(
            rate_ach_bps=float(bits_in_window / t_win),
            n_re_data=float(re_counts["n_re_data"]),
            n_re_raw=float(re_counts["n_re_raw"]),
            n_pilot=float(re_counts["n_pilot"]),
            # Occupied-bandwidth equation: B_occ = N_PRB * 12 * Delta f.
            b_occ_hz=float(sched.n_prb * 12.0 * rrc.delta_f_hz),
        )

    @staticmethod
    def _slot_level_re_counts(deployment, sched):
        """Per-slot NR resource-element accounting with explicit pilot subtraction."""

        n_re_raw = sched.n_prb * 12 * deployment.n_sym_data
        n_dmrs_re_per_prb = 12 * deployment.n_dmrs_sym
        n_pilot = sched.n_prb * n_dmrs_re_per_prb
        n_re_data = max(n_re_raw - n_pilot, 1.0)
        return {
            "n_re_raw": float(n_re_raw),
            "n_pilot": float(n_pilot),
            "n_re_data": float(n_re_data),
        }
