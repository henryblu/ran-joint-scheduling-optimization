from .eval_geometry import occupied_bandwidth_hz, slot_level_re_counts
from .eval_models import CandidateRateResult


class CandidateRateModel:
    """Compute the achieved rate and resource usage of one resolved candidate."""

    def __init__(self, mcs_table):
        self.mcs_table = dict(mcs_table)

    def compute_candidate_rate(self, deployment, rrc, candidate):
        """Compute the average achieved rate for one resolved scheduler state."""

        re_counts = slot_level_re_counts(deployment, candidate)
        # Rate equation: R_ach = (N_slots_on * N_RE,data * eta(m) * L) / T_frame.
        eta = self.mcs_table[candidate.mcs]["eta"]
        bits_per_slot = re_counts["n_re_data"] * eta * candidate.layers
        bits_in_frame = candidate.n_slots_on * bits_per_slot
        frame_duration_s = deployment.frame_n_slots * deployment.t_slot_s
        return CandidateRateResult(
            bits_per_slot=float(bits_per_slot),
            rate_ach_bps=float(bits_in_frame / frame_duration_s),
            n_re_data=float(re_counts["n_re_data"]),
            n_re_raw=float(re_counts["n_re_raw"]),
            n_pilot=float(re_counts["n_pilot"]),
            # Occupied-bandwidth equation: B_occ = N_PRB * 12 * Delta f.
            b_occ_hz=occupied_bandwidth_hz(rrc, candidate),
        )
