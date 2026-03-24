import numpy as np

from .candidate_geometry import get_n_streams, occupied_bandwidth_hz, slot_level_re_counts


class SinrChainModel:
    """Explicit SINR and source-power solve chain for one candidate.

    The high-level solve entry point appears first, followed by the lower-level SINR
    building blocks it depends on. That keeps the reader on the main reasoning path
    before dropping into the algebraic details.
    """

    def solve_required_source_power_for_target(self, rho_req, deployment, rrc, candidate, pa):
        """Solve the minimum PA input power for an explicit required SINR target."""
        comps = self.build_sinr_terms(deployment, rrc, candidate, pa)
        denom_coeff = comps["a_num"] - rho_req * comps["b_dist"]
        if denom_coeff <= 0.0:
            return None

        ps_lb = max(rho_req * comps["c_noise"] / denom_coeff, 0.0)
        state_lb = self.effective_sinr_from_ps(ps_lb, deployment, rrc, candidate, pa, comps=comps)
        if state_lb["rho_eff"] >= rho_req:
            return self._build_source_power_solution(ps_lb, rho_req, state_lb, comps)

        bracket = self._expand_solution_bracket(
            rho_req,
            ps_lb,
            deployment,
            rrc,
            candidate,
            pa,
            comps,
        )
        if bracket is None:
            return None

        lo, hi = bracket
        ps_sol = self._bisect_required_source_power(
            rho_req,
            lo,
            hi,
            deployment,
            rrc,
            candidate,
            pa,
            comps,
        )
        state = self.effective_sinr_from_ps(ps_sol, deployment, rrc, candidate, pa, comps=comps)
        return self._build_source_power_solution(ps_sol, rho_req, state, comps)

    def build_sinr_terms(self, deployment, rrc, candidate, pa):
        """Precompute per-tone SINR coefficients reused across the solve."""
        n_streams = get_n_streams(candidate)
        g_l = self.channel_gain_linear(deployment)
        g_bf_linear = max(deployment.n_tx_chains / n_streams, 1)
        g_l *= g_bf_linear

        n0_w_per_hz = self.noise_density_w_per_hz(deployment)
        f_lna = 10 ** (deployment.lna_noise_figure_db / 10.0)
        sigma_v2_tone = f_lna * n0_w_per_hz * rrc.delta_f_hz

        b_occ = occupied_bandwidth_hz(rrc, candidate)
        k_active = self.compute_k_active_re(deployment, candidate)
        re_counts = slot_level_re_counts(deployment, candidate)

        a_num = g_l * pa.g_pa_eff_linear * deployment.g_phi / max(k_active, 1.0)
        b_dist = g_l * self.distortion_gain_kappa(pa, candidate) / max(k_active, 1.0)
        c_noise = sigma_v2_tone + deployment.sigma_q2 + deployment.sigma_phi2

        return {
            "g_l": g_l,
            "g_bf_linear": g_bf_linear,
            "n_streams": n_streams,
            "a_num": a_num,
            "b_dist": b_dist,
            "c_noise": c_noise,
            "sigma_v2_tone": sigma_v2_tone,
            "sigma_v2": sigma_v2_tone,
            "b_occ": b_occ,
            "k_active_re": k_active,
            "n_re_raw": re_counts["n_re_raw"],
            "n_re_data": re_counts["n_re_data"],
            "n_dmrs_re_per_prb": re_counts["n_dmrs_re_per_prb"],
            "n_pilot": re_counts["n_pilot"],
            "papr_ofdm_linear": self.ofdm_papr_linear(deployment),
        }

    def rho_from_ps(self, ps_w_total, deployment, rrc, candidate, pa, comps=None):
        """Compute achievable per-stream per-tone SINR from total source power."""
        if comps is None:
            comps = self.build_sinr_terms(deployment, rrc, candidate, pa)
        n_streams = max(int(comps.get("n_streams", get_n_streams(candidate))), 1)
        ps_stream_w = ps_w_total / n_streams
        denom = comps["b_dist"] * ps_stream_w + comps["c_noise"]
        if denom <= 0.0:
            return np.inf
        return comps["a_num"] * ps_stream_w / denom

    def effective_sinr_from_ps(self, ps_w_total, deployment, rrc, candidate, pa, comps=None):
        """Evaluate the end-to-end effective SINR for a candidate power point."""
        if comps is None:
            comps = self.build_sinr_terms(deployment, rrc, candidate, pa)
        # Effective-SINR chain: rho_raw -> sigma_e^2 = 1 / (rho_raw * N_pilot) -> rho_eff.
        rho_ach = self.rho_from_ps(ps_w_total, deployment, rrc, candidate, pa, comps=comps)
        sigma_e2 = self.channel_estimation_error(rho_ach, comps["n_pilot"])
        rho_eff = self.effective_sinr(rho_ach, sigma_e2)
        return {
            "rho_ach": rho_ach,
            "sigma_e2": sigma_e2,
            "n_pilot": comps["n_pilot"],
            "rho_eff": rho_eff,
        }

    @staticmethod
    def noise_density_w_per_hz(deployment):
        """Convert thermal noise density from dBm/Hz to W/Hz."""
        return 10 ** ((deployment.n0_dbm_per_hz - 30.0) / 10.0)

    @staticmethod
    def channel_gain_linear(deployment):
        """Compute large-scale linear channel gain from link-budget terms."""
        path_loss_linear = 10 ** (deployment.path_loss_db / 10.0)
        g_tx_linear = 10 ** (deployment.g_tx_db / 10.0)
        g_rx_linear = 10 ** (deployment.g_rx_db / 10.0)
        return (g_tx_linear * g_rx_linear) / path_loss_linear

    @staticmethod
    def ofdm_papr_linear(deployment):
        """Waveform-level OFDM PAPR used as a fixed parameter."""
        return 10 ** (deployment.papr_db / 10.0)

    @staticmethod
    def distortion_gain_kappa(pa, sched):
        """Distortion scaling is waveform-dependent and independent of modulation order."""
        backoff_linear = 10 ** (pa.backoff_db / 10.0)
        kappa = getattr(pa, "kappa_distortion", getattr(pa, "distortion_kappa", 0.0))
        return kappa / max(backoff_linear, 1e-12)

    def sigma_z2(self, pa, ps_w, sched):
        """Return in-band distortion power for a given PA input power."""
        return self.distortion_gain_kappa(pa, sched) * ps_w

    @staticmethod
    def compute_k_active_re(deployment, candidate):
        """Active resource elements used in the estimator model."""

        k_active = int(candidate.n_prb * 12)
        return int(np.clip(k_active, 1, deployment.dft_size_N))

    @staticmethod
    def channel_estimation_error(rho_ach, n_pilot):
        """Compute channel-estimation error variance from the pilot RE budget."""
        rho_ach = max(float(rho_ach), 1e-12)
        n_pilot = max(float(n_pilot), 1.0)
        return 1.0 / (rho_ach * n_pilot)

    @staticmethod
    def effective_sinr(rho_ach, sigma_e2):
        """Map raw SINR and estimation error to effective per-RE SINR."""
        return rho_ach / (1.0 + rho_ach * sigma_e2)

    def _expand_solution_bracket(self, rho_req, ps_lb, deployment, rrc, candidate, pa, comps):
        """Expand an upper bracket until the effective-SINR target becomes reachable."""
        lo = ps_lb
        hi = max(ps_lb * 2.0 + 1e-12, 1e-12)
        while self.effective_sinr_from_ps(hi, deployment, rrc, candidate, pa, comps=comps)["rho_eff"] < rho_req and hi < 1e9:
            lo = hi
            hi *= 2.0
        if hi >= 1e9:
            return None
        return lo, hi

    def _bisect_required_source_power(self, rho_req, lo, hi, deployment, rrc, candidate, pa, comps):
        """Refine the bracketed source-power solution by bisection."""
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.effective_sinr_from_ps(mid, deployment, rrc, candidate, pa, comps=comps)["rho_eff"] >= rho_req:
                hi = mid
            else:
                lo = mid
        return hi

    @staticmethod
    def _build_source_power_solution(ps_sol, rho_req, state, comps):
        """Assemble the solved source-power state exposed to downstream consumers."""
        return {
            "ps_min_w": ps_sol,
            "rho_req_linear": rho_req,
            "rho_achieved_linear": state["rho_eff"],
            "rho_ach_raw_linear": state["rho_ach"],
            "sigma_e2": state["sigma_e2"],
            "n_pilot": state["n_pilot"],
            "sigma_v2": comps["sigma_v2"],
            "b_occ": comps["b_occ"],
            "k_active_re": comps["k_active_re"],
            "n_re_raw": comps["n_re_raw"],
            "n_re_data": comps["n_re_data"],
            "n_dmrs_re_per_prb": comps["n_dmrs_re_per_prb"],
            "n_streams": comps["n_streams"],
            "g_bf_linear": comps["g_bf_linear"],
            "papr_ofdm_linear": comps["papr_ofdm_linear"],
        }
