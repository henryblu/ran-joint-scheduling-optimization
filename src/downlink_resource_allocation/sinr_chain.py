import numpy as np


class SinrChainModel:
    """Explicit SINR and source-power solve chain for one candidate."""

    def __init__(self, grid_model, mcs_model):
        self.grid_model = grid_model
        self.mcs_model = mcs_model

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

    def build_sinr_terms(self, deployment, rrc, sched, pa):
        """Precompute per-tone SINR coefficients reused across the solve."""
        n_streams = self.grid_model.get_n_streams(sched)
        g_l = self.channel_gain_linear(deployment)
        g_bf_linear = max(sched.n_active_tx / n_streams, 1)
        g_l *= g_bf_linear

        n0_w_per_hz = self.noise_density_w_per_hz(deployment)
        f_lna = 10 ** (deployment.lna_noise_figure_db / 10.0)
        sigma_v2_tone = f_lna * n0_w_per_hz * rrc.delta_f_hz

        b_occ = self.grid_model.occupied_bandwidth_hz(rrc, sched)
        k_active = self.grid_model.compute_k_active_re(deployment, sched)
        re_counts = self.grid_model.slot_level_re_counts(deployment, sched)

        a_num = g_l * pa.g_pa_eff_linear * deployment.g_phi / max(k_active, 1.0)
        b_dist = g_l * self.distortion_gain_kappa(pa, sched) / max(k_active, 1.0)
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

    def rho_from_ps(self, ps_w_total, deployment, rrc, sched, pa, comps=None):
        """Compute achievable per-stream per-tone SINR from total source power."""
        if comps is None:
            comps = self.build_sinr_terms(deployment, rrc, sched, pa)
        n_streams = max(int(comps.get("n_streams", self.grid_model.get_n_streams(sched))), 1)
        ps_stream_w = ps_w_total / n_streams
        denom = comps["b_dist"] * ps_stream_w + comps["c_noise"]
        if denom <= 0.0:
            return np.inf
        return comps["a_num"] * ps_stream_w / denom

    def achievable_sinr(self, ps_w_total, deployment, rrc, sched, pa, comps=None):
        """Alias for raw per-tone SINR from power and system state."""
        return self.rho_from_ps(ps_w_total, deployment, rrc, sched, pa, comps=comps)

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

    def effective_sinr_from_ps(self, ps_w_total, deployment, rrc, sched, pa, comps=None):
        """End-to-end effective per-RE SINR evaluation for a candidate power."""
        if comps is None:
            comps = self.build_sinr_terms(deployment, rrc, sched, pa)
        rho_ach = self.achievable_sinr(ps_w_total, deployment, rrc, sched, pa, comps=comps)
        sigma_e2 = self.channel_estimation_error(rho_ach, comps["n_pilot"])
        rho_eff = self.effective_sinr(rho_ach, sigma_e2)
        return {
            "rho_ach": rho_ach,
            "sigma_e2": sigma_e2,
            "n_pilot": comps["n_pilot"],
            "rho_eff": rho_eff,
        }

    def solve_required_source_power(self, deployment, rrc, sched, pa):
        """Solve minimum PA input power that satisfies the candidate MCS requirement."""
        rho_req = self.mcs_model.get_required_sinr(deployment, sched)
        comps = self.build_sinr_terms(deployment, rrc, sched, pa)
        denom_coeff = comps["a_num"] - rho_req * comps["b_dist"]
        if denom_coeff <= 0.0:
            return None

        ps_lb = max(rho_req * comps["c_noise"] / denom_coeff, 0.0)
        state_lb = self.effective_sinr_from_ps(ps_lb, deployment, rrc, sched, pa, comps=comps)
        if state_lb["rho_eff"] >= rho_req:
            state = state_lb
            ps_sol = ps_lb
        else:
            lo = ps_lb
            hi = max(ps_lb * 2.0 + 1e-12, 1e-12)
            while self.effective_sinr_from_ps(hi, deployment, rrc, sched, pa, comps=comps)["rho_eff"] < rho_req and hi < 1e9:
                hi *= 2.0
            if hi >= 1e9:
                return None
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if self.effective_sinr_from_ps(mid, deployment, rrc, sched, pa, comps=comps)["rho_eff"] >= rho_req:
                    hi = mid
                else:
                    lo = mid
            ps_sol = hi
            state = self.effective_sinr_from_ps(ps_sol, deployment, rrc, sched, pa, comps=comps)

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
