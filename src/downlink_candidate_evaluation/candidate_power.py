import numpy as np

from pa_models import average_pa_power

from .candidate_geometry import get_n_streams
from .mcs_requirements import McsRequirementModel
from .models import CandidatePowerResult
from .sinr_chain import SinrChainModel


class CandidatePowerModel:
    """Solve the required power chain for one resolved candidate."""

    def __init__(self, mcs_table):
        self.mcs_model = McsRequirementModel(mcs_table)
        self.sinr_model = SinrChainModel()

    def solve_candidate_power(self, deployment, rrc, candidate, pa, *, gamma_req_lin=None):
        """Compute feasibility, RF power, and PA DC power for one resolved candidate."""

        ok, reason = self._check_structural_limits(deployment, rrc, candidate)
        if not ok:
            return CandidatePowerResult(is_feasible=False, infeasibility_reason=str(reason))

        if gamma_req_lin is not None:
            resolved_gamma_req_lin = float(gamma_req_lin)
            resolved_gamma_req_db = (
                float(10.0 * np.log10(resolved_gamma_req_lin))
                if resolved_gamma_req_lin > 0.0
                else float("-inf")
            )
        else:
            gamma_req = self.mcs_model.get_required_sinr_table(deployment)[candidate.mcs]
            resolved_gamma_req_lin = float(gamma_req["rho_req_linear"])
            resolved_gamma_req_db = float(gamma_req["rho_req_db"])

        ps_solution = self.sinr_model.solve_required_source_power_for_target(
            resolved_gamma_req_lin,
            deployment,
            rrc,
            candidate,
            pa,
        )
        if ps_solution is None:
            return CandidatePowerResult(
                is_feasible=False,
                infeasibility_reason="sinr_infeasible",
                gamma_req_lin=resolved_gamma_req_lin,
                gamma_req_db=resolved_gamma_req_db,
            )

        rf_terms = self._compute_rf_terms(deployment, pa, candidate, ps_solution)
        ok, reason = self._check_rf_and_psd_limits(
            deployment,
            pa=pa,
            candidate=candidate,
            ps_solution=ps_solution,
            rf_terms=rf_terms,
        )
        if not ok:
            return CandidatePowerResult(
                is_feasible=False,
                infeasibility_reason=str(reason),
                gamma_req_lin=resolved_gamma_req_lin,
                gamma_req_db=resolved_gamma_req_db,
            )
        p_dc_avg_total_w = self._compute_average_dc_power(
            deployment,
            pa,
            candidate,
            float(rf_terms["p_out_ant_w"]),
        )
        return CandidatePowerResult(
            is_feasible=True,
            infeasibility_reason="ok",
            gamma_req_lin=resolved_gamma_req_lin,
            gamma_req_db=resolved_gamma_req_db,
            gamma_achieved=float(ps_solution["rho_achieved_linear"]),
            rho_ach_raw_linear=float(ps_solution["rho_ach_raw_linear"]),
            sigma_e2=float(ps_solution["sigma_e2"]),
            n_streams=int(ps_solution["n_streams"]),
            g_bf_linear=float(ps_solution["g_bf_linear"]),
            ps_total_w=float(rf_terms["ps_total_w"]),
            p_sig_out_total_w=float(rf_terms["p_sig_out_total_w"]),
            p_out_total_w=float(rf_terms["p_out_total_w"]),
            p_out_ant_w=float(rf_terms["p_out_ant_w"]),
            p_dc_avg_total_w=float(p_dc_avg_total_w),
        )

    @staticmethod
    def _check_structural_limits(deployment, rrc, candidate):
        """Reject candidates that do not fit the discrete deployment and RRC limits."""

        if rrc is None:
            return False, "rrc_not_found"
        if not 1 <= candidate.layers <= rrc.max_layers:
            return False, "invalid_layer_count"
        if candidate.mcs > rrc.max_mcs:
            return False, "invalid_mcs"
        if not 1 <= candidate.n_slots_on <= deployment.n_slots_win:
            return False, "invalid_slot_count"
        if not 1 <= candidate.n_prb <= rrc.prb_max_bwp:
            return False, "insufficient_res"
        return True, "ok"

    def _compute_rf_terms(self, deployment, pa, candidate, ps_solution):
        """Translate the solved source-power point into RF output powers."""

        n_streams = get_n_streams(candidate)
        ps_total_w = float(ps_solution["ps_min_w"])
        ps_stream_w = ps_total_w / n_streams
        p_dist_stream_w = self.sinr_model.sigma_z2(pa, ps_stream_w, candidate)
        p_sig_out_stream_w = pa.g_pa_eff_linear * ps_stream_w
        p_sig_out_total_w = n_streams * p_sig_out_stream_w
        p_out_total_w = p_sig_out_total_w + n_streams * p_dist_stream_w
        p_out_ant_w = p_out_total_w / deployment.n_tx_chains
        return {
            "ps_total_w": ps_total_w,
            "p_sig_out_total_w": float(p_sig_out_total_w),
            "p_out_total_w": float(p_out_total_w),
            "p_out_ant_w": float(p_out_ant_w),
        }

    @staticmethod
    def _check_rf_and_psd_limits(deployment, *, pa, candidate, ps_solution, rf_terms):
        """Reject solved candidates that violate PHY, PA, or PSD constraints."""

        p_out_total = float(rf_terms["p_out_total_w"])
        p_out_ant = float(rf_terms["p_out_ant_w"])
        psd = p_out_total / max(float(ps_solution["b_occ"]), 1e-30)

        if p_out_total < 0.0:
            return False, "nonphysical_negative_power"
        if p_out_ant > pa.p_max_w:
            return False, "per_chain_pa_cap"
        if pa.curve_pout_w is not None and len(pa.curve_pout_w) and p_out_ant > float(pa.curve_pout_w[-1]):
            return False, "interpolation_out_of_range"
        if p_out_total > deployment.n_tx_chains * pa.p_max_w:
            return False, "total_pa_cap"
        if deployment.use_psd_constraint and psd > deployment.psd_max_w_per_hz:
            return False, "psd_violation"
        return True, "ok"

    @staticmethod
    def _compute_average_dc_power(deployment, pa, candidate, p_out_ant_w):
        """Compute average PA DC draw including idle chains."""

        alpha_t = candidate.n_slots_on / deployment.n_slots_win
        return deployment.n_tx_chains * average_pa_power(pa, p_out_ant_w, alpha_t)
